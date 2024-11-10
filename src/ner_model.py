from datasets import Dataset, DatasetDict
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from ray import tune
from sklearn.metrics import classification_report
import torch
import json
from TorchCRF import CRF


class NERModel:
    def __init__(self, model_name: str = 'tner/roberta-base-tweetner7-all'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )
        self.tokenizer_word_list = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            add_prefix_space=True
        )
        self.num_labels = None
        self.crf = None

    def predict(self, text: str, use_crf: bool = True) -> dict:
        """
        predict entities with CRF

        :param text: text to predict
        :param use_crf: apply CRF layer or not
        """
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        outputs = None
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_text["input_ids"],
                attention_mask=tokenized_text["attention_mask"]
            )

        predictions = None
        # Apply CRF
        if use_crf:
            predictions = self.crf.viterbi_decode(
                outputs.logits,
                mask=tokenized_text["attention_mask"]
            )
        else:
            predictions = torch.argmax(outputs.logits, dim=-1).tolist()

        # Convert predicted token class IDs back to labels
        predicted_tokens_classes = [self.model.config.id2label[t] for t in predictions[0]]
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"].squeeze())

        entities = NERModel.merge_bio_tags(tokens, predicted_tokens_classes)
        return {
            "entities": NERModel.find_entity_positions(entities, text)
        }

    @staticmethod
    def merge_bio_tags(
            tokens: list[str],
            predicted_tokens_classes: list[str]
    ) -> list[dict]:
        """
        Merges BIO tags into entity types;
        May contain duplicates due to multiple occurrence of an entity

        :param tokens: tokenized text
        :param predicted_tokens_classes: classes corresponding to tokens

        :return: list of {label: entity}
        """
        entities = []
        current_entity = []
        current_label = None

        for token, label in zip(tokens, predicted_tokens_classes):
            # Start of a new word
            clean_token = token.strip()
            is_new_word = False
            if token.startswith('Ġ'):
                clean_token = token.replace('Ġ', '').strip()
                is_new_word = True

            if not clean_token:
                continue

            # Outside entity
            if label == "O":
                if current_entity:
                    # Join the current entity tokens (without extra spaces for subwords)
                    entity_text = ''.join(current_entity).strip()
                    entities.append({
                        "label": current_label,
                        "entity": entity_text
                    })
                    current_entity = []
                    current_label = None
            else:
                # Get label without B- or I-
                label_type = label[2:]

                # Start a new entity
                if label.startswith("B-"):
                    # Finish the prev entity
                    if current_entity:
                        entity_text = ''.join(current_entity).strip()
                        entities.append({
                            "label": current_label,
                            "entity": entity_text
                        })

                    current_entity = [clean_token] if not is_new_word else [' ' + clean_token]
                    current_label = label_type
                elif label.startswith("I-") and current_label == label_type:
                    current_entity.append(clean_token if not is_new_word else ' ' + clean_token)

        # Deal with remaining
        if current_entity:
            entity_text = ''.join(current_entity).strip()
            entities.append({
                "label": current_label,
                "entity": entity_text
            })

        return entities

    @staticmethod
    def find_entity_positions(entities: list[dict], text: str) -> list[dict]:
        """
        Finds the start and end character positions of entities in the original text.

        :param entities: list of dictionaries with entity label and text
        :param text: the original text string
        :return: list of dictionaries with label, entity, start_char, and end_char
        """
        result = []
        masked_text = text

        for entity in entities:
            entity_name = entity['entity']
            entity_len = len(entity_name)

            # Get the first occurrence of the entity in the masked text
            start_char = masked_text.find(entity_name)

            # Entity not found (it's possible in rare cases due to CRF)
            if start_char == -1:
                continue

            end_char = start_char + entity_len

            # Add the entity with its start and end positions
            result.append({
                "label": entity['label'],
                "entity": entity_name,
                "start_char": start_char,
                "end_char": end_char
            })

            # Mask found entity to avoid multiple find on the same location
            masked_text = masked_text[:start_char] + " " * entity_len + masked_text[end_char:]

        return result

    def evaluate(self, test: Dataset, use_crf: bool = True) -> None:
        """
        evaluate the model based on test dataset

        :param test: Dataset object to test
        :param use_crf: use CRF layer or not
        :return: None
        """
        gold_labels_list = []
        predictions_list = []

        # predict in a loop
        for sample in test:
            text = sample["text"]
            word_level_gold = sample["label"]

            tokenized_text = self._tokenize(
                batch={
                    "text": [text],
                    "label": [word_level_gold]
                },
                label2id=self.model.config.label2id
            )

            outputs = None
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokenized_text["input_ids"],
                    attention_mask=tokenized_text["attention_mask"]
                )

            predictions = None
            if use_crf:
                # Apply CRF
                predictions = self.crf.viterbi_decode(
                    outputs.logits,
                    mask=tokenized_text["attention_mask"]
                )
            else:
                predictions = torch.argmax(outputs.logits, dim=-1).tolist()

            word_level_predictions = []

            previous_word_idx = None
            # only one text in the batch
            for idx, word_idx in enumerate(tokenized_text.word_ids(batch_index=0)):
                # Skip special tokens
                if word_idx is None:
                    continue
                # This is the first token of a word
                if word_idx != previous_word_idx:
                    word_level_predictions.append(
                        self.model.config.id2label[predictions[0][idx]]
                    )
                previous_word_idx = word_idx

            predictions_list.extend(word_level_predictions)
            gold_labels_list.extend(word_level_gold)

        report = classification_report(gold_labels_list, predictions_list)
        print(report)

    def fine_tune(self, dataset: DatasetDict, n_trials: int = 5) -> None:
        """
        fine-tune the model by parameter search

        :param dataset: Dataset containing train and validation
        :param n_trials: number of configs to try in param search

        :return: None
        """
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name
        )

        # Tokenize data
        train = dataset['train'].map(
            self._tokenize,
            batched=True,
            fn_kwargs={"label2id": model.config.label2id}
        )
        validation = dataset['validation'].map(
            self._tokenize,
            batched=True,
            fn_kwargs={"label2id": model.config.label2id}
        )

        # Set up args
        training_args = TrainingArguments(
            output_dir="../models/ckpt_ner",
            evaluation_strategy="epoch",
            logging_dir="./logs_ner",
            logging_steps=10,
            save_strategy="epoch",
        )

        # Initiate trainer
        trainer = Trainer(
            model_init=self._model_init,
            data_collator=NERModel.custom_data_collator,
            args=training_args,
            train_dataset=train,
            eval_dataset=validation
        )

        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-5),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-4),
                "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
            }

        best_run = trainer.hyperparameter_search(
            direction="minimize",
            hp_space=optuna_hp_space,
            n_trials=n_trials,
            backend="optuna",
            reuse_actors=False
        )

        # save hyperparameters
        with open('../models/ckpt_ner/best_run_hyperparameters.json', 'w') as f:
            json.dump(best_run.hyperparameters, f)

        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        # no need to re-evaluate
        setattr(trainer.args, "evaluation_strategy", 'no')

        trainer.train()
        # save
        trainer.save_model('../models/ckpt_ner/best_model')
        # self.model = trainer.model

    def load_model(self, is_fine_tuned: bool) -> None:
        """
        load model & setting up self.num_labels and self.crf

        :param is_fine_tuned: if true, load the fine-tuned model
        :return: None
        """
        if is_fine_tuned:
            self.model = AutoModelForTokenClassification.from_pretrained(
                '../models/ckpt_ner/best_model'
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name
            )

        self.num_labels = self.model.config.num_labels
        self.crf = CRF(num_labels=self.num_labels)

    @staticmethod
    def custom_data_collator(features) -> dict:
        """
        custom_data_collator for fine-tuning
        """
        return {
            'input_ids': torch.tensor([f['input_ids'] for f in features]),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features]),
            'labels': torch.tensor([f['labels'] for f in features])
        }

    def _tokenize(self, batch, label2id):
        """
        tokenize a batch; text should be a list of word tokens
        """
        tokenized_inputs = self.tokenizer_word_list(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True
        )

        labels_list = []
        for i, label in enumerate(batch["label"]):
            # Get word IDs for current text
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                # Handle special tokens (e.g., <s>, </s>) & padding
                if word_idx is None or word_idx >= len(label):
                    label_ids.append(-100)
                else:
                    if word_idx != previous_word_idx:  # First token of a word
                        # Use the original label for the first token (B-ENTITY)
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        # Assign I-ENTITY to subwords
                        # prev 0 and I-ENTITY would remain the same
                        label_ids.append(label2id[label[previous_word_idx].replace("B-", "I-")])
                    previous_word_idx = word_idx
            labels_list.append(label_ids)

        tokenized_inputs["labels"] = torch.tensor(labels_list)
        return tokenized_inputs

    def _model_init(self):
        """
        model_init for Trainer()
        """
        return AutoModelForTokenClassification.from_pretrained(
            self.model_name
        )
