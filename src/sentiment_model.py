from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from ray import tune
from sklearn.metrics import classification_report
import torch
import json


class SentimentModel:
    def __init__(
        self,
        model_name: str ='cardiffnlp/twitter-roberta-base-sentiment-latest'
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

    def predict(self, text: str) -> int:
        """
        :param text: text to predict sentiment
        :return: prediction
        """
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**tokenized_text)

        prediction = torch.argmax(outputs.logits, dim=-1).item()

        return self.model.config.id2label[prediction]

    def evaluate(self, test: Dataset) -> None:
        """
        evaluate the model based on test dataset
        :param test: Dataset object to test
        :return: None
        """
        gold_labels = []
        predictions = []

        # predict in a loop
        for sample in test:
            text = sample["text"]
            gold_label = sample["label"]

            tokenized_text = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True
            )

            with torch.no_grad():
                outputs = self.model(**tokenized_text)

            prediction = torch.argmax(outputs.logits, dim=-1).item()

            gold_labels.append(gold_label)
            predictions.append(prediction)

        # metrics report
        report = classification_report(
            y_true=gold_labels,
            y_pred=predictions
        )
        print(report)

    def load_model(self, is_fine_tuned: bool) -> None:
        """
        load model
        :param is_fine_tuned: if true, load the fine-tuned model
        :return: None
        """
        if is_fine_tuned:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                '../models/ckpt_sentiment/best_model'
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3
            )

    def fine_tune(self, dataset: DatasetDict) -> None:
        """
        fine-tune the model by parameter search
        :param dataset: Dataset containing train and validation
        :return: None
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        # tokenize_data
        train = dataset['train'].map(self._tokenize, batched=True)
        validation = dataset['validation'].map(self._tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir="../models/ckpt_sentiment",
            evaluation_strategy="epoch",
            logging_dir="./logs_sentiment",
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = Trainer(
            model_init=self._model_init,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train,
            eval_dataset=validation
        )

        search_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-4),
            "num_train_epochs": tune.choice(list(range(1, 4))),
            "per_device_train_batch_size": tune.choice([4, 8, 32])
        }
        best_run = trainer.hyperparameter_search(
            direction="minimize",
            hp_space=lambda x: search_space,
            n_trials=5,
            backend="ray",
            resources_per_trial={"cpu": 1, "gpu": 0}
        )

        # save hyperparameters
        with open('../models/ckpt_sentiment/best_run_hyperparameters.json', 'w') as f:
            json.dump(best_run.hyperparameters, f)

        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        # no need to re-evaluate
        setattr(trainer.args, "evaluation_strategy", 'no')

        trainer.train()
        # save
        trainer.save_model('../models/ckpt_sentiment/best_model')
        # self.model = trainer.model

    def _tokenize(self, batch):
        """
        tokenize a batch
        """
        return self.tokenizer(
            batch["text"], padding=False, truncation=True
        )

    def _model_init(self):
        """
        model_init for Trainer()
        """
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
