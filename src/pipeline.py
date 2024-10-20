import spacy
from spacy.tokens import Span
from spacy.language import Language
from spacy.tokens.doc import Doc
from ner_model import NERModel
from sentiment_model import SentimentModel
from collections import defaultdict
import numpy as np
import json

# handle sentiment
if not Span.has_extension("sentiment"):
    Span.set_extension(
        name="sentiment",
        default=None
    )

# handle alias
if not Span.has_extension("entity_id_"):
    Span.set_extension(
        name="entity_id_",
        default=None
    )


class PipelineNotBuiltError(Exception):
    pass


class EntitySentimentModel:
    def __init__(
        self,
        ner_model_name: str = 'tner/roberta-base-tweetner7-all',
        sentiment_model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        is_fine_tuned_ner: bool = False,
        is_fine_tuned_sentiment: bool = False,
        use_crf: bool = True,
        half_window_size: int = 20
    ):
        """
        :param ner_model_name: model name for NER
        :param sentiment_model_name: model name for Sentiment Analysis
        :param is_fine_tuned_ner: Use local fine-tuned version or not
        :param is_fine_tuned_sentiment: Use local fine-tuned version or not
        :param use_crf: use CRF layer or not
        :param half_window_size: half_window_size for the window to get relevant context
        """
        # NER
        self.ner_model_name = ner_model_name
        self.use_fine_tuned_ner = is_fine_tuned_ner
        self.ner_model = NERModel(model_name=ner_model_name)
        self.ner_model.load_model(is_fine_tuned=is_fine_tuned_ner)
        self.use_crf = use_crf
        self.half_window_size = half_window_size

        # Sentiment
        self.sentiment_model_name = sentiment_model_name
        self.is_fine_tuned_sentiment = is_fine_tuned_sentiment
        self.sentiment_model = SentimentModel(model_name=sentiment_model_name)
        self.sentiment_model.load_model(is_fine_tuned_sentiment)

        # pipeline
        self.nlp = None
        self.is_pipeline_built = False
        self.use_entity_ruler = None

    def build_pipeline(
        self,
        use_entity_ruler: bool = True,
        pattern_paths=None,
    ) -> None:
        """
        build the pipeline
        """
        if pattern_paths is None:
            pattern_paths = []

        # set self.use_entity_ruler
        self.use_entity_ruler = use_entity_ruler

        self.nlp = spacy.blank("en")
        self.add_ner_model()
        if use_entity_ruler:
            self.add_entity_ruler(pattern_paths)
            self.add_alias_handler()

        self.add_sentiment_model()

        self.is_pipeline_built = True

    def predict(self, text: str) -> list[dict]:
        """
        :param text: tweet context
        :return: list of entity predictions
        """
        if not self.is_pipeline_built:
            raise PipelineNotBuiltError("Pipeline is not built yet")

        results = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if self.use_entity_ruler:
                entity_name = ent._.entity_id_
            else:
                entity_name = ent.text

            results.append({
                "label": ent.label_,
                "entity": ent.text,
                "entity_name": entity_name,
                "sentiment": ent._.sentiment
            })

        return results

    def summarize(self, contexts):
        """
        :param contexts:
        :return:
        """
        if not self.is_pipeline_built:
            raise PipelineNotBuiltError("Pipeline is not built yet")

        print("Start analyzing!")
        label2id = self.sentiment_model.model.config.label2id
        id2label = self.sentiment_model.model.config.id2label

        entity_sentiment_summary = defaultdict(
            lambda: {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "total": 0
            }
        )

        for context in contexts:
            predictions = self.predict(context)

            # Handle multiple sentiments for the same entity
            entity_sentiment_cnt = defaultdict(list)
            for prediction_dict in predictions:
                # use entity_name to merge alias results
                entity = prediction_dict["entity_name"]
                sentiment = label2id[prediction_dict["sentiment"]]
                entity_sentiment_cnt[entity].append(sentiment)

            # For each entity, get avg sentiment & round it to int
            for entity, sentiment_list in entity_sentiment_cnt.items():
                avg_sentiment = np.round(np.mean(sentiment_list)).astype(int)
                entity_sentiment_summary[entity][id2label[avg_sentiment]] += 1
                entity_sentiment_summary[entity]["total"] += 1

        # Get top 5
        top5 = sorted(
            entity_sentiment_summary.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )[:5]
        # Print top 5
        for entity_sentiment in top5:
            print(entity_sentiment)

        print("Finished analyzing!")
        return entity_sentiment_summary

    @staticmethod
    def extract_relevant_context(doc: Doc, half_window_size: int) -> dict:
        """
        extract relevant context for each entity recognized
        :param doc: doc of tweet after NER
        :param half_window_size: half of the window size to get relevant context
        :return: dictionary containing entity: relevant_context pairs
        """
        context_dict = {}
        # Order entities for later usage
        entities = sorted(doc.ents, key=lambda x: x.start_char)
        for i, ent in enumerate(entities):
            entity_key = f"{ent.text}_{ent.start_char}_{ent.end_char}"

            # Start of window
            start_idx = max(0, ent.start_char - half_window_size)
            # End of window
            end_idx = min(len(doc.text), ent.end_char + half_window_size)

            # Check if another entity is within the context window
            # Before
            if i > 0:
                prev_entity_end = entities[i - 1].end_char
                if prev_entity_end > start_idx:
                    start_idx = prev_entity_end
            # After
            if i < len(entities) - 1:
                next_entity_start = entities[i + 1].start_char
                if next_entity_start < end_idx:
                    end_idx = next_entity_start

            # Extract relevant context
            context_dict[entity_key] = doc.text[start_idx:end_idx]

        return context_dict

    def add_ner_model(self) -> None:
        """
        wrapper for custom_ner_model
        """
        @Language.component("custom_ner_model")
        def ner_component(doc: Doc) -> Doc:
            # Get prediction
            ner_prediction = self.ner_model.predict(doc.text, self.use_crf)
            for ent in ner_prediction['entities']:
                start, end, label = ent['start_char'], ent['end_char'], ent['label']
                # Create a Span for each entity
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    doc.ents += (span,)
            return doc

        # Add to pipeline
        self.nlp.add_pipe("custom_ner_model", first=True)

    def add_sentiment_model(self) -> None:
        """
        wrapper for custom_sentiment_model
        """
        @Language.component("custom_sentiment_model")
        def sentiment_component(doc: Doc) -> Doc:
            context_dict = EntitySentimentModel.extract_relevant_context(
                doc=doc,
                half_window_size=self.half_window_size
            )
            for ent in doc.ents:
                entity_key = f"{ent.text}_{ent.start_char}_{ent.end_char}"
                if entity_key in context_dict:
                    entity_context = context_dict[entity_key]
                    # Get entity sentiment
                    sentiment_prediction = self.sentiment_model.predict(entity_context)
                    # Store it
                    ent._.sentiment = sentiment_prediction
            return doc

        self.nlp.add_pipe(factory_name="custom_sentiment_model", last=True)

    # TODO: add extra patterns
    def add_entity_ruler(
        self,
        pattern_paths: list[str]
    ) -> None:
        """
        add entity_ruler to the pipeline after NER Model
        """
        # Should be after NER in the pipeline
        ruler = self.nlp.add_pipe(
            factory_name="entity_ruler",
            after="custom_ner_model",
            config={
                "overwrite_ents": True
            }
        )
        # Some example patterns
        for path in pattern_paths:
            with open(path, "r") as json_file:
                patterns = json.load(json_file)
            ruler.add_patterns(patterns)
            ruler.add_patterns([{"label": "group", "pattern": "the gunners"}])

    def add_alias_handler(self):
        """
        wrapper for alias handler after add_entity_ruler()
        """
        @self.nlp.component("alias_handler")
        def alias_handler(doc: Doc) -> Doc:
            for ent in doc.ents:
                if ent.ent_id_:
                    ent._.entity_id_ = ent.ent_id_
                else:
                    ent._.entity_id_ = ent.text
            return doc

        self.nlp.add_pipe("alias_handler", after="entity_ruler")
