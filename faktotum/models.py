"""
faktotum.models
~~~~~~~~~~~~~~~

This module implements two model classes for named entity recognition and disambiguation.
"""

import logging

import transformers

from faktotum.typing import Pipeline

MODEL_NAMES = {
    "ner": {
        "literary-texts": "severinsimmler/literary-german-bert",
        "press-texts": "severinsimmler/german-press-bert",
    },
    "ned": {
        "literary-texts": "severinsimmler/literary-german-bert",
        "press-texts": "severinsimmler/bert-adapted-german-press",
    },
}


class NamedEntityRecognition:
    def __init__(self):
        self._literary = MODEL_NAMES["ner"]["literary-texts"]
        self._press = MODEL_NAMES["ner"]["press-texts"]

    def __getitem__(self, domain: str) -> Pipeline:
        if domain.lower() == "literary-texts":
            if not hasattr(self, "literary_pipeline"):
                logging.info("Loading named entity recognition model...")
                self.literary_pipeline = transformers.pipeline(
                    "ner",
                    model=self._literary,
                    tokenizer=self._literary,
                    ignore_labels=[],
                )
            return self.literary_pipeline
        elif domain.lower() == "press-texts":
            if not hasattr(self, "press_pipeline"):
                logging.info("Loading named entity recognition model...")
                self.press_pipeline = transformers.pipeline(
                    "ner", model=self._press, tokenizer=self._press, ignore_labels=[]
                )
            return self.press_pipeline
        else:
            raise ValueError(f"The domain {domain} is not supported.")


class NamedEntityDisambiguation:
    def __init__(self):
        self._literary = MODEL_NAMES["ned"]["literary-texts"]
        self._press = MODEL_NAMES["ned"]["press-texts"]

    def __getitem__(self, domain: str) -> Pipeline:
        if domain.lower() == "literary-texts":
            if not hasattr(self, "literary_pipeline"):
                logging.info("Loading named entity disambiguation model...")
                self.literary_pipeline = transformers.pipeline(
                    "feature-extraction", model=self._literary, tokenizer=self._literary
                )
            return self.literary_pipeline
        elif domain.lower() == "press-texts":
            if not hasattr(self, "press_pipeline"):
                logging.info("Loading named entity disambiguation model...")
                self.press_pipeline = transformers.pipeline(
                    "feature-extraction", model=self._press, tokenizer=self._press
                )
            return self.press_pipeline
        else:
            raise ValueError(f"The domain {domain} is not supported.")
