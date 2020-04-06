import logging
import transformers
from faktotum import typing, utils
from faktotum.kb import KnowledgeBase
from faktotum.pipelines import ned, nel, ner

logging.getLogger("transformers").setLevel(logging.ERROR)
