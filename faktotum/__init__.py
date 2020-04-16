import logging
import transformers
from faktotum import typing, utils
from faktotum.kb import KnowledgeBase
from faktotum.pipelines import ned, nel, ner

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
