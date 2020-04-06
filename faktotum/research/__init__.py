import logging
import transformers

logging.getLogger("transformers").setLevel(logging.ERROR)

from faktotum.research import evaluation
from faktotum.research.corpus import load_corpus, sentencize_corpus, tokenize_corpus
from faktotum.research.knowledge import KnowledgeBase
from faktotum.research.ontologia import FastText, TfIdf, Word2Vec
from faktotum.research.utils import sentencize, tokenize
from faktotum.research import vendor
from faktotum.research import clustering
from faktotum.research import regression, classification
