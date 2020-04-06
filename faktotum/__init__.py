from faktotum import evaluation, scripts
from faktotum.corpus import load_corpus, sentencize_corpus, tokenize_corpus
from faktotum.knowledge import KnowledgeBase
from faktotum.ontologia import FastText, TfIdf, Word2Vec
from faktotum.utils import sentencize, tokenize
from faktotum import vendor
from faktotum import clustering
from faktotum import regression, classification
from faktotum import pipelines
from faktotum.pipelines import ner, ned