import logging

from extract.utils import logger, tokenize, sentencize
from extract.corpus import Corpus
from extract.ontologia import FastText


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
