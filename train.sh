#!/bin/sh

echo "#### Running TF-IDF..."
#poetry run tfidf -- --corpus /mnt/data/users/simmler/corpora/plaintext/gutenberg
#poetry run tfidf -- --corpus /mnt/data/users/simmler/corpora/plaintext/pressemitteilungen

echo "#### Running word2vec..."
poetry run word2vec -- --corpus /mnt/data/users/simmler/corpora/plaintext/gutenberg --epochs 100 --algorithm cbow
poetry run word2vec -- --corpus /mnt/data/users/simmler/corpora/plaintext/gutenberg --epochs 100 --algorithm skipgram

poetry run word2vec  -- --corpus /mnt/data/users/simmler/corpora/plaintext/pressemitteilungen --epochs 100 --algorithm cbow
poetry run word2vec -- --corpus /mnt/data/users/simmler/corpora/plaintext/pressemitteilungen --epochs 100 --algorithm skipgram

echo "#### Running fastText..."
poetry run fasttext -- --corpus /mnt/data/users/simmler/corpora/plaintext/gutenberg --epochs 100 --algorithm skipgram

poetry run fasttext -- --corpus /mnt/data/users/simmler/corpora/plaintext/pressemitteilungen --epochs 100 --algorithm skipgram
