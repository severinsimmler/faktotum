#!/bin/sh


echo "#### Running TF-IDF..."
poetry run tfidf -- --corpus /mnt/data/users/simmler/corpora/plaintext/gutenberg
poetry run tfidf -- --corpus /mnt/data/users/simmler/corpora/plaintext/pressemitteilungen

