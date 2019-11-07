#!/bin/sh

poetry run clustering -- --topics /mnt/data/users/simmler/models/topic-models/gutenberg/gutenberg-10.topics --document-topics /mnt/data/users/simmler/models/topic-models/gutenberg/gutenberg-10.doctopics --stopwords /mnt/data/users/simmler/corpora/gutenberg-stopwords.json --algorithm tsne

poetry run clustering -- --topics /mnt/data/users/simmler/models/topic-models/gutenberg/gutenberg-10.topics --document-topics /mnt/data/users/simmler/models/topic-models/gutenberg/gutenberg-10.doctopics --stopwords /mnt/data/users/simmler/corpora/gutenberg-stopwords.json --algorithm umap

poetry run clustering -- --topics /mnt/data/users/simmler/models/topic-models/pressemitteilungen/pressemitteilungen-20.topics --document-topics /mnt/data/users/simmler/models/topic-models/pressemitteilungen/pressemitteilungen-20.doctopics --stopwords /mnt/data/users/simmler/corpora/pressemitteilungen-stopwords.json --algorithm tsne

poetry run clustering -- --topics /mnt/data/users/simmler/models/topic-models/pressemitteilungen/pressemitteilungen-20.topics --document-topics /mnt/data/users/simmler/models/topic-models/pressemitteilungen/pressemitteilungen-20.doctopics --stopwords /mnt/data/users/simmler/corpora/pressemitteilungen-stopwords.json --algorithm umap