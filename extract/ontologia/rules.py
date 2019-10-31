def filter_corpus(corpus, words):
    for name, document in corpus.items():
        sentences = list(filter_sentences(document, words))
        if sentences:
            yield name, sentences


def filter_sentences(document, words):
    for sentence in document:
        if any(word.lower() in " ".join(sentence).lower() for word in words):
            yield sentence
