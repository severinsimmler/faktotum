import random

from extract.corpus.core import Corpus
from extract.exploration import TopicModel


random.seed(23)


def evaluate_topic_model(corpus, model):
    correct = 0
    random_documents = select_random_documents(corpus)
    for i, document in enumerate(random_documents):
        print(f"Document {i + 1}/10: {document.name}")
        print("~~~~~~~~~~~~~~\n")
        print(document.content[:3000])
        print("\n--------------------------------------------------\n")
        topics, intruder = select_topics(model, document.name)
        random.shuffle(topics)
        for i, topic in enumerate(topics):
            print(f"Topic {i}:", topic)
        potential_intruder = input("\nWhich topic is the intruder? ")
        print("\n##################################################\n\n\n")
        if int(potential_intruder) == topics.index(intruder):
            correct += 1
    return correct / 10


def select_random_documents(corpus: Corpus, n: int = 10):
    documents = list(corpus.documents)
    return random.sample(documents, n)


def select_topics(model: TopicModel, document_name: str):
    distribution = model.document_topics.loc[:, document_name]
    top3 = list(distribution.sort_values(ascending=False)[:3].index)
    bottom3 = list(distribution.sort_values(ascending=False)[-3:].index)
    intruder_index = random.choice(bottom3)
    intruder = ", ".join(model.topics.iloc[intruder_index][:10])
    top3 = [", ".join(topic[:10]) for _, topic in model.topics.iloc[top3].iterrows()]
    top3.append(intruder)
    return top3, intruder
