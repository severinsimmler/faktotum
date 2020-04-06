import math
import random

from faktotum.research.corpus.core import Corpus
from faktotum.research.exploration import TopicModel

random.seed(23)


def evaluate_topic_model(corpus, model):
    scores = list()
    random_documents = select_random_documents(corpus)
    for i, document in enumerate(random_documents):
        print(f"Document {i + 1}/10: {document.name}")
        print("~~~~~~~~~~~~~~\n")
        print(document.content[:3000])
        print("\n--------------------------------------------------\n")
        topics, intruder = select_topics(model, document.name)
        random.shuffle(topics)
        for i, (index, _) in enumerate(topics):
            topic = ", ".join([word for word in model.topics.iloc[index][:10] if word])
            print(f"Topic {i}:", topic)
        guess = input("\nWhich topic is the intruder? ")
        print("\n##################################################\n\n\n")
        guess = topics[int(guess)][1]
        score = calculate_topic_log_odds(intruder, guess)
        print(score)
        scores.append(score)
    print(scores)
    return scores


def calculate_topic_log_odds(intruder: float, guess: float):
    return math.log(intruder / guess)


def select_random_documents(corpus: Corpus, n: int = 10):
    documents = list(corpus.documents)
    return random.sample(documents, n)


def select_topics(model: TopicModel, document_name: str):
    distribution = model.document_topics.loc[:, document_name]
    top3 = distribution.sort_values(ascending=False)[:3]
    bottom3 = distribution.sort_values(ascending=False)[-3:]
    top3 = [(index, score) for index, score in zip(top3.index, top3)]
    bottom3 = [(index, score) for index, score in zip(bottom3.index, bottom3)]
    intruder = random.choice(bottom3)
    top3.append(intruder)
    return top3, intruder[1]
