from typing import Dict, List

from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings, Sentence
import torch
import numpy as np
import sklearn.metrics


Dataset = Dict[str, Dict[str, List[str]]]

random.seed(23)


def select_documents(dataset: Dataset, n: int = 5000):
    selection = random.sample(dataset, n)
    return {name: data for name, data in dataset.items() if name in selection}




class Embedding:
    def __init__(self, filepath):
        self._bert = BertEmbeddings(filepath)
        self.document_embeddings = DocumentPoolEmbeddings([self._bert])

    def get_vector(text: str) -> np.ndarray:
        sentence = Sentence(text, use_tokenizer=False)
        self.document_embeddings.embed(sentence)
        with torch.no_grad():
            vector = sentence.get_embedding()
            return vector.numpy()

    def process_batch(documents):
        for name, sentence in self.build_sentences(documents):
            yield name, self.get_vector(sentence)

    def build_sentences(documents):
        for name, sentences in documents.items():
            yield f"{name}-{index}", " ".join(
                [token["text"] for token in sentence]
                for index, sentence in sentences.items()
            )




def foo(filepath, n=5000, model=""):
    embedding = Embedding(model)
    with Path(filepath).open("r", encoding="utf-8") as file_:
        dataset = json.load(file_)
    batch = select_documents(dataset, n)
    features = pd.DataFrame(embedding.process_batch(batch)).T
    similarities = sklearn.metrics.pairwise.cosine_similarity(features)
