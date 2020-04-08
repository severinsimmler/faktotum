import tqdm
import transformers

from faktotum.utils import MODELS, extract_features, pool_entity, align_index


class KnowledgeBase:
    def __init__(self, data, domain):
        self.data = data
        self._model_name = MODELS["ned"][domain]
        self._pipeline = transformers.pipeline("feature-extraction", model=self._model_name, tokenizer=self._model_name)
        self._vectorize_contexts()

    def _vectorize_contexts(self):
        for key, value in tqdm.tqdm(self.data.items()):
            for entity_indices, context in zip(value["ENTITY_INDICES"], value["CONTEXTS"]):
                index_mapping, features = extract_features(self._pipeline, context)
                aligned_indices = align_index(entity_indices, index_mapping)
                embeddings = pool_entity(aligned_indices, features)
                if "EMBEDDINGS" not in self.data[key]:
                    self.data[key]["EMBEDDINGS"] = [embeddings]
                else:
                    self.data[key]["EMBEDDINGS"].append(embeddings)

    def items(self):
        for key, value in self.data.items():
            yield key, value