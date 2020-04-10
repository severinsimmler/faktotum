import tqdm
import transformers

from faktotum.utils import MODELS, align_index, extract_features, pool_tokens


class KnowledgeBase:
    def __init__(self, data, domain):
        self.data = data
        self._model_name = MODELS["ned"][domain]
        self.pipeline = transformers.pipeline(
            "feature-extraction", model=self._model_name, tokenizer=self._model_name
        )
        self._vectorize_contexts()

    def _vectorize_contexts(self):
        for key, value in self.data.items():
            for entity_indices, context in zip(
                value["ENTITY_INDICES"], value["CONTEXTS"]
            ):
                # Lazy load embeddings on inference
                embeddings = None
                if "EMBEDDINGS" not in self.data[key]:
                    self.data[key]["EMBEDDINGS"] = [embeddings]
                else:
                    self.data[key]["EMBEDDINGS"].append(embeddings)

    def items(self):
        for key, value in self.data.items():
            yield key, value

    def __len__(self):
        return len(self.data)
