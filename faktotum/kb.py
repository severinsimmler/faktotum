class KnowledgeBase:
    def __init__(self, data, domain):
        self.data = data
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
