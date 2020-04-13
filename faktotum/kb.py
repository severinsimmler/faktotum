"""
faktotum.kb
~~~~~~~~~~~

This module implements a basic class for knowledge bases.
"""

import logging
from pathlib import Path
from typing import Union

from faktotum.typing import KnowledgeBaseDump


class KnowledgeBase:
    def __init__(self, data: KnowledgeBaseDump):
        self.data = data
        for identifier, knowledge in self.data.items():
            for context in knowledge["CONTEXTS"]:
                if "EMBEDDINGS" not in self.data[identifier]:
                    self.data[identifier]["EMBEDDINGS"] = [None]
                else:
                    self.data[identifier]["EMBEDDINGS"].append(None)

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_dump(cls, filepath: Union[str, Path]):
        filepath = Path(filepath)
        logging.info(f"Loading knowledge base from {filepath.name}...")
        with filepath.open("r", encoding="utf-8") as dump:
            data = json.load(dump)
            return cls(data)
