from typing import List, Dict, Union
import transformers
import pandas as pd

Entities = List[Dict[str, Union[str, float]]]
Pipeline = transformers.pipelines.Pipeline
KnowledgeBase = Dict[str, Dict[str, Union[List[str], str]]]
TaggedTokens = pd.DataFrame