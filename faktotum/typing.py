from typing import Dict, List, Union

import pandas as pd
import transformers

Entities = List[Dict[str, Union[str, float]]]
Pipeline = transformers.pipelines.Pipeline
KnowledgeBase = Dict[str, Dict[str, Union[List[str], str]]]
TaggedTokens = pd.DataFrame
