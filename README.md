# Extract information from unstructured text

## Installation

```
$ pip install faktotum
```

## Getting started

```python
>>> import json
>>> import faktotum
>>> domain = "literary-texts"
>>> with open("kb.json", "r", encoding="utf-8") as f:
...     data = json.load(f)
>>> kb = faktotum.KnowledgeBase(data, domain=domain)
>>> text = "Ihr Name war Charlotte, sein Name war Eduard."
>>> faktotum.nel(text, domain=domain)

```