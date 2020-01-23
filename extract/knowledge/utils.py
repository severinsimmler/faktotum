import json
from pathlib import Path
from typing import Generator


def load_wikidata_dump(filepath: Union[str, Path]) -> Generator[dict, None, None]:
    with Path(filepath).open("r", encoding="utf-8") as file_:
        for line in file_:
            yield json.loads(line)
