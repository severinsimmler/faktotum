import json
from pathlib import Path
from typing import List, Generator, Union, Tuple

import SPARQLWrapper


SPARQL = SPARQLWrapper.SPARQLWrapper("http://dbpedia.org/sparql")
PROPERTIES = {
    "http://dbpedia.org/ontology/abstract": "ABSTRACT",
    "http://dbpedia.org/ontology/almaMater": "ALMA_MATER",
    "http://dbpedia.org/ontology/birthName": "NAME",
    "http://dbpedia.org/ontology/board": "BOARD",
    "http://dbpedia.org/ontology/knownFor": "KNOWN_FOR",
    "http://dbpedia.org/ontology/occupation": "OCCUPATION",
    "http://dbpedia.org/property/after": "SUCCESSOR",
    "http://dbpedia.org/property/before": "PREDECESSOR",
    "http://dbpedia.org/property/organization": "ORGANIZATION",
    "http://dbpedia.org/ontology/locationCity": "LOCATION_CITY",
    "http://dbpedia.org/ontology/locationCountry": "LOCATION_COUNTRY",
    "http://dbpedia.org/ontology/foundedBy": "FOUNDER",
    "http://dbpedia.org/ontology/foundingYear": "FOUNDING_YEAR",
    "http://dbpedia.org/ontology/industry": "INDUSTRY",
    "http://dbpedia.org/ontology/keyPerson": "KEY_PERSON",
    "http://dbpedia.org/ontology/location": "LOCATION",
    "http://dbpedia.org/property/owner": "OWNER",
    "http://purl.org/dc/terms/description": "DESCRIPTION",
    "http://purl.org/linguistics/gold/hypernym": "HYPERNYM",
}


def query_dbpedia(uri: str) -> List[dict]:
    query = f"SELECT * WHERE {{ <{uri}> ?property ?value }}"
    SPARQL.setQuery(query)
    SPARQL.setReturnFormat(SPARQLWrapper.JSON)
    results = SPARQL.query().convert()
    return [
        result
        for result in results["results"]["bindings"]
        if result["property"]["value"] in PROPERTIES
    ]


def format_results(results: List[dict]) -> Generator[Tuple[str, List[str]], None, None]:
    for result in results:
        property_ = PROPERTIES[result["property"]["value"]]
        if property_ == "ABSTRACT" and result["value"]["xml:lang"] != "de":
            continue
        value = result["value"]["value"]
        yield property_, [value]


def enrich_knowledge_base(filepath: Union[str, Path]):
    kb = dict()

    with Path(filepath).open("r", encoding="utf-8") as file_:
        data = json.load(file_)

    for i, (uri, values) in enumerate(data.items()):
        results = query_dbpedia(uri)
        results = dict(format_results(results))
        if "NAME" in results:
            results["NAME"].extend(values["MENTION"])
        else:
            results["NAME"] = values["MENTION"]
        results["URI"] = uri
        kb[f"{values['TYPE']}_{i}"] = results
    return kb
