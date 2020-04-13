import faktotum
from faktotum.research.linking.droc import EntityLinker as DROC
from faktotum.research.linking.smartdata import EntityLinker as SMARTDATA


def droc():
    e = DROC()

    for novel in e.test.values():
        for sentence in novel:
            text = " ".join([token[0] for token in sentence])
            result = faktotum.ner(text, domain="literary-texts")
            result = result.fillna("O")
            if len(result) == len(sentence):
                for token, tag in zip(sentence, result["entity"]):
                    if tag == "O":
                        token[2] = "-"
                    if tag != "O" and token[2] == "-":
                        tag = "O"
                    token[1] = tag
    e.similarities()


def smartdata():
    e = SMARTDATA("/mnt/data/users/simmler/kb")

    for sentence in e.test:
        text = " ".join([token[0] for token in sentence])
        result = faktotum.ner(text, domain="press-texts")
        result = result.fillna("O")
        if len(result) == len(sentence):
            for token, tag in zip(sentence, result["entity"]):
                if tag == "O":
                    token[2] = "-"
                if tag != "O" and token[2] == "-":
                    tag = "O"
                token[1] = tag
    e.similarities()


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "droc":
        droc()
    else:
        smartdata()
