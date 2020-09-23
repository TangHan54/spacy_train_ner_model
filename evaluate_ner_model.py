import srsly
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

input_fpath = "ner_profile_test_v1.jsonl"

obj = srsly.read_jsonl(input_fpath)
TRAIN_DATA = []
for record in obj:
    if record["answer"] == "accept":
        spans = record.get("spans", [])
        entities = [(span["start"], span["end"], span["label"]) for span in spans]
        TRAIN_DATA.append((record["text"], {"entities": entities}))
print("test data size:")
print(len(TRAIN_DATA))

model_path = "en_core_web_md"
nlp = spacy.load(model_path)

scorer = nlp.evaluate(TRAIN_DATA, verbose=False)
print("total precision:")
print(scorer.scores["ents_p"])
print("total recall:")
print(scorer.scores["ents_r"])
print("total f score")
print(scorer.scores["ents_f"])
print(scorer.scores["ents_per_type"])
