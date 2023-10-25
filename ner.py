from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import torch
doc="Zeekacin 500mg, Zozidime 1000mg, Zardpime T 1000mg, Arkamin"
results=ner_prediction(corpus=doc,compute='cpu')

print(results)