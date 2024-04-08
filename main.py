
#from tkinter import _Padding
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from cross_scorer_model import *
device="cuda"
def run_model(model, tokenizer, prompt, response):


    batch = tokenizer(prompt, response, padding='longest',\
            truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        sims = model.score_forward(**batch).sigmoid().flatten().tolist()
    return sims

model_name = "roberta-base"
encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cross_scorer = CrossScorerCrossEncoder(encoder).to(device)

scorer_path = # your model 
c_ckpt = torch.load(scorer_path, map_location=torch.device(device) )
cross_scorer.load_state_dict(c_ckpt["model_state_dict"])
cross_scorer.eval()

while True:
    print("/"*30)
    prompt = input("Give prompt: ")
    response = input("Give respose: ")
    score = run_model(cross_scorer, tokenizer, prompt, response)
    print("Score: ", score)
    print()