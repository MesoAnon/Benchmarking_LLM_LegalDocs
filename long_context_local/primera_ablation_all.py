from transformers import AutoTokenizer, LEDForConditionalGeneration
from datasets import load_dataset
import torch
import numpy as np
import json
import os
from argparse import ArgumentParser

def select_docs_random_selection(docket, limit_docket_docs = 10):
    if type(docket) is list:
        if len(docket) > limit_docket_docs:
            docs = np.random.choice(docket, size = limit_docket_docs)
            docs = docs.tolist()
        else:
            docs = docket
    else:
        docs = [docket]

    return docs

# select first 5 and last 5 docs
def select_docs_first5last5(docket, limit_docket_docs = 10):
    half_docs = limit_docket_docs // 2

    if type(docket) is list :
        if len(docket) > limit_docket_docs:
            subset_docs = docket[:half_docs] + docket[-half_docs:]
        else:
            subset_docs = docket
    else:
        subset_docs = [docket]

    return subset_docs

def get_response(text, model, tokenizer):
    tokens = tokenizer(text, truncation=True, padding="max_length", return_tensors="pt", max_length=4096)["input_ids"].to(device)
    outputs = model.generate(tokens)[0]

    return tokenizer.decode(outputs, skip_special_tokens=True)

model_map = {
    "primera_tiny": "primera-multi_lexsum-source-tiny",
    "primera_short": "primera-multi_lexsum-source-short",
    "primera_long": "primera-multi_lexsum-source-long",
    "led_tiny": "led-base-16384-multi_lexsum-source-tiny", 
    "led_short": "led-base-16384-multi_lexsum-source-short", 
    "led_long": "led-base-16384-multi_lexsum-source-long"
}

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument("-d", type=str)
    args_parser.add_argument("-i", type=str)
    args_parser.add_argument("-m", type=str)

    args = args_parser.parse_args()

    gpu_id = args.i
    device = torch.device(f"cuda:{gpu_id}")
    
    datasubset = args.d
    model_name = model_map[args.m]
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_name}")
    model = LEDForConditionalGeneration.from_pretrained(f"models/{model_name}").to(device)

    dataset = load_dataset("allenai/multi_lexsum", name="v20230518")
    dataset = dataset["test"].filter(lambda x: x[f"summary/{datasubset}"] != None)["sources"]

    for doc_selection_func in [select_docs_random_selection, select_docs_first5last5]:
        selection_name = "_".join(doc_selection_func.__name__.split("_")[2:])
        path = f"answers_multilex/{args.m}/{selection_name}/"
        os.makedirs(path, exist_ok=True)
        for idx, docket in enumerate(dataset[:]):
            if f"{idx}.json" in os.listdir(path):
                continue
            docs = doc_selection_func(docket, 10)
            if "primera" in model_name:
                sep_token = "<doc-sep>"# from PRIMERA paper
            elif "led" in model_name:
                sep_token = "</s>"# from LED paper
            text = sep_token.join(docs) 

            summary = get_response(text, model, tokenizer)    
            json.dump({
                "summary": summary
            }, indent=2, fp=open(path+f"{idx}.json", "w"))