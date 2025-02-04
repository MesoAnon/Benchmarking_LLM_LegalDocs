from transformers import AutoTokenizer, LEDForConditionalGeneration
from datasets import load_dataset
import torch
import numpy as np
import json
import os
from argparse import ArgumentParser

PAD_TOKEN_ID = None
DOCSEP_TOKEN_ID = None

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
    input_ids = tokenizer.encode(text, truncation=True, padding="max_length", return_tensors="pt", max_length=4096).to(device)
    global_attention_mask = torch.zeros_like(input_ids).to(device)

    # attention on first <s> and all <doc-sep>
    global_attention_mask[:, 0] = 1
    global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
    
    outputs = model.generate(input_ids = input_ids, global_attention_mask = global_attention_mask)

    return tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces = True)

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

    PAD_TOKEN_ID = tokenizer.pad_token_id
    DOCSEP_TOKEN_ID = tokenizer.convert_tokens_to_ids("<doc-sep>")

    dataset = load_dataset("allenai/multi_lexsum", name="v20230518")
    dataset = dataset["test"].filter(lambda x: x[f"summary/{datasubset}"] != None)["sources"]

    path = f"answers_multilex_replication_att/{args.m}/"
    os.makedirs(path, exist_ok=True)
    for idx, docket in enumerate(dataset[:]):
        if f"{idx}.json" in os.listdir(path):
            continue
        # docs = doc_selection_func(docket, 10)
        docs = docket
        if "primera" in model_name:
            sep_token = "<doc-sep>"# from PRIMERA paper
        elif "led" in model_name:
            sep_token = "</s>"# from LED paper
        text = sep_token.join(docs) 

        summary = get_response(text, model, tokenizer)    
        json.dump({
            "summary": summary
        }, indent=2, fp=open(path+f"{idx}.json", "w"))