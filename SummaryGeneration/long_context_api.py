import argparse
import json
import os
import numpy as np

from openai import OpenAI
from openai.types import Completion
from datasets import load_dataset
from math import ceil
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import sys
from functools import partial
import math
from SummaryGeneration.env_utils import load_env_from_file
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError

# specify which GPU is visible to script
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_map = {
    "multilex_tiny": "summary/tiny",
    "multilex_short": "summary/short",
    "multilex_long": "summary/long",
    "eurlexsum": "reference",
    "eurlexsum_test": "reference",
    "eurlexsum_validation": "reference"
}

# based on addage from OpenAI: 1 token ~ 0.75 words --> 1 word ~ 1.333 tokens; +10 words for to complete things

input_api_map = {
    "openai": partial(OpenAI),
    "deepinfra": partial(OpenAI, base_url = "https://api.deepinfra.com/v1/openai")
}

model_map = {
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

def load_data_local(dataset_name, attribute):
    # return split of dataset only for EURLEXsum (so we can compute metrics for individual splits of the original dataset)
    split = None
    if "multilex" in dataset_name:
        match attribute:
            case "summary":
                attribute = dataset_map[dataset_map]
            case "source":
                attribute = "sources"

        dataset = load_dataset("allenai/multi_lexsum", name="v20230518")
        dataset = dataset["test"].filter(lambda x: x[dataset_map[dataset_name]] != None)[attribute]
    else:
        match attribute:
            case "summary":
                attribute = "summary"
            case "source":
                attribute = "reference"

        dataset = load_dataset("dennlinger/eur-lex-sum", "english")
        match dataset_name:
            case "eurlexsum_test":
                split = slice(len(dataset["train"][attribute]), len(dataset["train"][attribute] + dataset["test"][attribute]))
                dataset = dataset["test"][attribute]
            case "eurlexsum_validation":
                split = slice(len(dataset["train"][attribute] + dataset["test"][attribute]), len(dataset["train"][attribute] + dataset["test"][attribute] + dataset["validation"][attribute]))
                dataset = dataset["validation"][attribute]
            case "eurlexsum":
                split = slice(0,len(dataset["train"][attribute] + dataset["test"][attribute] + dataset["validation"][attribute]))
                dataset = dataset["train"][attribute] + dataset["test"][attribute] + dataset["validation"][attribute]

        print(split, split.stop - split.start)    

    print(f"Number of test cases={len(dataset)}")
    return dataset, split    

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return tokenizer

def load_mds(x: list, current_prompt_length, tokenizer, max_context_length):
    remaining_input_size = max_context_length - current_prompt_length
    
    text_lens = np.asarray([len(tokenizer.encode(text)) for text in x])
    text_lens_arg_sort = np.argsort(text_lens)

    aux_text_list = []
    for arg_text in text_lens_arg_sort:
        aux_text_list += [x[arg_text]]
        if len(tokenizer.encode("\n".join(aux_text_list))) > remaining_input_size:
            aux_text_list.pop()
            break

    text = "\n".join(aux_text_list)
    prompt = prompt_template.format(INPUT=text)

    assert len(tokenizer.encode(prompt)) <= max_context_length, f"{len(tokenizer.encode(text))}, {len(aux_text_list)}"

    return prompt

def load_sds(x, current_prompt_length, tokenizer, max_context_length):
    remaining_input_size = max_context_length - current_prompt_length
    text = tokenizer.decode(tokenizer.encode(x)[:remaining_input_size])
    prompt = prompt_template.format(INPUT=text)

    assert len(tokenizer.encode(prompt)) <= max_context_length

    return prompt

# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def completion_with_retry(file_path, client, **kwargs):
    # file_path: model/prompt_type/extract_sum_type/idx
    # _, prompt_type, extract_sum_type, idx = file_path.split("/")[2:]
    success = False
    try:
        chat_completion = client.chat.completions.create(**kwargs)
        json.dump([*kwargs["messages"], {"role": "assistant", "content": chat_completion.choices[0].message.content, "usage": dict(chat_completion.usage)}], open(f"{file_path}_{WORDS}.json", "w"), indent=2)  
        success = True  
    except Exception as exc:
        # self.logger.error(f"({type(exc).__name__} ({exc.args}) - {prompt_type} - {extract_sum_type} - {idx}")
        success = False  

    return success

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("--model", type=str)
    args_parser.add_argument("--word_count", type=int)
    args_parser.add_argument("--dataset", type=str, choices=["multilex_tiny", "multilex_short", "multilex_long", "eurlexsum", "eurlexsum_test", "eurlexsum_validation"])
    args = args_parser.parse_args()

    model_name = model_map[args.model]
    dataset_name = args.dataset

    WORDS = args.word_count
    # based on addage from OpenAI: 1 token ~ 0.75 words --> 1 word ~ 1.333 tokens; +10 words for to complete things
    MAX_OUTPUT_LENGTH = ceil(WORDS * 1.333 + 10)
    SAFETY_MARGIN = 50
    SYSTEM_PROMPT = f"You are a legal expert. You are tasked with reading legal texts and creating summaries. Your summaries are truthful, relevant, and faithful to the source documents, using only facts and entities present in them, while also including as many as possible. Your summaries read like histories of cases, useful for other lawyers. Your summary must be around {WORDS} words long."

    input_model_params = {
    "mixtral": {
        "max_context_length": 32768,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "mistral": {
        "max_context_length": 32768,
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "gemma": {
        "max_context_length": 8192,
        "model": "google/gemma-1.1-7b-it",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "gpt-3.5": {
        "max_context_length": 16384,
        "model": "gpt-3.5-turbo-1106",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "llama3": {
        "max_context_length": 8192,
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    },
    "llama3.1": {
        "max_context_length": 131_072,
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": MAX_OUTPUT_LENGTH
    }
}

    # llm, tokenizer = load_model_tokenizer(model_name)
    load_env_from_file(".")
    tokenizer = load_tokenizer(model_name)

    system_prompt_len = len(tokenizer.encode(SYSTEM_PROMPT))
    base_length = MAX_OUTPUT_LENGTH + SAFETY_MARGIN + system_prompt_len

    dataset, split = load_data_local(dataset_name, "source")
    prompt_path = "prompts/"
    iterator = tqdm(dataset[:], desc="Token count:=0")

    openai_client =  OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai",)
    max_context_length = None

    # for prompt_type in ["prompt_detailed.json", "prompt_basic.json", "prompt_cod.json"]:
    for prompt_type in ["prompt_detailed_eurlex.json", "prompt_basic_eurlex.json", "prompt_cod_eurlex.json"]:
    # for prompt_type in os.listdir(prompt_path)[:]:
        # if ("eurlex" in dataset_name and "eurlex" not in prompt_type) or ("multilex" in dataset_name and "eurlex" in prompt_type):
        #     continue
        path_response = f"answers_longcontext/{dataset_name}_{args.model}/{prompt_type[:-5]}/"
        prompt_template = json.load(open(os.path.join(prompt_path, prompt_type), "r"))["content"]
        os.makedirs(path_response, exist_ok=True)
        current_prompt_length = base_length + len(tokenizer.encode(prompt_template))

        llm_params = input_model_params[args.model]
        if not max_context_length:
            max_context_length = llm_params.pop("max_context_length")
        for idx, entry in enumerate(iterator):
            if f"{idx}_{WORDS}.json" in os.listdir(path_response):
                continue

            if isinstance(entry, list):
                prompt = load_mds(entry, current_prompt_length, tokenizer, max_context_length)
            else:
                prompt = load_sds(entry, current_prompt_length, tokenizer, max_context_length)

            iterator.set_description(f"{idx}) Token count={len(tokenizer.encode(prompt))}", refresh=True)

            message_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            file_path = f"{path_response}{idx}"
            chat_completion_success = completion_with_retry(
                        file_path=file_path,
                        client=openai_client,
                        messages=message_history,
                        **llm_params
                    )