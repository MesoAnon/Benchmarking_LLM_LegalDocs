import json
import os
import torch
import warnings
import numpy as np
import spacy
import sys
import pytextrank
from functools import partial

from datasets import load_dataset
from summarizer import Summarizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

# get list of most salient sentences by passing a full length and untokenized document
def get_extractive_summary_bert(doc, model, limit_sentences = 10, **kwargs):
    return model(doc, use_first = False, return_as_list = True, num_sentences = limit_sentences) 

# get list of most salient sentences by passing a full length and untokenized document
# based on co occurence of words and keywords, not tfidf/modern embeddings
def get_extractive_summary_textrank(doc, model, limit_sentences = 10, **kwargs):
    parsed_doc = model(doc)
    limit_phrases = None
    sentence_bounds = [[sentence.start, sentence.end, set([])] for sentence in parsed_doc.sents]

    phrase_id = 0
    unit_vector = []
    # get original text according to rank
    # get rank of all phrases
    for p in parsed_doc._.phrases:
        unit_vector.append(p.rank)

        for chunk in p.chunks:
            for sent_start, sent_end, sent_vector in sentence_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if limit_phrases and phrase_id >= limit_phrases:
            break

    # euclidean distance between phrases, choose those with smallest distance
    unit_vector = np.asarray(unit_vector)
    sum_ranks = np.sum(unit_vector)
    unit_vector /= sum_ranks 
    sent_rank = {}
    sent_id = 0
    for sent_start, sent_end, sent_vector in sentence_bounds:
        sum_sq = 0
        # only add to sum if phrase id not in sent vector so a phrase will not count itself
        sum_sq = np.sqrt(np.sum([unit_vector[phrase_id]*unit_vector[phrase_id] for phrase_id in range(len(unit_vector)) if phrase_id not in sent_vector]))
        sent_rank[sent_id] = sum_sq
        sent_id += 1

    sent_rank = dict(sorted(sent_rank.items(), key = lambda x: x[1]))

    sent_id = 0
    sent_text = {}
    for sentence in parsed_doc.sents:
        sent_text[sent_id] = sentence.text
        sent_id += 1

    limit = 0
    summary = []
    for id_sentence in sent_rank.keys():
        summary.append(sent_text[id_sentence])
        limit += 1
        if limit > limit_sentences:
            break
    # [:-1] = ignore first sentence - for the documents we used, the first sentence did not provide meaningful information
    return_first_sentence = kwargs.pop("return_first_sentence", False)
    if return_first_sentence:
        return summary
    else:
        return summary[:-1] if len(summary) > 1 else summary

# build in layers, from smallest number of doc types to largest
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

def select_all(docket, limit_docket_docs):
    return docket

EXTRACTIVE_SUM_MAP = {
    "bert": get_extractive_summary_bert,
    "textrank": get_extractive_summary_textrank
}
SELECTION_MAP = {
    "first5last5": select_docs_first5last5,
    "random_selection": select_docs_random_selection,
    "all": select_all
}

# given a dataset (any type should work), an output directory, an extractive measure, and a document selection method
# create json files for each sample in the dataset
# this system can handle both multi document and single document texts
def extractive_system(docket, extractive_sum_func, selection_func, limit_docket_docs):
    documents = selection_func(docket, limit_docket_docs = limit_docket_docs)
    summaries = []
    for doc in documents:
        summary_aux = extractive_sum_func(doc=doc, limit_sentences = 10)
        summaries.append(summary_aux)
    return summaries

def extractive_system_loop(data, path, extractive_sum_func, selection_func, limit_docket_docs):
    os.makedirs(path, exist_ok=True)

    iterator = enumerate(tqdm(data, total = len(data), desc=f"Extraction={extractive_sum_func.__name__} | Selection={selection_func.__name__} | Data={path.split('/')[-2]}"))
    for docket_id, docket in iterator:
        if f"{docket_id}.json" in os.listdir(path):
            continue

        summaries = extractive_system(docket, extractive_sum_func, selection_func, limit_docket_docs)
        json.dump(summaries, open(f"{path}/{docket_id}.json", "w"), indent = 2)

def load_eurlex_data():
    data = load_dataset("dennlinger/eur-lex-sum", "english")
    return data["train"]["reference"] + data["test"]["reference"] + data["validation"]["reference"]

if __name__ == "__main__":
    # use command line arguments to get input for system
    data_name, extractive_sum, selection = sys.argv[1:]


    data_map = {
        "multilex_long": load_dataset("allenai/multi_lexsum", name="v20230518")["test"].filter(lambda x: x["summary/long"] != None)["sources"],
        "multilex_short": load_dataset("allenai/multi_lexsum", name="v20230518")["test"].filter(lambda x: x["summary/short"] != None)["sources"],
        "multilex_tiny": load_dataset("allenai/multi_lexsum", name="v20230518")["test"].filter(lambda x: x["summary/tiny"] != None)["sources"],
        "eurlexsum": load_eurlex_data()
    }

    match extractive_sum:
        case "bert":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Summarizer("distilbert-base-uncased", hidden_concat = True, hidden = [-1, -2], gpu_id = 0)
        case "textrank":
            model = spacy.load("en_core_web_sm")
            model.add_pipe("textrank", last = True)
            model.max_length = 3_000_000 # needed for long documents, otherwise spaCy cannot handle that many tokens

    path = f"extracted_sums/{data_name}_full/extracted_sums_json_{selection}_{extractive_sum}"
    extractive_sum_func = EXTRACTIVE_SUM_MAP[extractive_sum]
    extractive_sum_func = partial(extractive_sum_func, model=model)
    selection_func = SELECTION_MAP[selection]
    data = data_map[data_name]

    # un-nest the data
    if "eurlex" in data_name:
        data = [[d] for d in data]

    # open(f"example_doc_{data_name}.txt", "w").write(data[0])
    limit_docket_docs = 10
    extractive_system_loop(data, path, extractive_sum_func, selection_func, limit_docket_docs)
