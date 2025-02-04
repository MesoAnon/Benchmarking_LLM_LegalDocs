import json
import os
import torch
import warnings
import numpy as np
import spacy
import sys
import pytextrank

from datasets import load_dataset
from summarizer import Summarizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)

# load data
multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20230518")
modified_dataset = multi_lexsum["test"].filter(lambda x: x["summary/short"] != None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_summ = Summarizer("distilbert-base-uncased", hidden_concat = True, hidden = [-1, -2], gpu_id = 0)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", last = True)
nlp.max_length = 3_000_000 # needed for long documents, otherwise spaCy cannot handle that many tokens

# get list of most salient sentences by passing a full length and untokenized document
def get_extractive_summary_bert(doc, limit_sentences = 10):
    return model_summ(doc, use_first = False, return_as_list = True, num_sentences = limit_sentences) 

# get list of most salient sentences by passing a full length and untokenized document
# based on co occurence of words and keywords, not tfidf/modern embeddings
def get_extractive_summary_textrank(doc, limit_sentences = 10):
    parsed_doc = nlp(doc)
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

    return summary[:-1]

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

# given a dataset (any type should work), an output directory, an extractive measure, and a document selection method
# create json files for each sample in the dataset
# this system can handle both multi document and single document texts
def extractive_system(data, path, extractive_sum_func, selection_func):
    os.makedirs(path, exist_ok=True)

    limit_docket_docs = 10
    for docket_id, docket in enumerate(tqdm(data, total = len(data), desc=f"Extraction={extractive_sum_func.__name__} | Selection={selection_func.__name__} | Data={path.split('/')[-2]}")):
        if f"{docket_id}.json" in os.listdir(path):
            continue

        documents = selection_func(docket, limit_docket_docs = limit_docket_docs)

        summaries = []
        for doc in documents:
            summary_aux = extractive_sum_func(doc, limit_sentences = 100)

            summaries.append(summary_aux)
        json.dump(summaries, open(f"{path}/{docket_id}.json", "w"), indent = 2)

def load_eurlex_data():
    data = load_dataset("dennlinger/eur-lex-sum", "english")
    return data["train"]["reference"] + data["test"]["reference"] + data["validation"]["reference"]

if __name__ == "__main__":
    # use command line arguments to get input for system
    data_name, extractive_sum, selection = sys.argv[1:]

    extractive_sum_map = {
        "bert": get_extractive_summary_bert,
        "textrank": get_extractive_summary_textrank
    }
    selection_map = {
        "first5last5": select_docs_first5last5,
        "random_selection": select_docs_random_selection
    }
    data_map = {
        "multilex_long": load_dataset("allenai/multi_lexsum", name="v20230518")["test"].filter(lambda x: x["summary/long"] != None)["sources"],
        "multilex_short": load_dataset("allenai/multi_lexsum", name="v20230518")["test"].filter(lambda x: x["summary/short"] != None)["sources"],
        "multilex_tiny": load_dataset("allenai/multi_lexsum", name="v20230518")["test"].filter(lambda x: x["summary/tiny"] != None)["sources"],
        "eurlexsum": load_eurlex_data()
    }

    path = f"extracted_sums/{data_name}/extracted_sums_json_{selection}_{extractive_sum}"
    extractive_sum_func = extractive_sum_map[extractive_sum]
    selection_func = selection_map[selection]
    data = data_map[data_name]

    open(f"example_doc_{data_name}.txt", "w").write(data[0])

    extractive_system(data, path, extractive_sum_func, selection_func)
