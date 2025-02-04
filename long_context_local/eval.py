import sys
import os
cuda_id = sys.argv[2]
# os.environ["CUDA_VISIBLE_DEVICES"]=cuda_id
import pickle
from datasets import load_dataset
import numpy as np
import torch
dataset_name = sys.argv[1]

# set env before importing to use custom model
model_path = "../models/"
device = torch.device(f"cuda:{cuda_id}")

# os.environ["MOVERSCORE_MODEL"] = model_path + "longformer-base-4096"
# from moverscore_v2 import get_idf_dict, word_mover_score
print(device)

d_map = {
    "multilex_tiny": "summary/tiny",
    "multilex_short": "summary/short",
    "multilex_long": "summary/long",
    "eurlexsum": "reference",
    "eurlexsum_test": "reference",
    "eurlexsum_validation": "reference"
}

if f"results_{dataset_name}.pickle" in os.listdir():
    exit(1)

print("Getting data")
split = None
if "multilex" in dataset_name:
    dataset = load_dataset("allenai/multi_lexsum", name="v20230518", trust_remote_code=True)
    dataset = dataset["test"].filter(lambda x: x[d_map[dataset_name]] != None)[d_map[dataset_name]]
else:
    dataset = load_dataset("dennlinger/eur-lex-sum", "english", trust_remote_code=True)
    match dataset_name:
        case "eurlexsum_test":
            split = slice(len(dataset["train"]["summary"]), len(dataset["train"]["summary"] + dataset["test"]["summary"]))
            dataset = dataset["test"]["summary"]
        case "eurlexsum_validation":
            split = slice(len(dataset["train"]["summary"] + dataset["test"]["summary"]), len(dataset["train"]["summary"] + dataset["test"]["summary"] + dataset["validation"]["summary"]))
            dataset = dataset["validation"]["summary"]
        case "eurlexsum":
            split = slice(0,len(dataset["train"]["summary"] + dataset["test"]["summary"] + dataset["validation"]["summary"]))
            dataset = dataset["train"]["summary"] + dataset["test"]["summary"] + dataset["validation"]["summary"]

    print(split, split.stop - split.start)    
    # dataset = dataset["train"]["summary"] + dataset["test"]["summary"] + dataset["validation"]["summary"]

print(f"Number of test cases={len(dataset)}")
import evaluate

rouge_scoring = evaluate.load("rouge")
bertscore = evaluate.load("bertscore", device=device)

import os
import json
def load_predicted_data(path):
    predicted = []
    ordered_files = os.listdir(path)
    ordered_files = sorted(ordered_files, key = lambda x: int(x.split(".")[0]))
    for file in ordered_files:
        predicted.append(json.load(open(path+file, "r"))[-1]["content"])
        print(path, file) if len(json.load(open(path+file, "r"))[-1]["content"]) < 1 else 1

    return predicted

from tqdm import tqdm


def compute_mover_score(predictions, references, orig_ref, orig_pred):
    idf_dict_pred = get_idf_dict(orig_pred)
    idf_dict_ref = get_idf_dict(orig_ref)

    print("Computed idf dictionaries for mover score")

    scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_pred, n_gram=1, remove_subwords=True, batch_size=3, device=device)
    torch.cuda.empty_cache()

    print("Mover score done")

    return scores
    
def evaluation(path_data, results, model, prompt_type, selection_type, limit):
    if "eurlexsum" in dataset_name:
        predicted = load_predicted_data(path_data)[split][:limit]
    else:
        predicted = load_predicted_data(path_data)[:limit]

    print(len(predicted))
    r_scores = rouge_scoring.compute(predictions=predicted, references=dataset[:len(predicted)], use_stemmer = True)
    bert_scores = bertscore.compute(predictions=predicted, references=dataset[:len(predicted)], model_type="microsoft/deberta-xlarge-mnli", batch_size = 1, verbose=True, device=device)
    # mover_score = compute_mover_score(predictions=predicted, references=dataset[:len(predicted)], orig_ref=dataset, orig_pred=predicted)

    # mover_score = np.mean(mover_score)
    mover_score = 0
    bert_scores = {"f1": np.mean(bert_scores["f1"])}
    r_scores = {metric: round(np.mean(val), 3) for metric, val in r_scores.items()}

    # mover_score = mover/len(predicted)
    # bert_scores = {"f1": bs/len(predicted)}
    # chosen deberta lange due to https://github.com/Tiiiger/bert_score/blob/master/README.md and paper
    # for some reason the data is all loaded onto the GPU so it doesn't fit anymore
    # bs = 0
    # mover = 0
    # batch_size = 50
    # for pred_idx in tqdm(range(0,len(predicted),batch_size)):
    #     # in bertscore.py, in evaluate metric bert (from evaluate huggingface), added del of scorer model to reduce VRAM usage
    #     bs_aux = bertscore.compute(predictions=predicted[pred_idx:pred_idx+batch_size], references=dataset[pred_idx:pred_idx+batch_size], model_type="microsoft/deberta-xlarge-mnli", batch_size = 1)
    #     mover_aux = compute_mover_score(references=dataset[pred_idx:pred_idx+batch_size], predictions=predicted[pred_idx:pred_idx+batch_size], orig_ref = dataset, orig_pred=predicted)
    #     mover += sum(mover_aux)
    #     bs += sum(bs_aux["f1"])

    results["model"] += [model] * 5
    results["selection_type"] += [selection_type] * 5
    results["prompt_type"] += [prompt_type] * 5
    results["score_value"] += [r_scores["rouge1"], r_scores["rouge2"], r_scores["rougeL"], bert_scores["f1"], mover_score]
    results["score_type"] += ["rouge1", "rouge2", "rougeL", "bert_score", "mover_score"]

results = {"model": [], "selection_type": [], "prompt_type": [], "score_value": [], "score_type": []}
if "eurlexsum" in dataset_name:
    dataset_name_aux = dataset_name.split("_")[0]
    path_dir = f"answers/{dataset_name_aux}"
else:
    path_dir = f"answers/{dataset_name}"
iter = 0
for model in tqdm(os.listdir(f"{path_dir}/")):
    for prompt_type in os.listdir(f"{path_dir}/{model}/"):
        for selection_type in os.listdir(f"{path_dir}/{model}/{prompt_type}/"):
            path_data = f"{path_dir}/{model}/{prompt_type}/{selection_type}/"
            evaluation(path_data, results, model, prompt_type, selection_type, len(dataset))
            iter += 1
            print(f"Iteration {iter}/{4*3*4}")


import pickle


pickle.dump(results, open(f"results_{dataset_name}.pickle", "wb"))

results = pickle.load(open(f"results_{dataset_name}.pickle", "rb"))

exit(1)

import pandas as pd

df_results = pd.DataFrame(results)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

doc_selection_map_name = {
    'random_selection': "Random10",
    'first5last5': "First5Last5"
}

prompt_type_map_name = {
    "cod": "CoD",
    "basic": "Basic",
    "detailed": "Detailed",
    "equal_cod": "CoD",
    "equal_basic": "Basic",
    "equal_detailed": "Detailed"
}

sum_ext_map_name = {
    "bert": "BERTExtSum",
    "textrank": "TextRank"
}

metric_map_name = {
    "rouge1": "Rouge1",
    "rouge2": "Rouge2",
    "rougeL": "RougeL",
    "bert_score": "BERT Score",
    "mover_score": "Mover Score"
}

model_map_name = {
    "mixtral-8x7b-32768": "Mixtral-8x7B",
    "gpt-3.5-turbo-1106": "GPT-3.5-Turbo-1106",
    "gemma-7b-it": "Gemma-7B",
    "Meta-Llama-3-8B-Instruct": "Llama3-8B"
}

exit(0)

sns.set_theme(style="whitegrid")
aux = df_results.copy(True)
aux = aux.sort_values(by=["prompt_type", "model"])
aux = aux[~aux["prompt_type"].str.contains("equal")]
aux["doc_selection_type"] = [doc_selection_map_name["_".join(doc_sel.split("_")[:-1])] for doc_sel in aux["selection_type"]]
aux["sum_ext_type"] = [sum_ext_map_name[doc_sel.split("_")[-1]] for doc_sel in aux["selection_type"]]
aux["score_type"] = [metric_map_name[metric_name] for metric_name in aux["score_type"]]
aux["model"] = [model_map_name[model_name] for model_name in aux["model"]]
aux["prompt_type"] = [prompt_type_map_name[prompt_name] for prompt_name in aux["prompt_type"]]


# original columns 'model', 'selection_type', 'prompt_type', 'score_value', 'score_type', 'doc_selection_type', 'sum_ext_type'
#     aux["selection_type"] = [selection_type_map_name[s] for s in aux["selection_type"]]
#     aux["prompt_type"] = [prompt_type_map_name[p] for p in aux["prompt_type"]]
aux.columns = ["LLM", "Selection Type", "Prompt", "Value", "Metric", "Document Selection Type", "Extractive Summarizer"]

fig = sns.relplot(data=aux, x="Prompt", y="Value", col="Metric", row="Extractive Summarizer", hue="Document Selection Type", style="LLM", s=100, edgecolor="black")

# ### add hatches to artists

### jitter points
jitter = 0.27
for ax_row in fig.axes:
    for ax in ax_row:
        points = ax.collections[-1]
        offsets = points.get_offsets() 
        ax.set(xlim=(-0.7,2.7))
        title_text = ax.title.get_text().replace(" | ", "\n")
        ax.title.set_text(title_text)
        ax.title.set_size(17)
        ax.yaxis.label.set_size(17)
        ax.xaxis.label.set_size(17)
        # every 6, 1 model + 2 selection_doc+ 3 prompts
        inner_point_jitter = [[-0.22, 0.0], [-0.18, 0.0], [-0.02, 0.0], [0.02, 0.0], [0.18, 0.0], [0.22, 0.0]]
        collection_size = 6
        for idx in range(0, offsets.shape[0], collection_size):
            of_ar = [-jitter+(idx//6*jitter), 0]
            offsets[idx:idx+collection_size] = offsets[idx:idx+collection_size] + np.asarray([of_ar]*collection_size) + np.asarray(inner_point_jitter)
        points.set_offsets(offsets)

fig.tick_params(axis='both', which='major', labelsize=16)
legend = fig.figure.legend(fontsize=22, frameon=False, bbox_to_anchor=(1.235,0.5), loc=5)
plt.tight_layout()
fig._legend.remove()
legend.get_texts()[0].set_fontweight("bold")
legend.get_texts()[3].set_fontweight("bold")

plt.savefig(f"combined_res_{dataset_name}.png", bbox_inches = "tight", transparent=True)


# In[ ]:


# from matplotlib.patches import Rectangle, Patch
# import seaborn as sns

# sns.set_theme(style="whitegrid")
# def barplot_results(df_results, metric):
#     aux = df_results.copy(True)
#     aux = aux[aux["score_type"] == metric]
#     aux["input_size"] = ["Maximum context" if "equal" not in p else "Limited context" for p in aux["prompt_type"]]
#     aux["prompt_type"] = ["".join(p.split("_")[1:]) if "equal" in p else p for p in aux["prompt_type"]]

#     selection_type_map_name = {
#         'first5last5_bert': "First5Last5 BERTExtSum", 
#         'random_selection_bert': "Random10 BERTExtSum", 
#         'random_selection_textrank': "Random10 TextRank",
#         'first5last5_textrank': "First5Last5 TextRank"
#     }

#     prompt_type_map_name = {
#         "cod": "CoD",
#         "basic": "Basic",
#         "detailed": "Detailed"
#     }

#     aux["selection_type"] = [selection_type_map_name[s] for s in aux["selection_type"]]
#     aux["prompt_type"] = [prompt_type_map_name[p] for p in aux["prompt_type"]]
#     aux.columns = ["Model", "Selection Type", "Prompt Type", "Score Value", "Metric", "Context Size"]

#     fig = sns.catplot(kind="bar", data=aux, x="Selection Type", y="Score Value", hue=aux[["Model", "Context Size"]].apply(lambda x: f"Model={tuple(x)[0]}\nContext Size={tuple(x)[1]}", axis=1), col="Prompt Type")
#     for ax in fig.axes[0]:
#         ax.tick_params(axis="x", rotation=45)

#     children = [c.get_label() for c in fig.axes[0][0].get_children()]
#     children = [c for c in fig.axes[0][0].get_children() if type(c) == Rectangle]
#     colors = [(c.get_facecolor(), c.get_hatch()) for c in children]
#     colors = [(children[c_idx]) for c_idx in range(len(children)) if c_idx % 2 == 0]
    
#     print(colors)

#     # handles = [Patch(facecolor=color, hatch=hatch, edgecolor="black") for color, hatch in colors]
#     # labels = [c.get_label() for c in children]
#     # handles = [(handles[idx], handles[idx+3]) for idx in range(0,len(handles)-3,1)]
#     # handles += [Patch(facecolor="white", hatch="///", edgecolor="black")]
#     # handles += [Patch(facecolor="white", hatch="", edgecolor="black")]
#     # labels = [labels[idx] for idx in range(0,len(labels)-3,1)]
#     # labels += ["Equal Size Input"]
#     # labels += ["Input to Context Limit"]

#     fig.legend.set_title("Model type and context size")
#     fig.fig.subplots_adjust(top=0.85)

#     metric = metric.title() if metric != "bert_score" else "BERT Score"
#     metric = "RougeL" if metric == "Rougel" else metric
#     fig.fig.suptitle(metric)
#     fig.fig.savefig(f"{metric}.png", transparent=True, bbox_inches="tight")

# for metric in df_results["score_type"].unique()[:1]:
#     barplot_results(df_results, metric)


# In[14]:


# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Patch
# from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
# import seaborn as sns

# def composite_barplot(ax, prompt_type, score_type):
#     subset = df_results.copy()
#     subset = subset[subset["score_type"] == score_type]
#     subset = subset[subset["prompt_type"].str.contains(prompt_type)]
#     subset_eq = subset[subset["prompt_type"].str.contains("equal")]
#     subset_normal = subset[~subset["prompt_type"].str.contains("equal")]

#     sns.set_theme(style="whitegrid")
#     # sns.catplot(kind = "bar", data = subset, x = "selection_type", y = "score_value", hue = "model", col = "score_type", row="prompt_type")
#     plot_eq = sns.barplot(ax=ax, data=subset_eq, y="selection_type", x="score_value", hue="model", alpha=1, palette=sns.color_palette(["red", "orange", "green"]), hatch="///", orient="h", edgecolor="black")
#     plot_normal = sns.barplot(ax=ax, data=subset_normal, y="selection_type", x="score_value", hue="model", alpha=0.8, orient="h")
#     plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
#     children = [c for c in plot_normal.axes.get_children() if type(c) == Rectangle and (c.get_label() != "_nolegend_" and c.get_label() != "")]
#     colors = [(c.get_facecolor(), c.get_hatch()) for c in children]
#     handles = [Patch(facecolor=color, hatch=hatch, edgecolor="black") for color, hatch in colors]
#     labels = [c.get_label() for c in children]
#     handles = [(handles[idx], handles[idx+3]) for idx in range(0,len(handles)-3,1)]
#     handles += [Patch(facecolor="white", hatch="///", edgecolor="black")]
#     handles += [Patch(facecolor="white", hatch="", edgecolor="black")]
#     labels = [labels[idx] for idx in range(0,len(labels)-3,1)]
#     labels += ["Equal Size Input"]
#     labels += ["Input to Context Limit"]
#     print(handles)
#     plot_normal.legend().remove()
#     plot_eq.legend().remove()
#     plt.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.05,1), frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)}, handleheight=3, handlelength=5)
#     sns.despine()
#     plot_eq.set_title(prompt_type.title())
#     plot_eq.set_xlabel(score_type)
    

# sns.set_theme(style = "whitegrid")
# for score in df_results["score_type"].unique():
#     fig, axes = plt.subplots(1, 3, figsize=(16,9), sharey=True)
#     composite_barplot(axes[0], "basic", score)
#     composite_barplot(axes[1], "detailed", score)
#     composite_barplot(axes[2], "cod", score)
#     fig.savefig(f"{score}.png", transparent=True)
#     # plt.show()

