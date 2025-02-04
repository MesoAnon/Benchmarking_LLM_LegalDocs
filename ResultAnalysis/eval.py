import sys
import pickle
from datasets import load_dataset
import numpy as np
import os
import torch
import evaluate
import os
import json
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moverscore_v2 import get_idf_dict, word_mover_score


# Set the environment variable for MoverScore to use a custom model
os.environ["MOVERSCORE_MODEL"] = "allenai/longformer-base-4096"

# Dataset mapping configuration based on input argument
dataset_name = sys.argv[-1]
d_map = {
    "multilex_tiny": "summary/tiny",
    "multilex_short": "summary/short",
    "multilex_long": "summary/long",
    "eurlexsum": "reference",
    "eurlexsum_test": "reference",
    "eurlexsum_validation": "reference"
}

# Exit if results for the dataset already exist
if f"results_{dataset_name}.pickle" in os.listdir():
    exit(1)

# Load the appropriate dataset based on the provided name
split = None
if "multilex" in dataset_name:
    dataset = load_dataset("allenai/multi_lexsum", name="v20230518")
    dataset = dataset["test"].filter(lambda x: x[d_map[dataset_name]] != None)[d_map[dataset_name]]
else:
    dataset = load_dataset("dennlinger/eur-lex-sum", "english")
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

print(f"Number of test cases={len(dataset)}")


# Load evaluation metrics
rouge_scoring = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def load_predicted_data(path):
    """
    Load predicted summaries from JSON files in the given directory.

    Args:
        path (str): Path to the directory containing JSON files.

    Returns:
        list: A list of predicted summaries.
    """

    predicted = []
    ordered_files = os.listdir(path)
    ordered_files = sorted(ordered_files, key = lambda x: int(x.split(".")[0]))
    for file in ordered_files:
        predicted.append(json.load(open(path+file, "r"))[-1]["content"])
        print(path, file) if len(json.load(open(path+file, "r"))[-1]["content"]) < 1 else 1

    return predicted


def compute_mover_score(references, predictions, orig_ref, orig_pred):
    """
    Compute MoverScore between reference and predicted summaries.

    Args:
        references (list): List of reference summaries.
        predictions (list): List of predicted summaries.
        orig_ref (list): Original reference summaries.
        orig_pred (list): Original predicted summaries.

    Returns:
        list: List of MoverScores.
    """

    idf_dict_pred = get_idf_dict(orig_pred)
    idf_dict_ref = get_idf_dict(orig_ref)

    scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_pred, n_gram=1, remove_subwords=True, batch_size=1)
    torch.cuda.empty_cache()

    return scores
    
def evaluation(path_data, results, model, prompt_type, selection_type, limit):
    """
    Evaluate predicted summaries using ROUGE, BERTScore, and MoverScore metrics.

    Args:
        path_data (str): Path to the predicted summaries directory.
        results (dict): Dictionary to store evaluation results.
        model (str): Name of the evaluated model.
        prompt_type (str): Type of prompt used during generation.
        selection_type (str): Selection method for documents.
        limit (int): Limit on the number of predictions to evaluate.

    Returns:
        None
    """

    if "eurlexsum" in dataset_name:
        predicted = load_predicted_data(path_data)[split][:limit]
    else:
        predicted = load_predicted_data(path_data)[:limit]

    print(len(predicted))
    r_scores = rouge_scoring.compute(predictions=predicted, references=dataset[:len(predicted)], use_stemmer = True)
    # mover_score = compute_mover_score(references=dataset, predictions=predicted, orig_ref=dataset, orig_pred=predicted)
    # chosen deberta lange due to https://github.com/Tiiiger/bert_score/blob/master/README.md and paper
    # bert_scores = bertscore.compute(predictio ns=predicted, references=dataset[:len(predicted)], model_type="microsoft/deberta-xlarge-mnli", batch_size = 1, verbose=True)
    
    # for some reason the data is all loaded onto the GPU so it doesn't fit anymore
    # we have to use batch-wise computation for BERTScore to reduce memory usage

    bs = 0
    mover = 0
    batch_size = 1
    for pred_idx in tqdm(range(0,len(predicted),batch_size)):
        # in bertscore.py, in evaluate metric bert (from evaluate huggingface), added del of scorer model to reduce VRAM usage
        bs_aux = bertscore.compute(predictions=predicted[pred_idx:pred_idx+batch_size], references=dataset[pred_idx:pred_idx+batch_size], model_type="microsoft/deberta-xlarge-mnli", batch_size = 1)
        # mover_aux = compute_mover_score(references=dataset[pred_idx:pred_idx+batch_size], predictions=predicted[pred_idx:pred_idx+batch_size], orig_ref = dataset, orig_pred=predicted)
        # mover += sum(mover_aux)
        bs += sum(bs_aux["f1"])

    mover_score = mover/len(predicted)
    r_scores = {metric: round(np.mean(val), 3) for metric, val in r_scores.items()}
    bert_scores = {"f1": bs/len(predicted)}

    # mover_score = np.mean(mover_score)
    # bert_scores = {"f1": np.mean(bert_scores["f1"])}

    # bert_scores = {metric: round(np.mean(val), 3) for metric, val in bert_scores.items() if metric != "hashcode"}

    results["model"] += [model] * 5
    results["selection_type"] += [selection_type] * 5
    results["prompt_type"] += [prompt_type] * 5
    results["score_value"] += [r_scores["rouge1"], r_scores["rouge2"], r_scores["rougeL"], bert_scores["f1"], mover_score]
    results["score_type"] += ["rouge1", "rouge2", "rougeL", "bert_score", "mover_score"]

# Initialize results dictionary
results = {"model": [], "selection_type": [], "prompt_type": [], "score_value": [], "score_type": []}
if "eurlexsum" in dataset_name:
    dataset_name_aux = dataset_name.split("_")[0]
    path_dir = f"answers/{dataset_name_aux}"
for model in tqdm(os.listdir(f"{path_dir}/")):
    for prompt_type in os.listdir(f"{path_dir}/{model}/"):
        for selection_type in os.listdir(f"{path_dir}/{model}/{prompt_type}/"):
            path_data = f"{path_dir}/{model}/{prompt_type}/{selection_type}/"
            evaluation(path_data, results, model, prompt_type, selection_type, len(dataset))



# Save results as a pickle file
pickle.dump(results, open(f"results_{dataset_name}.pickle", "wb"))

# Load results from the pickle file
results = pickle.load(open(f"results_{dataset_name}.pickle", "rb"))


# Transform results into a DataFrame for visualization
df_results = pd.DataFrame(results)

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

# Prepare the results DataFrame by filtering, sorting
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

# Rename columns to more descriptive names for visualization
aux.columns = [
    "LLM",                      # Large Language Model name
    "Selection Type",           # Document selection method
    "Prompt",                   # Prompt type
    "Value",                    # Metric value
    "Metric",                   # Metric name
    "Document Selection Type",  # Document selection method in readable form
    "Extractive Summarizer"     # Extractive summarizer name
]

# Generate a relational plot with seaborn
fig = sns.relplot(
    data=aux, 
    x="Prompt", 
    y="Value", 
    col="Metric", 
    row="Extractive Summarizer", 
    hue="Document Selection Type", 
    style="LLM", 
    s=100, 
    edgecolor="black")

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

# Customize legend and save the plot as a transparent PNG.
fig.tick_params(axis='both', which='major', labelsize=16)
legend = fig.figure.legend(fontsize=22, frameon=False, bbox_to_anchor=(1.235,0.5), loc=5)
plt.tight_layout()
fig._legend.remove()
legend.get_texts()[0].set_fontweight("bold")
legend.get_texts()[3].set_fontweight("bold")

plt.savefig(f"combined_res_{dataset_name}.png", bbox_inches = "tight", transparent=True)
