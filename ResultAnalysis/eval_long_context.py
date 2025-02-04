import sys
import os
import pickle
from datasets import load_dataset
import numpy as np
import torch
import evaluate
import os
import json
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Initialize device and dataset name from command-line arguments
cuda_id = sys.argv[2]
device = torch.device(f"cuda:{cuda_id}")

# os.environ["CUDA_VISIBLE_DEVICES"]=cuda_id
dataset_name = sys.argv[1]

# set env before importing to use custom model
model_path = "../models/"

# os.environ["MOVERSCORE_MODEL"] = model_path + "longformer-base-4096"
# from moverscore_v2 import get_idf_dict, word_mover_score

# Mapping dataset names to their relevant fields
d_map = {
    "multilex_tiny": "summary/tiny",
    "multilex_short": "summary/short",
    "multilex_long": "summary/long",
    "eurlexsum": "reference",
    "eurlexsum_test": "reference",
    "eurlexsum_validation": "reference"
}

# Exit if results already exist
if f"results_{dataset_name}.pickle" in os.listdir():
    exit(1)

print(f"Device: {device}")
print("Loading dataset")

# Load dataset and configure slices for specific splits
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

# Load evaluation metrics
rouge_scoring = evaluate.load("rouge")
bertscore = evaluate.load("bertscore", device=device)

def load_predicted_data(path):
    """
    Load predicted summaries from JSON files.

    Args:
        path (str): Path to the directory containing predicted summaries.

    Returns:
        list: List of predicted summaries.
    """
    predicted = []
    ordered_files = os.listdir(path)
    ordered_files = sorted(ordered_files, key = lambda x: int(x.split(".")[0]))
    for file in ordered_files:
        predicted.append(json.load(open(path+file, "r"))[-1]["content"])
        print(path, file) if len(json.load(open(path+file, "r"))[-1]["content"]) < 1 else 1

    return predicted


def compute_mover_score(predictions, references, orig_ref, orig_pred):
    """
    Compute the MoverScore for predictions and references.

    Args:
        predictions (list): Predicted summaries.
        references (list): Reference summaries.
        orig_ref (list): Original reference text for IDF.
        orig_pred (list): Original predicted text for IDF.

    Returns:
        list: List of MoverScore values.
    """

    idf_dict_pred = get_idf_dict(orig_pred)
    idf_dict_ref = get_idf_dict(orig_ref)

    print("Computed idf dictionaries for mover score")

    scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_pred, n_gram=1, remove_subwords=True, batch_size=3, device=device)
    torch.cuda.empty_cache()

    print("Mover score done")

    return scores
    
def evaluation(path_data, results, model, prompt_type, selection_type, limit):
    """
    Evaluate predicted summaries against reference summaries using ROUGE, BERTScore, and MoverScore.

    Args:
        path_data (str): Path to the predicted summaries.
        results (dict): Dictionary to store evaluation results.
        model (str): Model name.
        prompt_type (str): Type of prompt used during generation.
        selection_type (str): Document selection type.
        limit (int): Number of test cases to evaluate.

    Returns:
        None
    """

    if "eurlexsum" in dataset_name:
        predicted = load_predicted_data(path_data)[split][:limit]
    else:
        predicted = load_predicted_data(path_data)[:limit]

    print(f"Number of predictions:{len(predicted)}")
    r_scores = rouge_scoring.compute(predictions=predicted, references=dataset[:len(predicted)], use_stemmer = True)
    bert_scores = bertscore.compute(predictions=predicted, references=dataset[:len(predicted)], model_type="microsoft/deberta-xlarge-mnli", batch_size = 1, verbose=True, device=device)
    # mover_score = compute_mover_score(predictions=predicted, references=dataset[:len(predicted)], orig_ref=dataset, orig_pred=predicted)

    # Average the scores
    # mover_score = np.mean(mover_score)
    mover_score = 0 # Placeholder if MoverScore isn't used
    bert_scores = {"f1": np.mean(bert_scores["f1"])}
    r_scores = {metric: round(np.mean(val), 3) for metric, val in r_scores.items()}

    # Add results to the results dictionary
    results["model"] += [model] * 5
    results["selection_type"] += [selection_type] * 5
    results["prompt_type"] += [prompt_type] * 5
    results["score_value"] += [r_scores["rouge1"], r_scores["rouge2"], r_scores["rougeL"], bert_scores["f1"], mover_score]
    results["score_type"] += ["rouge1", "rouge2", "rougeL", "bert_score", "mover_score"]

# Initialize results dictionary and evaluate models
results = {"model": [], "selection_type": [], "prompt_type": [], "score_value": [], "score_type": []}
path_dir = f"answers/{dataset_name}" if "multilex" in dataset_name else f"answers/{dataset_name.split('_')[0]}"
iter = 0

# Iterate through models, prompt types, and selection methods
for model in tqdm(os.listdir(f"{path_dir}/")):
    for prompt_type in os.listdir(f"{path_dir}/{model}/"):
        for selection_type in os.listdir(f"{path_dir}/{model}/{prompt_type}/"):
            path_data = f"{path_dir}/{model}/{prompt_type}/{selection_type}/"
            evaluation(path_data, results, model, prompt_type, selection_type, len(dataset))
            iter += 1
            print(f"Iteration {iter}/{4*3*4}")



# Save results to a pickle file
pickle.dump(results, open(f"results_{dataset_name}.pickle", "wb"))

# Load results from a pickle file
results = pickle.load(open(f"results_{dataset_name}.pickle", "rb"))

exit(1)


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

# Rename columns for better readability
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

# Finalize and save the plot
fig.tick_params(axis='both', which='major', labelsize=16)
legend = fig.figure.legend(fontsize=22, frameon=False, bbox_to_anchor=(1.235,0.5), loc=5)
plt.tight_layout()
fig._legend.remove()
legend.get_texts()[0].set_fontweight("bold")
legend.get_texts()[3].set_fontweight("bold")
plt.savefig(f"combined_res_{dataset_name}.png", bbox_inches = "tight", transparent=True)

