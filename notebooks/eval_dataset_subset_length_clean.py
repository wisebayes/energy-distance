# ---------------------------
# Imports and Setup
# ---------------------------
import time
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch
from mteb.tasks.Retrieval import HotpotQA, FEVER
import random
import numpy as np

# Check if GPU is available, otherwise switch to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, switching to CPU")

print("Using device:", device)

# Define the sentence-transformers model name
# MAKE SURE THE MODEL, DATASET, HYPERPARAMETERS, AND QUERY LENGTHS BEING USED ARE CORRECT
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr1e-05-epochs10-zeropadding_noepsilon_temperature200_full_test_sanitycheck"
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr2e-05-epochs10-temperature100_full_dev_test_queries_less_than_13"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs10-temperature20_full_dev_test_queries_13_to_17"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs10-temperature200_full_test"
model_name = "distilbert-base-uncased-hotpotqa-lr3e-5-epochs10_full_dev_norm_1_10/checkpoint-23900"
#model_name = "Snowflake/snowflake-arctic-embed-m-v1.5_queries_13_to_17"
start_time = time.time()  # Record the start time

# Load the model
#model = SentenceTransformer("/moto/home/ggn2104/beir/examples/retrieval/training/output/distilbert-base-uncased_CosSim-fever-lr2e-05-epochs10-temperature20_full_dev")
model = SentenceTransformer("/insomnia001/depts/edu/COMSE6998/ck3255/beir/examples/retrieval/training/output/distilbert-base-uncased-hotpotqa-lr3e-5-epochs10_full_dev_norm_1_10/checkpoint-23900")
#model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
model = model.to(device)  # Ensure the model is loaded to the correct device

assert not np.isnan(model.encode(["test"])).any()

# ---------------------------
# Dataset Loading and Subsetting
# ---------------------------
# Load the task from MTEB
#hotpotqa_task = HotpotQA()
hotpotqa_task = FEVER()
hotpotqa_task.load_data()
assert len(hotpotqa_task.queries["test"]) > 0
# Print the total number of queries in the full test set
total_queries_test = len(hotpotqa_task.queries["test"])
print(f"Total number of queries in the full FEVER test set: {total_queries_test}")

# Print the total number of documents in the full HotpotQA test set
total_documents_test = len(hotpotqa_task.corpus["test"])
print(f"Total number of documents in the full FEVER test set: {total_documents_test}")

# Create a subset of queries where the length of each query is exactly 15 words
subset_query_ids = [
    key for key, query in hotpotqa_task.queries["test"].items()
    if len(query.split()) == 15
]

# Print the total number of queries with length 15 words
print(f"Total number of queries with 15 words: {len(subset_query_ids)}")

# Create a subset of relevant docs based on the selected queries
subset_relevant_docs = {
    key: hotpotqa_task.relevant_docs["test"][key]
    for key in subset_query_ids if key in hotpotqa_task.relevant_docs["test"]
}
print(f"Total number of relevant documents in the subset: {len(subset_relevant_docs)}")
# Subset the queries based on the selected queries with exactly 15 words
subset_queries = {
    key: hotpotqa_task.queries["test"][key] for key in subset_query_ids
}

# Update the task's queries and relevant_docs with the subset
hotpotqa_task.queries["test"] = subset_queries
hotpotqa_task.relevant_docs["test"] = subset_relevant_docs  # Update the relevant_docs for test

# ---------------------------
# Evaluation
# ---------------------------
# Create an MTEB evaluation instance with the modified task
evaluation = MTEB(tasks=[hotpotqa_task])
print("Evaluation instance created")
# Run the evaluation
results = evaluation.run(model, verbosity=3, eval_splits=["test"], output_folder=f"results/{model_name}_Hamming")
print("Evaluation results obtained")
end_time = time.time()  # Record the end time

# Calculate the duration
duration_seconds = end_time - start_time
hours = int(duration_seconds // 3600)
minutes = int((duration_seconds % 3600) // 60)
seconds = int(duration_seconds % 60)

print(f"Model evaluation took: {hours} hours, {minutes} minutes, and {seconds} seconds.")
print(f"Total number of queries in the subset: {len(hotpotqa_task.queries['test'])}")
print(f"Total number of documents in the subset corpus: {len(hotpotqa_task.corpus['test'])}")
