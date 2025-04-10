import time
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch
from mteb.tasks.Retrieval import HotpotQA, FEVER
import random

# Check if GPU is available, otherwise switch to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, switching to CPU")

print("Using device:", device)

# Define the sentence-transformers model name MAKE SURE THE MODEL, DATASET, HYPERPARAMETERS, AND QUERY LENGTHS BEING USED ARE CORRECT
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr1e-05-epochs10-zeropadding_noepsilon_temperature200_full_test_sanitycheck"
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr2e-05-epochs10-temperature100_full_dev_test_queries_less_than_13"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs10-temperature20_full_dev_test_queries_13_to_17"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs10-temperature200_full_test"
model_name = "distilbert-base-uncased_ED-fever-lr2e-05-epochs10-temperature20_full_dev_queries_15"
#model_name = "Snowflake/snowflake-arctic-embed-m-v1.5_queries_13_to_17"
start_time = time.time()  # Record the start time

# Load the model
#model = SentenceTransformer("/moto/home/ggn2104/beir/examples/retrieval/training/output/distilbert-base-uncased_CosSim-fever-lr2e-05-epochs10-temperature20_full_dev")
model = SentenceTransformer("/moto/home/ggn2104/beir/examples/retrieval/training/output/distilbert-base-uncased_ED-fever-lr2e-05-epochs10-temperature20_full_dev")
#model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
model = model.to(device)  # Ensure the model is loaded to the correct device

# Load the task from MTEB
#hotpotqa_task = HotpotQA()
hotpotqa_task = FEVER()
hotpotqa_task.load_data()

# Print the total number of queries in the full test set
total_queries_test = len(hotpotqa_task.queries["test"])
print(f"Total number of queries in the full FEVER test set: {total_queries_test}")

# Print the total number of documents in the full HotpotQA test set
total_documents_test = len(hotpotqa_task.corpus["test"])
print(f"Total number of documents in the full FEVER test set: {total_documents_test}")

# Create a subset of queries where the length of each query is at least 20 words
subset_query_ids = [
    key for key, query in hotpotqa_task.queries["test"].items()
    if len(query.split()) == 15
]

# Print the total number of queries with length 16-17 words
print(f"Total number of queries with 15 words: {len(subset_query_ids)}")

# Create a subset of relevant docs based on the selected queries
subset_relevant_docs = {
    key: hotpotqa_task.relevant_docs["test"][key]
    for key in subset_query_ids if key in hotpotqa_task.relevant_docs["test"]
}

# Subset the queries based on the selected queries with at least 18 words
subset_queries = {
    key: hotpotqa_task.queries["test"][key] for key in subset_query_ids
}


# You can now use subset_queries and subset_relevant_docs for further processing

# Collect all relevant document IDs
#relevant_doc_ids = set(doc_id for docs in subset_relevant_docs.values() for doc_id in docs)

# Now make sure that corpus still has the "test" key with the subset of corpus data
hotpotqa_task.queries["test"] = subset_queries
hotpotqa_task.relevant_docs["test"] = subset_relevant_docs  # Update the relevant_docs for test

# Create an MTEB evaluation instance with the modified task
evaluation = MTEB(tasks=[hotpotqa_task])

# Run the evaluation
results = evaluation.run(model, verbosity=2, eval_splits=["test"], output_folder=f"results/{model_name}")

end_time = time.time()  # Record the end time

# Calculate the duration
duration_seconds = end_time - start_time
hours = int(duration_seconds // 3600)
minutes = int((duration_seconds % 3600) // 60)
seconds = int(duration_seconds % 60)


print(f"Model evaluation took: {hours} hours, {minutes} minutes, and {seconds} seconds.")
#print(f"Total number of queries in the subset: {len(hotpotqa_task.queries["test"])}")
print(f"Total number of queries in the subset: {len(hotpotqa_task.queries['test'])}")
print(f"Total number of documents in the subset corpus: {len(hotpotqa_task.corpus['test'])}")
