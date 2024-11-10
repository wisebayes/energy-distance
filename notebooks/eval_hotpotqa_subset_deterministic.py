import time
import random
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch
from mteb.tasks.Retrieval import HotpotQA
import json
import pickle


# Check if GPU is available, otherwise switch to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, switching to CPU")

print("Using device:", device)

# Define the sentence-transformers model name
#model_name = "nq-distilbert-base-v1_ED_v2_hotpotqa_subset_deterministic"
#model_name = "nq-distilbert-base-v1-v1_hotpotqa_subset_deterministic"
#model_name = "nq-distilbert-base-v1_CosSim-hotpotqa-lr3e-05-epochs5_subset_test"
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr3e-05-epochs5_subset_test"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs5_subset_test"
#model_name = "distilbert-base-uncased_ED-hotpotqa-lr2e-05-epochs10_subset_minlength_20_notruncate"
#model_name = "distilbert-base-uncased_CosSim-hotpotqa-lr2e-05-epochs10_subset_minlength_20"
model_name = "distilbert-base-uncased_ED-hotpotqa-lr2e-05-epochs10-zeropadding_noepsilon"
start_time = time.time()  # Record the start time

# Load the model
model = SentenceTransformer("/moto/home/ggn2104/beir/examples/retrieval/training/output/distilbert-base-uncased_ED-hotpotqa-lr2e-05-epochs10-zeropadding_noepsilon")
#model = SentenceTransformer("distilbert-base-uncased")
model = model.to(device)  # Ensure the model is loaded to the correct device

# Load the HotpotQA task from MTEB
hotpotqa_task = HotpotQA()
hotpotqa_task.load_data()

# Print the total number of queries in the full HotpotQA test set
total_queries_test = len(hotpotqa_task.queries["test"])
print(f"Total number of queries in the full HotpotQA test set: {total_queries_test}")

# Print the total number of documents in the full HotpotQA test set
total_documents_test = len(hotpotqa_task.corpus["test"])
print(f"Total number of documents in the full HotpotQA test set: {total_documents_test}")

# Get the keys (query IDs) from the queries for the test set and sort them for deterministic selection
#test_query_ids = sorted(list(hotpotqa_task.queries["test"].keys()))
#subset_size = int(0.20 * len(test_query_ids))  # Calculate 20% of the dev set

# Select the first 20% of the queries based on sorted query IDs
#subset_query_ids = test_query_ids[:subset_size]

# Use all queries from the test set
subset_query_ids = list(hotpotqa_task.queries["test"].keys())

# Create a subset of queries where the length of each query is at least 20 words
#subset_query_ids = [
#    key for key, query in hotpotqa_task.queries["test"].items()
#    if len(query.split()) >= 20
#]

print("Done creating query id subset")

# Create a subset of relevant docs based on the selected queries
subset_relevant_docs = {key: hotpotqa_task.relevant_docs["test"][key] for key in subset_query_ids if key in hotpotqa_task.relevant_docs["test"]}

# Subset the queries based on the selected 20% queries
subset_queries = {key: hotpotqa_task.queries["test"][key] for key in subset_query_ids}

# Collect all relevant document IDs
relevant_doc_ids = set(doc_id for docs in subset_relevant_docs.values() for doc_id in docs)

# Now select irrelevant document IDs deterministically from the corpus, excluding relevant ones
all_doc_ids = set(hotpotqa_task.corpus["test"].keys())
irrelevant_doc_ids = list(all_doc_ids - relevant_doc_ids)  # Exclude relevant doc IDs

# Sort the irrelevant document IDs to make selection deterministic
sorted_irrelevant_doc_ids = sorted(irrelevant_doc_ids)

# Select the first N irrelevant documents to reach a total of 50,000 documents in the subset
print("Deterministic test subset!")
additional_irrelevant_doc_count = 50000 - len(relevant_doc_ids)
if additional_irrelevant_doc_count > 0:
    sampled_irrelevant_doc_ids = sorted_irrelevant_doc_ids[:additional_irrelevant_doc_count]
else:
    sampled_irrelevant_doc_ids = []

# Combine relevant and sampled irrelevant document IDs
final_doc_ids = list(relevant_doc_ids) + sampled_irrelevant_doc_ids

# Subset the corpus based on the final set of document IDs
subset_corpus = {doc_id: hotpotqa_task.corpus["test"][doc_id] for doc_id in final_doc_ids}

# Now make sure that corpus still has the "test" key with the subset of corpus data
hotpotqa_task.corpus["test"] = subset_corpus
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
print(f"Total number of queries in the subset: {len(subset_queries)}")
print(f"Total number of documents in the subset corpus: {len(subset_corpus)}")


# Option 1: Save as JSON
#with open("evaluation_results.json", "w") as f:
#    json.dump(results, f)

# Option 2: Save as Pickle (for complex structures)
#with open("evaluation_results.pkl", "wb") as f:
#    pickle.dump(results, f)
