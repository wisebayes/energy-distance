# Energy Distance for RAG Systems in LLMs and Sentence Transformers

## Project Description
This project explores the application of [energy distance](https://en.wikipedia.org/wiki/Energy_distance) as an alternative metric to [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) in RAG systems used with LLMs and Sentence Transformers. The aim is to improve the accuracy and relevance of retrieved documents by using a metric that potentially captures finer nuances in sentence embeddings.

Our approach capitalizes on existing benchmarking frameworks, traditionally used to evaluate the efficacy of cosine similarity with fully trained LLMs. We have introduced custom implementations of energy distance into these well-established, open-source frameworks. By employing the same benchmarks universally recognized in the research community, specifically Massive Text Embedding Benchmark ([MTEB](https://huggingface.co/blog/mteb#:~:text=MTEB%20is%20a%20massive%20benchmark,on%20a%20variety%20of%20tasks.)), we establish a robust, scientific, and quantifiable basis to assess whether energy distance offers any substantive advantages over cosine similarity. Benchmarks currently utilized originate from [Hugging Face](https://huggingface.co) and [SBERT](https://www.sbert.net). Our modifications to these frameworks are accessible via the following links:

**[Sentence Transformer with Energy Distance](https://github.com/gnatesan/sentence-transformers-energydistance)**

We have extended the functionality of the Sentence Transformer by integrating an energy distance calculation into the Contrastive Loss module. This adaptation aims to minimize the energy distance between embeddings of similar sentences during model training, promoting more effective similarity measures. Additionally, to enhance query representation, we adapted the model to generate multivector embeddings for queries, ensuring detailed and nuanced semantic capture.

**[MTEB Evaluator](https://github.com/gnatesan/mteb-evaluator)**

This includes enhancements to the Massive Text Embedding Benchmark, a tool designed to assess information retrieval systems. We incorporated energy distance as a novel metric to refine the selection of top matches for a given query within this framework. To accommodate multivector queries against single-vector documents, we modified the query encoding process, generating token embeddings that better capture the complexity of the queries.

## Repository Outline
- `README.md`: Description of repository contents
- `requirements.txt`: Required python packages
- `/models`: Trained sentence transformer models
- `/notebooks`: Source code used to train sentence transformers and to run benchmarks
- `/results`: Benchmark results

## How to Run

**Pre-requisites:**
- GPU VRAM >= 12GB
- Python >= 3.8

```bash
pip install -r requirements.txt
```

To run the code, follow the instructions included in the python notebooks under `/notebooks`. Benchmarks can be run from MTEB.ipynb. Sentence transformer training can be run from train_energy_sentence_transformer.ipynb.


## Results

More results are coming, so this repo may change.

| Model and Benchmark | Cosine Similarity | Energy Distance |
|----------|:--------:|---------:|
| NULL   | NULL   |  NULL  |
