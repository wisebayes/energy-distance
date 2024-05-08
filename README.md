# Energy Distance for RAG Systems in LLMs and Sentence Transformers

## Project Description
This project explores the application of energy distance as an alternative metric to cosine similarity in RAG systems used with LLMs and Sentence Transformers. The aim is to improve the accuracy and relevance of retrieved documents by using a metric that potentially captures finer nuances in sentence embeddings.

Our approach capitalizes on existing benchmarking frameworks, traditionally used to evaluate the efficacy of cosine similarity with fully trained LLMs. We have introduced custom implementations of energy distance into these well-established, open-source frameworks. By employing the same benchmarks universally recognized in the research community, specifically Massive Text Embedding Benchmark ([MTEB](https://huggingface.co/blog/mteb#:~:text=MTEB%20is%20a%20massive%20benchmark,on%20a%20variety%20of%20tasks.)), we establish a robust, scientific, and quantifiable basis to assess whether energy distance offers any substantive advantages over cosine similarity. Benchmarks currently utilized originate from [Hugging Face](https://huggingface.co) and [SBERT](https://www.sbert.net). Our modifications to these frameworks are accessible via the following links:

[Sentence Transformer with Energy Distance](https://github.com/gnatesan/mteb-evaluator)

[MTEB Evaluator](https://github.com/gnatesan/mteb-evaluator)


More results are coming, so this repo may change.

## Repository Outline


## How to Run


## Results
