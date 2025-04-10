# Energy Distance Project Training and Inference Instructions

## Setting up Python Environment and Installing Required Libraries
1. conda create --name myenv39 python=3.9
2. conda activate myenv39
3. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
4. git clone https://github.com/gnatesan/sentence-transformers-3.4.1.git
5. git clone https://github.com/gnatesan/mteb-1.34.14.git
6. pip install -e /path_to_sentence-transformers/sentence-transformers-3.4.1
7. pip install -e /path_to_mteb/mteb-1.34.14
8. git clone https://github.com/gnatesan/beir.git

## Sanity Check
1. conda create --name testenv python=3.9
2. conda activate testenv
3. pip install sentence-transformers==3.5.0
4. pip install mteb==1.34.14
5. 

## Model Training
1. cd /path_to_beir/beir/examples/retrieval/training
2. Before running training, make sure the model, model_name, and hyperparameters (LR, scale) are correct. nano train_sbert_latest_2.py or nano train_sbert_ddp_2.py. nano MultipleNegativesRankingLoss.py to change scale. 
3. sbatch train.sh OR sbatch train_ddp.sh if using multiple GPUs
4. 

## IMPORTANT FILES
1. train.sh - Batch script to run model training on a single GPU.  
2. train_ddp.sh - Batch script to run model training on multiple GPUs. Make sure number of GPUs requested are properly set. 
3. train_sbert_2.py - Python script to run model training on a single GPU. Uses ir_evaluator to evaluate on a dev set after each epoch of training, make sure ir_evaluator is enabled.
4. train_sbert_ddp_2.py - Python script to run model training on multiple GPUs using DDP. Currently does not use an ir_evaluator to evaluate on a dev set after each epoch of training. 
