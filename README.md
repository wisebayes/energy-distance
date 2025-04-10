Energy Distance Project Training and Inference Instructions

Setting up Python Environment and Installing Required Libraries
1. conda create --name myenv39 python=3.9
2. conda activate myenv39
3. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
4. git clone https://github.com/gnatesan/sentence-transformers-3.4.1.git
5. git clone https://github.com/gnatesan/mteb-1.34.14.git
6. pip install -e /path_to_sentence-transformers/sentence-transformers-3.4.1
7. pip install -e /path_to_mteb/mteb-1.34.14

Model Training
1. 
