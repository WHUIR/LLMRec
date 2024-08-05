Quick Start:
2. Download amazon dataset meta data(.json.gz) and rating data(.csv) files to dataset/Metadata and dataset/Ratings respectively.
2. python preprocess_amazon.py
3. python run_llmrec.py

Note:
1. The code is based on Torch's DDP (Distributed Data Parallel) and defaults to using 4 GPUs for training.
2. The code set the backbone to bert-base-uncased in default
3. This code defaults to directly accessing the Hugging Face website to download pre-trained models.

