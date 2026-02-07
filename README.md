# GPT 2.5 - Modern and improved reproduction of GPT 2 124M
This repository reproduces GPT 2 124M (completion model) with architectural changes to some components, which are used in all of the latest SOTA models. 

The reproduction started off with building and training a BPE tokenizer that was used originally in training GPT 2. 

Then, was built the model starting from Linear layer all the way to the complete implementation of Transformer Decoder. (All of the components such as Linear, Embedding, softmax were written manually, for the sake of learning. Specifically, torch.nn, torch.nn.functional or torch.optim was not used except for the following):
- torch.nn.Parameter
- Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)
- The torch.optim.Optimizer base class
- torch.nn.functional.scaled_dot_product_attention (for the sake of using flash attention, custom scaled dot product attention was also implemented)

The model was wrapped with a FSDP object to enable distributed training.

Lastly, we tokenized the datasets and implemented the training loop.

The model was then successfully trained, and experiment tracked, the results of which are provided below.

## Differences from the original GPT 2 (following most modern LLMs)
- No bias term in Linears 
- RMSNorm instead of LayerNorm
- Pre-norm instead of post-norm
- SwiGLU instead of GELU
- RoPE instead of absolute positional embeddings
- Flash attention instead of non-optimized attention
- AdamW optimizer instead of Adam
- Different weight initialization, truncation, no residual scaling
- Different dataset, namely fineweb 10B

## Hyperparameters used
- tokens processed per step: 524288
- batch_size: 32
- grad_accum_steps: 2
- total_tokens_processed: ~42B
- training_steps: 80000
- context_length: 1024
- num_layers: 12
- num_heads: 12
- d_model: 768
- d_ff: 2048
- max_lr: 18e-4
- weight_decay: 0.1
- optimizer_betas: 0.9, 0.95
- eps: 1e-8
- temp: 0.8
- top_p: 0.9

## Hardware used
A cluster of 4x H100 SXM5 80GB

## Results achieved
- Train loss: ~3.2
- Validation loss: ~3.12
- Perplexity score: ~26.6
- HellaSwag score: ~0.36

<img src=results.png>
Note: HellaSwag graph looks very erratic is due to the number of examples that was used per single evaluation (it was evaluated every 100 steps), specifically only the batch_size of examples were used to evaluate the performance on the HellaSwag and not the full validation dataset. But nevertheless, the fact that as the training goes, it is visible that the score keeps marking higher lowes and higher highs, which signals model quality improvement.

## Some generated samples (generation of 50 tokens or until hit EOS token)
Starting prompt: "Once upon a time, "

Samples:
- Once upon a time, my brother-in-law and I would be all but broke and in debt. Let me say for the record, I was fully healthy in the summer of 2011 and we did just about everything we could to get a house together, start our
- Once upon a time, we have a chance to see the transformation of technology. It will not be so easy, as it may take many years for it to fully be implemented. We need to know the best ways to use this technology.
- Once upon a time, many people thought that “the young” might be better off staying with the big, fancy houses. But that is not true. Young people are not allowed to live in luxury, with their parents, or with their families.
- Once upon a time, the fact that the Green Bay Packers used to be an underdog of the Super Bowl is a mystery, but it is now part of the legend of the Super Bowl.
In 1986, the Green Bay Packers won the Super Bowl by beating the Seattle Seahawks


## How to run this

This section provides instructions on how to set up the environment, acquire data, and run the various components of the GPT-2.5 project. We use `uv` for dependency management.

### 1. Data Acquisition

You can download the dataset and pre-trained weights directly from the Hugging Face Hub. Note that the dataset download includes both raw text and tokenized binary files (`.bin`).

**Download Dataset (includes tokenized data):**
```bash
uv run hf download itskoma/GPT2.5 --repo-type dataset --local-dir .
```

**Download Pre-trained Checkpoints:**
```bash
uv run hf download itskoma/GPT2.5 --repo-type model --local-dir .
```

### 2. Tokenizer Training

If you wish to train the BPE tokenizer from scratch on your own data:

1. Prepare a raw text file (e.g., `data/fineweb.txt`).
2. Run the training script:
   ```bash
   uv run python tokenizer/train_tokenizer.py
   ```
   *Note: Modify the `input_path` and output paths in `tokenizer/train_tokenizer.py` as needed.*

### 3. Dataset Tokenization (Optional)

If you have downloaded the dataset using the command in Step 1, you already have the tokenized `.bin` files. However, if you have new raw text data, you can tokenize it using:

```bash
uv run python data/src/tokenize_dataset.py
```
*Note: This script uses multiprocessing to speed up tokenization. Adjust `input_path` and `final_output` in the script.*

### 4. Training the Model

#### Training from Scratch
To start training GPT 2.5 from scratch using distributed training:

```bash
uv run torchrun --nproc_per_node 4 train.py \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --context_length 1024 \
    --num_layers 12 \
    --d_model 768 \
    --num_heads 12 \
    --d_ff 2048 \
    --theta 10000 \
    --train_steps 80000 \
    --lr 18e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1e-8 \
    --weight_decay 0.1
```

#### Resuming/Loading from Checkpoint
To resume training or load a specific checkpoint from the Hugging Face Hub (or your own local checkpoint):

```bash
uv run torchrun --nproc_per_node 4 train.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --batch_size 32 \
    ... (other hyperparameters)
```
*(Ensure all hyperparameters match the original training run for consistency.)*
