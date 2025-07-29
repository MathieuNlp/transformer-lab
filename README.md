# transformer-lab
# GPT-2 Model
## Architecture
- [x] Causal Self attention layer
- [x] Feed forward layer with dropout and ReLU
- [x] Multi head self attention
- [ ] Subblock of the transformer
    - [ ] Add residual layers
    - [ ] Add pre layer norm
- [ ] Add final layer norm before softmax
- [ ] Final softmax layer

## Data preparation
- [ ] Create data loader
- [ ] train/valid/test set split
- [ ] Verify data tokenization + EOS tokens
- [ ] Padding for batches
- [ ] Attention mask
- [ ] Collate if necessary 

## Training
- [ ] Optimizer
- [ ] Loss function
- [ ] Training loop
    - [ ] Mini batching
    - [ ] SGD
    - [ ] torch compile
- [ ] Checkpointing
- [ ] Plots
[ ] Tensorboard pytorch
- [ ] Distributed training [!url](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

## Evals
- [ ] Metrics
    - [ ] Token / second
    - [ ] Perplexity
    - [ ] Accuracy
    - [ ] Execution time

## Sampling
- [ ] Sampling loop with text answer
- [ ] Add metrics to sampling
- [ ] API