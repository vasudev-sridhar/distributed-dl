# Source: https://towardsdatascience.com/distributed-neural-network-training-in-pytorch-5e766e2a9e62

import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# prase the local_rank argument from command line for the current process
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# setup the distributed backend for managing the distributed training
torch.distributed.init_process_group('nccl')

# Setup the distributed sampler to split the dataset to each GPU.
dist_sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=dist_sampler)

# set the cuda device to a GPU allocated to current process .
device = torch.device('cuda', args.local_rank)
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                  output_device=args.local_rank)

# Start training the model normally.
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    preds = model(inputs)
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()

# TO start the process run the following command from terminal.
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=1234 distributedDataParallel.py <OTHER TRAINING ARGS>