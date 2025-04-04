# Memo: Using GPU oin Pytorch

import torch

import argparse

# Using parser
parser = argparse.ArgumentParser(description="Selecting device for Pytorch")
parser.add_argument(
    "--device",
    type=str,
    choices=["cuda", "cpu"],
    default="cuda" if torch.cuda.is_available() else "cpu",
    help= "Device to use, cuda or cpu"
)
args = parser.parse_args()

# Usage
device = args.device
print(f"Device selectd, using {device}")

# Output:
'''Device selectd, using cuda'''