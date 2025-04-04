# Memo: Using GPU in Pytorch

# Import several packages
import torch
import argparse
import numpy as np


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
print(f"Device selected, using {device}")

# Output:
'''Device selected, using cuda'''

# Move Tensor on Gpu

def puttingModels():
    """_summary_: Moving Models on GPU, using the Pretrained ResNet18
    """
    from torchvision.models import resnet18
    from torchvision.models import ResNet18_Weights  
    
    model = torch.nn.DataParallel(resnet18(weights = ResNet18_Weights.DEFAULT))
    model = model.to(device = device)
    print("Moving to GPU successfully!")

def puttingTensors(input_tensor: torch.Tensor):
    """Putting the tensors from CPU to GPU

    Args:
        input_tensor (torch.tensor): the original tensor on CPU
    """
    print("Original Tensor:")
    print(input_tensor)
    print("Original Devices:")
    print(input_tensor.device)

    # Moving the tensors on GPU
    tensor_on_gpu = input_tensor.to(device)
    print("After moving to GPUs")
    print("Current devices:")
    print(tensor_on_gpu.device)

    return tensor_on_gpu

def backCPU(tensor_on_gpu: torch.Tensor):
    """_summary_: Moving Tensors Back to CPU (numpy)
    """
    tensor_back = tensor_on_gpu.cpu().numpy()
    print("Moving Back on CPU")
    print(f"Devices: {tensor_back.device}")
    print(f"Types: {type(tensor_back)}")
    return tensor_back


if __name__ == "__main__":
    print("Test importing a pretrained model and put it on devices...")
    puttingModels()

    print("\nTest putting tensors on GPU")
    test_tensor = torch.tensor([[1,2,4,5],[3,4,5,6]])
    tensor_on_gpu = puttingTensors(test_tensor)

    print("\nMoving Tensors back to CPU")
    tensor_back = backCPU(tensor_on_gpu)





