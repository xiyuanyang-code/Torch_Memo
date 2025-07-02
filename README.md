# Torch Memo

This is my **PyTorch study memo**.

For a programming language, reading and trying to build your own code in practice is a much more efficient learning mode compared to monotonous and boring courses and books! In this repository, I will open source all the test codes I have written while learning PyTorch, hoping to be helpful to you.

## Table of Contents

- `construct.py`: Checking Dependicies.

- `GPU_usage`:

  - `test_gpu_speed`: Using s simplfied matrix opeation to test speed on GPU, and showing basic information of cuda using `torch.cuda`.

  - `using_gpu.py`: Demos for using GPU to make parallel computations in Pytorch.

- `Tensor`:

  - `Tensor_operations.ipynb`: Basic Operations for **Tensors** in Pytorch.

- `Model_Training`: **Templates** for classic Deep Learning Methods and Neural Network Trainings.

  - `Linear_Regression.py`: Basic Linear Regression Model.

  - `CNN_basic` (**dir**): Basic demo for convolutional neural network.

  - `ResNet_Kaggle_Competition` (**dir**): A simple Kaggle image classifier competition for MNIST dataset.

- `Tensor_Board`: Basic tutorial for using tensorboard in torch, including demonstrations.

## To be done in the future

- More techniques for loading data and writing own classes for data transformation

- Optimizer (Adam, e.g.)

- More advanced and complex neural networks, including the usage of `torch.nn` module.