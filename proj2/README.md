# EE-559 Deep Learning - Project 2

The second project developed for the Deep Learning course was about implementing a mini deep-learning framework from scratch, i.e., without using the autograd or the neural-network module from PyTorch. Furthermore, a toy dataset was used to validate the implementation.

The content of this project is composed of different parts:

- the folder `figures` contains all the figures used for the development of this project. 

- several `.py` files:

    - **`utils.py`**: contains utilitary functions used throughout the project.

    - **`module.py`**: contains the `Module` class as well as the Linear and Sequential modules.

    - **`activation.py`**: contains the Tanh, ReLU, LeakyReLU and PReLU activation functions.

    - **`loss.py`**: contains the MSE and CrossEntropy losses.

    - **`optimizer.py`**: contains the `Optimizer` class as well as the SGD optimizer.

    - **`test.py`**: is the main file which can be run.