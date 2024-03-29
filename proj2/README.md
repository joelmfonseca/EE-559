# EE-559 Deep Learning - Project 2

The second project developed for the Deep Learning course was about implementing a mini deep-learning framework from scratch, i.e., without using the autograd or the neural-network module from PyTorch. Furthermore, a toy dataset was used to validate the implementation.

The content of this project is composed of different parts:

- the folder `figures` contains all the figures used for the development of this project. 

- several `.py` files:

    - **`activation.py`**: contains the Tanh, ReLU, LeakyReLU and PReLU activation functions.

    - **`loss.py`**: contains the MSE and CrossEntropy losses.

    - **`module.py`**: contains the `Module` class as well as the Linear and Sequential modules.

    - **`optimizer.py`**: contains the `Optimizer` class as well as the SGD optimizer.

    - **`test.py`**: is the main file which can be run.

    - **`utils.py`**: contains utilitary functions used throughout the project.

---
**NOTE**

You might need to install `tqdm` to run our project properly (if you use the provided VM). To install it simply run the following command:

```terminal
>> conda install -c conda-forge tqdm
```