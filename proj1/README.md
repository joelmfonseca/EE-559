# EE-559 Deep Learning - Project 1

The first project developed for the Deep Learning course was about predicting which digit from a 2-channel image was bigger than the other. The goal was to try different architectures: standard, using weight sharing or auxiliary loss. We also implemented an additional architecture where the comparision between the digits is hardcoded and the digit recognition is the only task.

The content of this project is composed of different parts:

- the folder `figures` contains all the figures used for the development of this project. 

- the folder `old_material` contains a notebook used for exploration done at the very beginning of the project.

- several `.py` files:

    - **`dlc_practical_prologue.py`**: contains help functions given by the course.

    - **`models.py`**: contains all the models implemented throughout the project.

    - **`settings.py`**: defines the main settings used for the training (nb epochs, batch size, learning rate).

    - **`test.py`**: is the main file which can be run.

    - **`train.py`**: contains the different training functions for every type of model.

    - **`utils.py`**: contains utilitary functions used throughout the project.