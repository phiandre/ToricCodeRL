# ToricCodeRL

Contains the code for https://arxiv.org/abs/1811.12338

**Quantum error correction for the toric code using deep reinforcement learning**

_Philip Andreasson, Joel Johansson, Simon Liljestrand, Mats Granath_

# Training a new RL agent
1) In AllSettings.py, choose the desired parameters for the training. The most important parameter to specify is the size of the training data to be generated.

2) In the main method of GenerateData.py, specify the size of the toric code. Then run GenerateData.py to generate training data.

3) Run run_train.py. The network parameters are saved in TrainedNetwork.h5. 

# Testing a trained RL agent
1) Specify the name of the network that you want to test, in the constructor of run_ready.py. You can either use your own trained network or one of the three provided networks for sizes 3, 5 and 7.

2) In the Generate method of GenerateData.py, specify the size of the toric code. This size must be the same as the trained network used.

3) In the run method of run_ready.py, insert the error rates that you want to test in the array testProbs. The results are continually printed in the terminal.
