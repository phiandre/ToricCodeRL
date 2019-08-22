"""""""""""""""""""""""""""""""""""""""""""""""""""

Policy for the approximation of the Q-function.

Utilizes a QNet object as the associated neural
network. Selects appropriate action based on an
epsilon-greedy policy.

"""""""""""""""""""""""""""""""""""""""""""""""""""

# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import clone_model
from QNet import QNet
import keras
import math
import numpy as np
import pandas as pd


# Class definition
class RLsys:

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	RL class constructor.
		@param
			actions: the possible actions of the system.
			state_size: the size of the state matrix.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, actions, state_size, optt='adam', miniBatchSize = 32, TNRate = 100 , memorySize = 1000000, reward_decay=0.95, e_greedy=0.9):
		# Save parameters for later use
		self.state_size = state_size
		self.actions = actions
		self.gamma = reward_decay
		self.epsilon = e_greedy
		# Produce neural network
		self.qnet = QNet(self.state_size,optt)
		self.targetNet = QNet(self.state_size,optt)
		self.targetNet.network = clone_model(self.qnet.network)
		self.memorySize = memorySize
		self.memory = list()
		self.TNRate = TNRate
		self.count = 0
		self.miniBatchSize = miniBatchSize
	
	def storeTransition(self, oldState, action, reward, newState):
		if len(self.memory) >= self.memorySize:
			i = np.random.randint(0,self.memorySize)
			self.memory[i] = (oldState, action, reward, newState)
		else:
			self.memory.append((oldState, action, reward, newState))
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Method which returns the action based on specified state and error.
		@param
			observation: the current state of the system, centered
			around the errors. Dimensionality: NxNxE, where E is the
			amount of errors we wish to evaluate actions for.
		@return
			int: the given action based on the state.
			int: the associated error.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def choose_action(self, observation):

		numErrors = observation.shape[2]
		predQ = self.predQ(observation)

		# Check the epsilon-greedy criterion
		if np.random.uniform() > self.epsilon:
			index = np.unravel_index(predQ.argmax(), predQ.shape)
			action = index[0]
			error = index[1]
		else:
			action = np.random.choice(self.actions)
			error = np.random.choice(range(numErrors))

		return action, error
		
	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Returns the predicted Q-value for each error in each direction
		@param:
			observation: the current state of the system, centered
			around the errors. Dimensionality: NxNxE, where E is the
			amount of errors we wish to evaluate actions for.
		
		@return:
			predQ: 2D-vector with Q-values for each error in the
			observation.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def predTargetQ(self,observation):
		numErrors = observation.shape[2]
		predQ = np.zeros([4, numErrors])
		for x in range(numErrors):
			state = observation[:,:,x, np.newaxis]
			predQ[:,x] = self.targetNet.predictQ(state)
		return predQ

	def predQ(self,observation):
		numErrors = observation.shape[2]
		predQ = np.zeros([4, numErrors])
		for x in range(numErrors):
			state = observation[:,:,x, np.newaxis]
			predQ[:,x] = self.qnet.predictQ(state)
		return predQ
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Trains the neural network given the outcome of the action.
		@param
			state: the previous state of the system.
			action: the action taken.
			reward: the immediate reward received.
			observation_p: the resulting observation.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def learn(self):
		# Q is the more optimal Q
		B = list()
		for i in range(self.miniBatchSize):
			j = np.random.randint(0,len(self.memory))
			B.append(self.memory[j])
		
		state = np.zeros((self.miniBatchSize,self.state_size,self.state_size,1))
		Q = np.zeros((self.miniBatchSize,4))
		
		for i in range(self.miniBatchSize):
			
			transition = B[i]
			
			state_ = transition[0]
			action = transition[1]
			reward = transition[2]
			observation_p = transition[3]

			state_ = state_[:,:,np.newaxis]

			state[i,:,:,:] = state_

			Q_ = self.qnet.predictQ(state_)[0,:]
			if observation_p != 'terminal':
				predQ = self.predTargetQ(observation_p)
				Q_[action] = reward + self.gamma * predQ.max()
			else:
				Q_[action] = reward
			
			Q[i,:] = Q_
		self.qnet.improveQ(state, Q)
		self.count += 1
		if self.count % self.TNRate == 0:
			self.targetNet.network.set_weights(self.qnet.network.get_weights())

			
			


	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Changes the epsilon in the epsilon-greedy policy.
		@param
			epsilon: the new epsilon.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       
	def changeEpsilon(self, epsilon):
		self.epsilon = epsilon


if __name__ == '__main__':

	rl = RLsys(4, 3)
	M = np.zeros([3, 3, 2])
	a, c = rl.choose_action(M)
	print(a)
	print(c)
