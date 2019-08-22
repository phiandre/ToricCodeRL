import numpy as np
from RL import RLsys
from Env import Env
from keras.models import load_model
import pandas as pd

class MainClass:

	def __init__(self):
		self.loadNetwork = False #train an existing network
		self.networkName = '' #Name of network to be loaded, if any
		self.gsRGrowth = np.load("Tweaks/GSgrowth.npy")
		self.checkGS = np.load("Tweaks/checkGS.npy")
		
		#Epsilon decay parameters
		self.epsilonDecay = np.load("Tweaks/epsilonDecay.npy")
		if self.epsilonDecay:
			self.alpha = np.load("Tweaks/alpha.npy")
			self.k = np.load("Tweaks/k.npy")
		self.saveRate = 1
		self.avgTol = 1000
		self.fR = np.load("Tweaks/correctGsR.npy")
		
		self.run()



	def rotateHumanRep(self,humanRep,j):
		tmp = np.concatenate([humanRep, humanRep[:,0:1]],axis=1)
		tmp1 = np.concatenate([tmp,tmp[0:1,:]])
		humanRep = np.rot90(tmp1,j)
		state = humanRep[0:(humanRep.shape[0]-1),0:(humanRep.shape[1]-1)]
		return state
		
	
	def labelState(self, s, size):
		state = s
		label = 1
		for j in range(size):
			for k in range(size):
				if state[j,k] == 1:
					state[j,k] = label
					label +=1
		return state
		
	def run(self):
		actions = 4
		comRep = np.load('ToricCodeComputer.npy')
		humRep=np.load('ToricCodeHuman.npy')
		size = comRep.shape[0]

		validation_data_comp = np.load('ToricCodeComputerTest.npy')
		validation_data_hum = np.load('ToricCodeHumanTest.npy')


		rl = RLsys(actions, size)
		if self.loadNetwork:
			importNetwork = load_model(self.networkName)
			rl.qnet.network = importNetwork
			rl.targetNet.network = importNetwork


		averager = np.zeros(comRep.shape[2]*4)

		n=0

		rl.epsilon = np.load("Tweaks/epsilon.npy")

		trainingIteration = 0
		if self.gsRGrowth:
			A = np.load("Tweaks/AGS.npy")
			B = np.load("Tweaks/BGS.npy")
			w = np.load("Tweaks/wGS.npy")
			b = np.load("Tweaks/bGS.npy")

		isCollectingTrainingStats = np.load("Tweaks/isCollectingTrainingStats.npy")
		incorrectGsR = np.load("Tweaks/incorrectGsR.npy")
		stepR = np.load("Tweaks/stepR.npy")

		validation_result = np.zeros((comRep.shape[2],1))


		for i in range(comRep.shape[2]):
			for j in range(4):
				state = np.copy(comRep[:,:,i])
				state = np.rot90(state,j)
				humanRep = humRep[:,:,i]
				humanRep = self.rotateHumanRep(humanRep,j)

				env = Env(state, humanRep, checkGroundState=self.checkGS)
				env.incorrectGsR = incorrectGsR
				env.stepR = stepR
				numSteps = 0

				if self.epsilonDecay:
					rl.epsilon = ((self.k+trainingIteration+12000)/self.k)**(self.alpha)
				if self.gsRGrowth:
					env.correctGsR = A*np.tanh(w*(trainingIteration+b)) + B
				else:
					env.correctGsR = self.fR

				r = 0

				while len(env.getErrors()) > 0:
					numSteps = numSteps + 1
					observation = env.getObservation()
					a, e = rl.choose_action(observation)
					r = env.moveError(a, e)
					new_observation = env.getObservation()
					rl.storeTransition(observation[:,:,e], a, r, new_observation)
					rl.learn()

				print("Number of steps taken at iteration", trainingIteration, ': ', numSteps)

				if(isCollectingTrainingStats):
					proportionCorrectGroundState = 0
					rl.epsilon = 0
					for validation in range(validation_data_comp.shape[2]):
						state = np.copy(validation_data_comp[:,:,validation])
						human_state = np.copy(validation_data_hum[:,:,validation])

						env = Env(state, human_state, checkGroundState=True)

						env.incorrectGsR = incorrectGsR
						env.stepR = stepR

						numSteps = 0
						while len(env.getErrors()) > 0:
							numSteps = numSteps + 1
							observation = env.getObservation()
							a, e = rl.choose_action(observation)
							r = env.moveError(a, e)
							if numSteps > 50:
								r = -1
								break
						if r == env.correctGsR:
							proportionCorrectGroundState = proportionCorrectGroundState + 1

					proportionCorrectGroundState = proportionCorrectGroundState/validation_data_comp.shape[2]

					validation_result[trainingIteration] = proportionCorrectGroundState

					if self.checkGS:
						if r != 0:
							if r == env.correctGsR:
								averager[n] = 1
							n += 1

						if n < self.avgTol:
							average = np.sum(averager)/n
						else:
							average = np.sum(averager[(n-self.avgTol):n])/self.avgTol



					print("Proportion correct GS at iteration " +str(trainingIteration) + ": ", proportionCorrectGroundState)
					if self.checkGS:
						if n<self.avgTol:
							print("Probability of correct GS last " + str(n) + ": " + str(average*100) + " %")
						else:
							print("Probability of correct GS last " + str(self.avgTol) + ": " + str(average*100) + " %")




				if((trainingIteration+1) % self.saveRate == 0):
					filenameNetworkParameters = 'TrainedNetwork.h5'
					filename_validation_result = 'ValidationResult.csv'
					rl.qnet.network.save(filenameNetworkParameters)
					try:
						np.save('MemoryBuffer.npy', rl.memory)
						df = pd.DataFrame(validation_result)
						df.to_csv(filename_validation_result)
					except Exception as e:
						print('Failed to save memory buffer')
						print(e)


				trainingIteration = trainingIteration + 1


		


if __name__ == '__main__':
	MainClass()


