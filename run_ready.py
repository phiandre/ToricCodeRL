
import numpy as np
from RL import RLsys
from Env import Env
from BlossomEnv import Env as BEnv
from Blossom import Blossom
from keras.models import load_model
from GenerateData import Generate
from Drawer import Drawer


class MainClass:

	def __init__(self):
		self.graphix = False
		self.networkName = 'Network3x395gamma.h5'
		self.run()


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
		Generator = Generate()
		importNetwork = load_model(self.networkName)
		rl = RLsys(4, importNetwork.input_shape[2])
		rl.qnet.network = importNetwork
		rl.changeEpsilon(0)
		testProbs = [0.1]

		draw = Drawer()
		for prob in testProbs:

			numberOfLogicalErrorsRL = 0
			numberOfLogicalErrorsMWPM = 0
			Generator.DataGeneration(prob)

			humRep=np.load('ToricCodeHumanTest.npy')
			comRep=np.load('ToricCodeComputerTest.npy')


			size = comRep.shape[0]

			for i in range(comRep.shape[2]):


				state=comRep[:,:,i]
				human=humRep[:,:,i]

				env = Env(state, human, checkGroundState=True)

				blossomReward = 0

				if np.count_nonzero(state) > 0:
					state_ = np.copy(state)
					state_ = self.labelState(state_,size)
					BlossomObject = Blossom(state_)
					MWPM = BlossomObject.readResult()
					Benv = BEnv(state_, human, checkGroundState = True)
					for element in MWPM:
						error1 = element[0]+1
						error2 = element[1]+1
						blossomReward, _ = Benv.blossomCancel(error1, error2)


					if blossomReward != Benv.correctGsR:
						numberOfLogicalErrorsMWPM += 1


				r = env.correctGsR
				numIter = 0

				if self.graphix:
					print(f"state before correction {i}: ")
					draw.Draw(state, human)
				while len(env.getErrors()) > 0:

					numIter = numIter + 1
					observation = env.getObservation()
					a, e = rl.choose_action(observation)
					r = env.moveError(a, e)
					if numIter > 50:
						r=-1
						break

				if self.graphix:
					print(f"state after correction {i}: ")
					draw.Draw(env.state, env.humanState)

				if r != env.correctGsR:
					numberOfLogicalErrorsRL += 1



				if ((i+1)%100 == 0):
					print("Iteration: ", str(i))
					print("Logical error probability RL agent: ", numberOfLogicalErrorsRL/(i+1))
					print("Logical error probability MWPM: ", numberOfLogicalErrorsMWPM / (i + 1))







if __name__ == '__main__':
	MainClass()



