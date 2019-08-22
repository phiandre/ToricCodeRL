import numpy as np
import math
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

	The Tweaker class handles various tweaks used when testing and
	developing. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Tweaker:
	
	def __init__(self):


		# Number of observations in training data to be generated.
		self.trainingIterations = 3000
		
		# Number of observations in test data to be generated.
		self.testIterations = 3000


		""" Reward parameters"""
		
		# Take ground state into account.
		self.checkGS = False
		
		# Growing reward scheme for ground state.
		self.GSgrowth = False
		
		# Reward for correct ground state (if growing reward scheme, this is the final reward).
		self.correctGsR = 5


		
		# Reward for incorrect ground state.
		self.incorrectGsR = -1
		
		# Reward for taking a step (action).
		self.stepR = -1		
		
		# Growing reward scheme for correct ground state according to the formula.
		# R_GS = A * tanh(w(x+b))+ B 

		
		# Reward scheme curve independent of number of observations.
		self.groundStateShape = False

		# Growing reward scheme function parameters.
		self.AGS = 0.5 * self.correctGsR
		self.BGS = self.AGS
		if self.groundStateShape:
			self.wGS = math.pi / (0.275 * self.trainingIterations)
			self.bGS = 0.39 * self.trainingIterations
		else:
			self.wGS = math.pi / 55000
			self.bGS = 78000
		
		""" Epsilon decay """
		# Epsilon if decay is not used.
		self.epsilon = 0.1

		# Use decaying epsilon.
		self.epsilonDecay = False

		# Epsilon dependent of time.
		self.time = 3600 * 22 / 10


		
		# Epsilon decay curve independent of number of iterations.
		self.epsilonShape = True
		
		self.alpha = -0.9
		
		if self.epsilonShape:
			self.k = self.time / 10
		else:
			self.k = 7000
		
		""" Error rate while training"""
		
		# Use growing error rate.
		self.errorGrowth = False
		
		# Error growth curve shape indepdenten of number of iterations.
		self.errorShape = False
		
		# Error rate if growing error rate is not used. (if growing: this is the final error rate)
		self.Pe = 0.1
		
		# Initial error rate (only relevant if growing error rate is used)
		self.Pei = 0.04
		
		# Error rate for test data.
		self.PeTest = 0.1
		# Error rate is growing according to the formula
		# P_e = A * tanh(w(x+b))+ B 


		# Set parameters in error growth rate function.
		self.AE = 0.5 * self.Pe - 0.5 * self.Pei
		self.BE = 0.5 * self.Pe + 0.5 * self.Pei
		
		if self.errorShape:
			self.wE = math.pi / (0.125*self.trainingIterations)
			self.bE = -0.13 * self.trainingIterations
		else:
			self.wE = math.pi/25000
			self.bE = -26000


		# If validation should be conducted while training.
		self.isCollectingTrainingStats = False
		
	

if __name__ == '__main__':
	tweak = Tweaker()
	np.save("Tweaks/trainingIterations.npy",tweak.trainingIterations)
	np.save("Tweaks/testIterations.npy",tweak.testIterations)
	np.save("Tweaks/checkGS.npy",tweak.checkGS)
	np.save("Tweaks/GSgrowth.npy",tweak.GSgrowth)
	np.save("Tweaks/correctGsR.npy",tweak.correctGsR)
	np.save("Tweaks/incorrectGsR.npy",tweak.incorrectGsR)
	np.save("Tweaks/stepR.npy",tweak.stepR)
	np.save("Tweaks/groundStateShape.npy",tweak.groundStateShape)
	np.save("Tweaks/AGS.npy",tweak.AGS)
	np.save("Tweaks/BGS.npy",tweak.BGS)
	np.save("Tweaks/wGS.npy",tweak.wGS)
	np.save("Tweaks/bbGS.npy",tweak.bGS)
	np.save("Tweaks/epsilonDecay.npy",tweak.epsilonDecay)
	np.save("Tweaks/epsilon.npy",tweak.epsilon)
	np.save("Tweaks/epsilonShape.npy",tweak.epsilonShape)
	np.save("Tweaks/alpha.npy",tweak.alpha)
	np.save("Tweaks/k.npy",tweak.k)
	np.save("Tweaks/errorGrowth.npy",tweak.errorGrowth)
	np.save("Tweaks/errorShape.npy",tweak.errorShape)
	np.save("Tweaks/Pe.npy",tweak.Pe)
	np.save("Tweaks/Pei.npy",tweak.Pei)
	np.save("Tweaks/PeTest.npy",tweak.PeTest)
	np.save("Tweaks/AE.npy",tweak.AE)
	np.save("Tweaks/BEcap.npy",tweak.BE)
	np.save("Tweaks/wE.npy",tweak.wE)
	np.save("Tweaks/bE.npy",tweak.bE)
	np.save("Tweaks/isCollectingTrainingStats.npy", tweak.isCollectingTrainingStats)
