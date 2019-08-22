"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

	The Env class creates an object which represents the toric code as
	a matrix.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

##########
# Import #
##########
import math
import numpy as np


###############
# Klassen Env #
###############
class Env:

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Constructor
		@param
			compState: the initial state matrix (syndrome), represented in numpy.
			humanState: the initial spin matrix, represented in numpy.
			groundState: the four groundstates are coded as integers in range 0 to 3.
			checkGroundState: if ground state should be taken into account while training.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def __init__(self, compState, humanState=np.zeros(0), groundState=0, checkGroundState=False):

		self.checkGroundState = checkGroundState
		self.state = np.copy(compState) 
		self.humanState = np.copy(humanState) 
		self.length = self.state.shape[0]
		self.groundState = groundState

		self.updateErrors()
		
		self.stepR = -1
		self.correctGsR = 5
		self.incorrectGsR = -1
		self.elimminationR = -1

	"""""""""""""""""""""""""""""""""""""""""""""""""""
	Find all errors and update the matrix errors which
	contains the coordinates of all errors.
	"""""""""""""""""""""""""""""""""""""""""""""""""""
	def updateErrors(self):
		self.errors = np.transpose(np.nonzero(self.state))

	"""""""""""""""""""""""""""""""""""""""
		@return
			numpy: matrix containing all errors.
	"""""""""""""""""""""""""""""""""""""""
	def getErrors(self):					
		return self.errors

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Move errors and annihilate two colliding errors.
	Actions are coded as: [u = 0, d = 1, l = 2, r = 3]
		@param
			action: the desired action to conduct.
			errorIndex: index in matrix of the error that should be moved
		@return
			int: reward obtained from the action
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def moveError(self, action, errorIndex):
		amountErrors = len(self.errors)
		firstPos = self.errors[errorIndex, :]
		secondPos = self.getPos(action, firstPos)
		

		if self.checkGroundState:
			firstHumPos=2*firstPos+1
			secondHumPos=2*secondPos+1
			if action==0 and firstPos[0]==0:
				vertexPos = [0, firstHumPos[1]]
			elif action==1 and firstPos[0]==self.length - 1:
				vertexPos = [0, firstHumPos[1]]
			elif action==2 and firstPos[1]==0:
				vertexPos = [firstHumPos[0], 0]
			elif action==3 and firstPos[1]==self.length - 1:
				vertexPos = [firstHumPos[0], 0]
			else:
				vertexPos = 1/2 * (firstHumPos + secondHumPos)
				vertexPos = vertexPos.astype(int)
			self.humanState[vertexPos[0], vertexPos[1]] *= -1
		

		self.state[firstPos[0], firstPos[1]] = 0

		if self.state[secondPos[0], secondPos[1]] == 0:
			self.state[secondPos[0], secondPos[1]] = 1
		else:
			self.state[secondPos[0],secondPos[1]] = 0

		self.updateErrors()

		if self.checkGroundState:
			if len(self.errors) == 0:
				if (self.evaluateGroundState() == self.groundState):
					return self.correctGsR
				else:
					return self.incorrectGsR
		if amountErrors > len(self.errors):
			return self.elimminationR
		return self.stepR

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Return a centralized view of the specified error.
		@param
			error: index to the error that should be centralized.
		@return
			numpy: the centralized state
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""		
	def centralize(self, error):
		# state är matrisen som karaktäriserar tillståndet
		# error är koordinaterna för felet
		state_=np.concatenate((self.state[:,error[1]:],self.state[:,0:error[1]]),1)
		state_=np.concatenate((state_[error[0]:,:],state_[0:error[0],:]),0)
		rowmid=int(np.ceil(self.state.shape[0]/2))
		colmid=int(np.ceil(self.state.shape[1]/2))
		state_=np.concatenate((state_[:,colmid:],state_[:,0:colmid]),1)
		state_=np.concatenate((state_[rowmid:,:],state_[0:rowmid,:]),0)
		
		return state_
	
	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Return the position after movement in a specified direction.
		@param
			action: the desired action to conduct.
			position: position before movement.
		@return
			numpy: coordinates of the new position
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def getPos(self, action, position):
		nextPos = np.array(position, copy=True)
		if action == 0:
			if nextPos[0] == 0:
				nextPos[0] = self.length - 1
			else:
				nextPos[0] -= 1  
		if action == 1:
			if nextPos[0] == self.length - 1:
				nextPos[0] = 0
			else:
				nextPos[0] += 1
		if action == 2:
			if nextPos[1] == 0:
				nextPos[1] = self.length - 1
			else:
				nextPos[1] -= 1
		if action == 3:
			if nextPos[1] == self.length - 1:
				nextPos[1] = 0
			else:
				nextPos[1] += 1
		return nextPos

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Create a 3D matrix which contains a centralized view for each
	present error.
		@return
			numpy: 3D matrix of centralized views
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def getObservation(self):
		if len(self.errors)==0:
			return 'terminal'
		else:
			numerror=self.errors.shape[0]
			observation=np.zeros((self.length,self.length,numerror))
			for i in range(numerror):
				observation[:,:,i]=self.centralize(self.errors[i,:])
			return observation

	""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Get the ground state of the toric code.
	Ground state encoding: 0 (no non-trivial loops)
						   1 (vertical non-trivial loop)
					       2 (horizontal non-trivial loop)
					       3 (vertical + horizontal non-trivial loop)
		@return
			int: ground state, coded as described above
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def evaluateGroundState(self):
		yProd = 1
		xProd = 1
		groundState = 0
		for i in range(self.length):
			yProd = yProd * self.humanState[2*i+1, 0]
			xProd = xProd * self.humanState[0, 2*i+1]

		if (yProd == -1):
			groundState += 1
		if (xProd == -1):
			groundState += 2
		return groundState

	"""""""""""""""""""""""""""""""""""""""""""""
	Returns a copy of the Env object
		@return
			Env: a copy of Env object.
	"""""""""""""""""""""""""""""""""""""""""""""
	def copy(self):
		copyState = np.copy(self.state)
		copyHuman = np.copy(self.humanState)
		copyEnv = Env(copyState, copyHuman, self.groundState, self.checkGroundState)
		return copyEnv
		
	

