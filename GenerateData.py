import numpy as np
import random

class Generate:
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		
		@param
			size: the side-length to be used. The lattice is a square. 
			errorProbability: the error rate to be used while generating the data.
		@return
			humanRepresentation: a representation which contains information about each individual spin.
								This is used to assess performance of the algorithm and not for training.
			computerRepresentation: a size x size matrix which represents the syndrome. 
			numberOfErrors: the number of errors created in this observation.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def generateData(self, size, errorProbability):
		
		comRep = np.ones((size,size))
		
		humanRepresentation, computerRepresentation, numberOfErrors = self.initialize(size, errorProbability, comRep)
		computerRepresentation = abs((computerRepresentation-1)/(2))
		
		
		return humanRepresentation, computerRepresentation, numberOfErrors
		
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		@param
			size: lattice side-length
			errorProbability: the error rate to be used while generating the data.
			computerRep: an empty lattice.
		@return
			humanRepresentation: a representation which contains information about each individual spin.
								This is used to assess performance of the algorithm and not for training.
			computerRepresentation: a size x size matrix which represents the syndrome. 
			numberOfErrorsCreated: the number of errors created in this observation.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	def initialize(self, size, errorProbability, computerRep):
		humanRepresentation = np.zeros((2*size,2*size))
		comRep = computerRep

		# The following lines are useful when analyzing low error probabilities, as it will only explicitly generate states
		# with two or more errors. Other states are trivial to solve.
		# If using these lines, be mindful of how to interpret the data!
		"""
		row1, col1 = self.flipRandomIndex(size)
		row2, col2 = self.flipRandomIndex(size)
		#row3, col3 = self.flipRandomIndex(size)
		#row4, col4 = self.flipRandomIndex(size)


		while ((row1 == row2) and (col1 == col2)):
			row2, col2 = self.flipRandomIndex(size)

		
		humanRepresentation[row1, col1] = -1
		humanRepresentation[row2, col2] = -1
		
		comRep = self.updateComputerRepresentation(row1, col1, size, comRep)
		comRep = self.updateComputerRepresentation(row2, col2, size, comRep)
		"""

		numberOfErrorsCreated = 0
		for i in range(0,2*size):
			if i%2==0:
				for j in range(0,2*size):
					if j%2==1:
						if humanRepresentation[i,j] != -1:
							if np.random.uniform() < errorProbability:
								humanRepresentation[i,j] = -1
								numberOfErrorsCreated = numberOfErrorsCreated + 1
								comRep = self.updateComputerRepresentation(i,j,size,comRep)
							else:
								humanRepresentation[i,j] = 1
					
			else:
				for j in range(0,2*size):
					if j%2==0:
						if humanRepresentation[i, j] != -1:
							if np.random.uniform() < errorProbability:
								humanRepresentation[i,j] = -1
								numberOfErrorsCreated = numberOfErrorsCreated + 1
								comRep = self.updateComputerRepresentation(i,j,size,comRep)
							else:
								humanRepresentation[i,j] = 1

		return humanRepresentation, comRep, numberOfErrorsCreated
		

	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		@param
			humanRowIndex: the row index  of the spin in humanRepresentation that was flipped.
			humanColumnIndex: the column index of the spin in humanRepresentation that was flipped.
			size: lattice side-length
			rep: the current version of computerRepresentation
		
		@return
			computerRepresentation: the updated version of computerRepresentation 
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

	def updateComputerRepresentation(self, humanRowIndex, humanColumnIndex, size, rep):
		computerRepresentation = rep
		rowIndex = int(np.floor(humanRowIndex / 2))
		columnIndex = int(np.floor(humanColumnIndex / 2))
		if humanRowIndex == 0:
			computerRepresentation[(rowIndex, columnIndex)] = (-1)*computerRepresentation[(rowIndex, columnIndex)]
			computerRepresentation[(size-1, columnIndex)] = (-1)*computerRepresentation[(size-1, columnIndex)]
		elif humanColumnIndex == 0:
			computerRepresentation[(rowIndex, columnIndex)] = (-1)*computerRepresentation[(rowIndex, columnIndex)]
			computerRepresentation[(rowIndex, size-1)] = (-1)*computerRepresentation[(rowIndex, size-1)]
		elif humanRowIndex % 2 == 1:
			computerRepresentation[(rowIndex,columnIndex)] = (-1)*computerRepresentation[(rowIndex,columnIndex)]
			computerRepresentation[(rowIndex,columnIndex-1)] = (-1)*computerRepresentation[(rowIndex,columnIndex-1)]
		else:
			computerRepresentation[(rowIndex,columnIndex)] = (-1)*computerRepresentation[(rowIndex,columnIndex)]
			computerRepresentation[(rowIndex-1,columnIndex)] = (-1)*computerRepresentation[(rowIndex-1,columnIndex)]

		return computerRepresentation
		
	
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		@param
			size: lattice side-length
		
		@return
			rowIndex: row index of spin in humanRepresentation to be flipped.
			columnIndex: column index of spin in humanRepresentation to be flipped.
	"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	
	def flipRandomIndex(self,size):
		r = np.random.rand()
		if (r < 0.5): # <-----------> in SYNDROME!!!!!!
			rowIndex_tmp = random.randint(0, size - 1)
			rowIndex = (rowIndex_tmp+1)*2 - 1

			colIndex_tmp = random.randint(0, size-1)
			colIndex = 2*colIndex_tmp
		else:
			rowIndex_tmp = random.randint(0, size - 1)
			rowIndex = (rowIndex_tmp)*2

			colIndex_tmp = random.randint(0, size - 1)
			colIndex = 2 * (colIndex_tmp+1) - 1
		return rowIndex , colIndex

	
	"""
	def adjacentIndex(self,size,rowIndex, columnIndex):
		adjRow = random.randint(-1,1)
		adjCol = random.randint(-1,1)
		
		while adjRow == 0:
			adjRow = random.randint(-1,1)
		while adjCol == 0:
			adjCol = random.randint(-1,1)
		
		rowIndex = rowIndex + adjRow
		columnIndex = columnIndex + adjCol
		
		if rowIndex == 2*size:
			rowIndex = 0
		if rowIndex == -1:
			rowIndex = 2*size-1
		if columnIndex == 2*size:
			columnIndex = 0
		if columnIndex == -1:
			columnIndex = 2*size-1
		
		
		return rowIndex, columnIndex
	"""

	"""
		Save all the generated data to files.
	"""
	def saveToFile(self, human, computer, humanTest, computerTest, Pe, testProb, numberOfTrainingErrors, numberOfTestErrors):
		np.save('ToricCodeHuman',human)
		np.save('ToricCodeComputer', computer)
		np.save('ToricCodeHumanTest',humanTest)
		np.save('ToricCodeComputerTest', computerTest)

		with open("trainingNumberOfErrors.txt", "a") as f:
			f.write(str(Pe))
			f.write("\n")
			for i in range(len(numberOfTrainingErrors)):
				if numberOfTrainingErrors[i] > 0:
					f.write(str(i) + ": " + str(numberOfTrainingErrors[i]) + "\n")
			f.write("\n \n")

		with open("testNumberOfErrors_5.txt", "a") as f:
			f.write(str(testProb))
			f.write("\n")
			for i in range(len(numberOfTestErrors)):
				if numberOfTestErrors[i] > 0:
					f.write(str(i) + ": " + str(numberOfTestErrors[i]) + "\n")
			f.write("\n \n")


	"""
		@param:
			testProb: the error rate of test data.
			
	"""
	def DataGeneration(self,testProb):
		size = 3  #Size of lattice, MUST BE CHANGED WHEN GENERATING NEW DATA.
		numGenerations = np.load("Tweaks/trainingIterations.npy")
		testGenerations = np.load("Tweaks/testIterations.npy")
		Pe = np.load("Tweaks/Pe.npy")
		errorProb = Pe
		Pei = np.load("Tweaks/Pei.npy")
		AE = np.load("Tweaks/AE.npy")
		BE = np.load("Tweaks/BEcap.npy")
		wE = np.load("Tweaks/wE.npy")
		bE = np.load("Tweaks/bE.npy")
		errorGrowth = np.load("Tweaks/errorGrowth.npy")

		generator = Generate()

		tmpHuman = np.zeros((size * 2, size * 2, numGenerations))
		tmpComputer = np.zeros((size, size, numGenerations))

		trainingNumberOfErrors = np.zeros(2*size**2)
		testNumberOfErrors = np.zeros(2*size**2)

		for i in range(numGenerations):
			if errorGrowth:
				errorProb = AE * np.tanh(wE * (i + 1 + bE)) + BE
			human, computer, numberOfErrors = generator.generateData(size, errorProb)
			tmpHuman[:, :, i] = human
			tmpComputer[:, :, i] = computer
			trainingNumberOfErrors[numberOfErrors] += 1

		errorProb = testProb

		tmpHumanTest = np.zeros((size * 2, size * 2, testGenerations))
		tmpComputerTest = np.zeros((size, size, testGenerations))
		for i in range(testGenerations):
			humanTest, computerTest, numberOfErrors = generator.generateData(size, errorProb)
			tmpHumanTest[:, :, i] = humanTest
			tmpComputerTest[:, :, i] = computerTest
			testNumberOfErrors[numberOfErrors] += 1

		generator.saveToFile(tmpHuman, tmpComputer, tmpHumanTest, tmpComputerTest, Pe, testProb, trainingNumberOfErrors, testNumberOfErrors)

			
if __name__ == '__main__':
	size = 5 #Size of lattice, MUST BE CHANGED WHEN GENERATING NEW DATA.
	numGenerations = np.load("Tweaks/trainingIterations.npy")
	testGenerations = np.load("Tweaks/testIterations.npy")
	testProb = np.load("Tweaks/PeTest.npy")
	Pe = np.load("Tweaks/Pe.npy")
	errorProb = Pe
	Pei = np.load("Tweaks/Pei.npy")
	AE = np.load("Tweaks/AE.npy")
	BE = np.load("Tweaks/BEcap.npy")
	wE = np.load("Tweaks/wE.npy")
	bE = np.load("Tweaks/bE.npy")
	errorGrowth = np.load("Tweaks/errorGrowth.npy")

	generator = Generate()

	tmpHuman = np.zeros((size*2,size*2,numGenerations))
	tmpComputer = np.zeros((size,size,numGenerations))

	trainingNumberOfErrors = np.zeros(2*size**2)
	testNumberOfErrors = np.zeros(2*size**2)

	for i in range(numGenerations):
		if errorGrowth:
			errorProb = AE * np.tanh(wE*(i+1+bE))+BE
		human, computer, numberOfErrors = generator.generateData(size,errorProb)
		tmpHuman[:,:,i] = human
		tmpComputer[:,:,i] = computer
		trainingNumberOfErrors[numberOfErrors] += 1

	errorProb = testProb

	tmpHumanTest = np.zeros((size*2,size*2,testGenerations))
	tmpComputerTest = np.zeros((size,size,testGenerations))


	for i in range(testGenerations):
		humanTest, computerTest, numberOfErrors = generator.generateData(size,errorProb)
		tmpHumanTest[:,:,i] = humanTest
		tmpComputerTest[:,:,i] = computerTest
		testNumberOfErrors[numberOfErrors] += 1

	generator.saveToFile(tmpHuman, tmpComputer, tmpHumanTest, tmpComputerTest, Pe, testProb, trainingNumberOfErrors, testNumberOfErrors)
