import numpy as np
class Drawer:
	def __init__(self):
		
		self.filledCircle = chr(9679)
		self.verticalBar = chr(124)
		self.emptyCircle = chr(9675)
		self.filledBlock = chr(9608)
		self.dottedBlock = chr(9617)
		self.halfBlockV = chr(9612)
		self.halfBlockH = chr(9604)
		self.ischBlock = chr(9619)
		self.horizontalThick = chr(9644)
		
		"""
		self.positiveSpinHor = self.emptyCircle + "-"
		self.positiveSpinVer = self.emptyCircle + "|"
		self.negativeSpinHor = self.filledCircle + "-"
		self.negativeSpinVer = self.filledCircle + "|"
		self.vertex = "+ "
		self.vertexSyndrome = "+-"
		self.plaquette = "  "
		self.syndromeEmpty = self.dottedBlock*2
		self.syndromeExcitation = self.filledBlock*2
		self.vertical = self.verticalBar
		self.horizontal = "-"
		"""
		self.positiveSpinHor = "  "
		self.positiveSpinVer = " "
		self.negativeSpinHor = self.horizontalThick*2
		self.negativeSpinVer = self.halfBlockV
		self.vertex = " "
		self.vertexSyndrome = "  "
		self.plaquette = "  "
		self.syndromeEmpty = self.dottedBlock*2
		self.syndromeExcitation = self.ischBlock*2
		self.vertical = self.verticalBar
		self.horizontal = "-"
		
		
		
		"""
		self.positiveSpinHor = self.dottedBlock + "-"
		self.positiveSpinVer = self.dottedBlock + "|"
		self.negativeSpinHor = self.filledBlock + "-"
		self.negativeSpinVer = self.filledBlock + "|"
		self.vertex = "+ "
		self.plaquette = "  "
		self.syndromeEmpty = "[]"
		self.syndromeExcitation = "XX"
		self.vertical = self.verticalBar
		self.horizontal = "-"
		"""
		
	
	def DrawSyndrome(self,syndrome):
		size = syndrome.shape[0]
		vertexRowTmp = [self.vertexSyndrome for i in range(size)]
		vertexRow = " ".join(element for element in vertexRowTmp)+" \n"
		symbolSyndrome = list()
		for i in range(size):
			iRowList = []
			for j in range(size):
				ijPlaquette = syndrome[i,j]
				if ijPlaquette == 0:
					iRowList.append(self.syndromeEmpty)
				else:
					iRowList.append(self.syndromeExcitation)
			iRow = " " + "   ".join(element for element in iRowList) +"\n"
			symbolSyndrome.append(iRow)
		output = vertexRow + vertexRow.join(row for row in symbolSyndrome)
		print(output)
		
		
	
	def DrawSpins(self,spinLattice):
		size = int(spinLattice.shape[0]/2)
		fakeSyndrome = np.zeros((size,size))
		self.Draw(fakeSyndrome,spinLattice)
		
	def Draw(self,syndrome,spinLattice):
		size = syndrome.shape[0]
		
		symbolSyndrome = list()
		for i in range(size):
			iRow = []
			for j in range(size):
				ijPlaquette = syndrome[i,j]
				if ijPlaquette == 0:
					iRow.append(self.syndromeEmpty)
				else:
					iRow.append(self.syndromeExcitation)
			symbolSyndrome.append(iRow)
		
		symbolLatticeHor = list()
		
		for i in range(size):
			iRow = []
			for j in range(size):
				index1 = 2*i
				index2 = 2*j +1
				ijSpin = spinLattice[index1,index2]
				if ijSpin == 1:
					iRow.append(self.positiveSpinHor)
				else:
					iRow.append(self.negativeSpinHor)
			symbolLatticeHor.append(iRow)
		
		symbolLatticeVert = list()
		
		for i in range(size):
			iRow = []
			for j in range(size):
				index1 = 2*i+1
				index2 = 2*j
				ijSpin = spinLattice[index1,index2]
				if ijSpin ==1:
					iRow.append(self.positiveSpinVer)
				else:
					iRow.append(self.negativeSpinVer)
			symbolLatticeVert.append(iRow)
		
		outputList = list()
		for i in range(size):
			vertexRow = []
			plaquetteRow = []
			for j in range(size):
				vertexRow.append([self.vertex,symbolLatticeHor[i][j]])
				plaquetteRow.append([symbolLatticeVert[i][j],symbolSyndrome[i][j]])
			outputList.append(" ".join(" ".join(string for string in element) for element in vertexRow))
			outputList.append(" ".join(" ".join(string for string in element) for element in plaquetteRow))
		output = "\n".join(row for row in outputList)
		print(output)
		return

if __name__ == '__main__':
	drawer = Drawer()
	
	comRep = np.load('ToricCodeComputer.npy')
	humRep=np.load('ToricCodeHuman.npy')
	
	syndrome = comRep[:,:,0]
	spinLattice = humRep[:,:,0]
	print("Combined:")
	drawer.Draw(syndrome,spinLattice)
	print("Syndrome:")
	drawer.DrawSyndrome(syndrome)
	print("Spin Lattice:")
	drawer.DrawSpins(spinLattice)
