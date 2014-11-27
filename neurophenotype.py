# Module for Genetic Algorithms and Neuro-evolutionary Programming

from bitarray import bitarray
import random
import numpy as np


#########################################
## SOME GENERAL PURPOSE MATH FUNCTIONS ##
#########################################.

def bitshift(theBitarray, dir=0):
	assert type(theBitarray) == bitarray

	if dir == 0: # left
		newBitarray = bitarray(theBitarray[1:]+theBitarray[:1])
	elif dir == 1: # right
		newBitarray = bitarray(theBitarray[-1:]+theBitarray[:-1])
	return newBitarray

# note: the gray<=>bin<=>int functions are lifted from Rosetta Code

def gray2bin(bits):
	# change to have it take and output bitarrays not lists
	b = [bits[0]]
	for nextb in bits[1:]: b.append(b[-1] ^ nextb)
	return b

def bin2int(bits):
	"""From binary bits, msb at index 0 to integer"""
	i = 0
	for bit in bits:
		i = i * 2 + bit
	return i

def int2bin(n):
	"""From positive integer to list of binary bits, msb at index 0"""
	if n:
		bits = []
		while n:
			n,remainder = divmod(n, 2)
			bits.insert(0, remainder)
		return bits
	else: return [0]

def bin2gray(bits):
	return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

def getGrayCodeInteger(theBitarray):
	# a much easier way:
	grayInt = bin2int(gray2bin(theBitarray))
	return grayInt

def getGrayCodeBinary(integer):
	grayBin = bitarray(bin2gray(int2bin(integer)))
	return grayBin

def pairMult(pair1,pair2):
	"""Pairwise multiplies two pairs (tuples or lists) of integers or floats, and returns a listed pair."""
	return [pair1[0]*pair2[0],pair1[1]*pair2[1]]

def countOnes(data):
	""" Takes a bitarray and returns a float that is the ratio of 1s to 0s in that bitarray."""

	c = data.count() # a method built into the bitarray class, which counts the number of ones in the bitarray
	
	maxiumumC = len(data) # since the length of the bitarray clearly sets an upper bound on c. 
	ratio = float(c)/float(maxiumumC)
	return ratio

#######################

class NeuroPhenotype(object):

	def __init__(self,genome_bits):
		
		self.genome = genome_bits
		
		 # note: add bias node?
		self.numSynapses = self.layers[0]*self.layers[1] + self.layers[1]*self.layers[2]

		self.weightFields = self.perceptronWeights()[0]
		self.dropoutArray = self.perceptronWeights()[1]

		self.perceptron = self.buildPerceptron()
		self.memory = []

		# bitsNeeded = (self.numSynapses + self.dropout)*8
		# assert len(self.genome) <= bitsNeeded

	def perceptronWeights(self):
		"""Determines the weights for a two-layer perceptron with five input nodes and two output nodes, to provide the worm with a sensorimotor system, on the basis of the genome."""
		
		self.bitFields = []
		for i in range(self.numSynapses+self.dropout):
			step = i*8
			self.bitFields.append(self.genome[step:step+8])

		if self.dropout:
			endOfWeights = -self.dropout
		else:
			endOfWeights = len(self.bitFields)
		
		weightFields = []
		for b in self.bitFields[:endOfWeights]:
			gray = getGrayCodeInteger(b[1:])
			if b[0] == 1:   sign = -1
			elif b[0] == 0: sign = +1
			if gray != 0: 
				weight = sign * (1.0/gray)
			else:
				weight = 0.0
			weightFields.append(weight)

		if self.dropout:
			dropoutArray = self.genome[-(self.numSynapses*8):len(self.genome)] 
		else:
			dropoutArray = bitarray("1")*self.numSynapses
		C1 = self.layers[0]*self.layers[1]
		
		# let's mitigate the worst effects of dropout by setting C1 to full connectivity
		dropoutArray[:C1] = bitarray("1")*C1

		self.connectivity = (countOnes(dropoutArray[:C1]), countOnes(dropoutArray[C1:]))

		return [weightFields, dropoutArray]


	def buildPerceptron(self):
		self.weights = {}
		w = 0

		#print self.weightFields,"len = ",len(self.weightFields)
		
		for h in range(len(self.layers)-1):
			for i in range(self.layers[h]):
				for j in range(self.layers[h+1]):
					self.weights[(h,i,j)] = self.weightFields[w]
					w += 1	

	def feedForward(self, inputs):
		dropoutCounter = 0
		outputs = [0 for i in range(self.layers[-1])]
		self.registers = [inputs, [0 for i in range(self.layers[1])], outputs]
		for h in range(len(self.layers)-1):
			for i in range(self.layers[h]):
				for j in range(self.layers[h+1]):
					## do the dropout shuffle ##
					self.dropoutArray = bitshift(self.dropoutArray, dropoutCounter%2)
					# alt method: use dropoutCounter%2 to alternate rather than randomize the direct of the bitshift. But note that the random method might shift the dropout filter along more than once in a single direction, leading to new configurations via 1d random walk.
					############################
					self.registers[h+1][j] += (self.weights[(h,i,j)] * np.tanh(self.registers[h][i])) * self.dropoutArray[dropoutCounter]
					dropoutCounter += 1	
		return self.registers[-1]

	def mutate(self):
		# with probability prob, this method should flip one bit, chosen at random, in the bitstring.
		fractionToMutate = len(self.bitFields)
		if random.random() <= self.mutation_rate:
			for i in range(len(self.genome)/random.randrange(1,len(self.genome)/fractionToMutate)): 
				j = random.randrange(len(self.genome))
				self.genome[j] = not(self.genome[j])
			self.mutations += 1
		# swap
		if random.random() <= self.mutation_rate*2:

			field1 = random.randrange((len(self.genome)-8)/8)
			field2 = field1
			while field2 == field1:
				field2 = random.randrange((len(self.genome)-8)/8)

			for i in range(8):
				b = field1*8 + i
				self.genome[field2*8+i] = b
			self.mutations += 1