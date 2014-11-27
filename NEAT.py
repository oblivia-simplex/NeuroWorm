# NEAT
import random
import numpy as np 

class Neuron:

	def __init__(self, ID, IN=False, OUT=False, state=0.0, decay=(lambda x: x*0.0)):

		self.ID = ID
		self.IN  = IN
		self.OUT = OUT

		self.state = state
		self.threshold = np.tanh
		self.decay = decay # not sure what the default should be. room to tinker here.
		self.fired = False

	def __eq__(self, other):
		return self.ID == other.ID

	def __lt__(self, other):
		return self.ID < other.ID

	def __gt__(self, other):
		return self.ID > other.ID
		
class Synapse:

	def __init__(self, source, target, weight=0.0, enabled=True):
		self.source  = source
		self.target  = target
		self.weight  = weight
		self.enabled = enabled

	def __getitem__(self, index):
		if index == 0:
			return self.source
		elif index == 1:
			return self.target
		else:
			raise IndexError


	def strip(self):
		return (self.source.ID, self.target.ID)

	def encode(self):
		return (self.source.ID, self.target.ID, self.weight)

	def isIn(self, net):
		ans = any(self.target == apse.target and self.source == apse.source for apse in net.synapses)
		return ans

	def iso(self, other):
		return self.source == other.source and self.target == other.target

	def __eq__(self, other):
		"""Note that equality for synapses is treated in an idiosyncratic fashion. The point of this is to simplify certain search functions, and in particular, the builtin function 'in'.""" # trying to deprecate
		return self.source == other.source and self.target == other.target

	def __lt__(self, other):
		if self.source < other.source:
			return True
		elif self.source > other.source:
			return False
		else:
			if self.target < other.target:
				return True
			elif self.target > other.target:
				return False
			else:
				return False



	def __gt__(self, other):
		if self.source > other.source:
			return True
		elif self.source < other.source:
			return False
		else:
			if self.target > other.target:
				return True
			elif self.target < other.target:
				return False
			else:
				return False

	def __str__(self):
		string = "<"+str(self.source.ID)+"|"+str(self.target.ID)+"> :"+str(self.weight)
		if self.enabled:
			string += " ENABLED "
		else:
			string += " DISABLED "
		return string

	def identical(self, other):
		return self == other and self.weight == other.weight and self.decay == other.decay and self.enabled == other.enabled

class NEATnet(object):

	def __init__(self, inOut, neurons=[], synapses=[], initial_pop=True):
		# each net needs its own ID number. The best way to assign this will just be to pass the init the value of len(self.population).
		#self.ID = ID
		
		if initial_pop:
			self.in_neurons = [Neuron(IN=True, ID=n) for n in xrange(inOut[0])]
			self.out_neurons = [Neuron(OUT=True, ID=inOut[0]+n) for n in xrange(inOut[1])]
			self.neurons = self.out_neurons + self.in_neurons
			self.synapses = []
			self.synaptic_array = [[None for i in xrange(len(self.neurons))] for j in xrange(len(self.neurons))]
			
			
			self.makeFullyConnectedPerceptron()
			self.updateFeeds()

		else:
			self.neurons = neurons
			self.in_neurons = [neuron for neuron in neurons if neuron.IN==True]
			self.out_neurons = [neuron for neuron in neurons if neuron.OUT==True]

			self.synapses = synapses
			self.synaptic_array = [[None for i in xrange(len(self.neurons))] for j in xrange(len(self.neurons))]
			for syn in self.synapses:
				self.synaptic_array[syn[0].ID][syn[1].ID] = syn
			self.updateFeeds()

	def encode(self):
		encoding = []

		for synapse in self.synapses:
			encoding.append(synapse.encode)
		encoding.sort()
		return encoding

	def decode(self, code):
		pass

	def fetchNeuron(self, id):
		"""Returns the neuron in the net with ID==id, if such a neuron exists, and returns False otherwise."""
		for N in self.neurons:
			if N.ID == id:
				return N
		return False #If we go through the whole loop without finding a neuron with that ID, then we've made an erroneous request. Let's have this return False instead of raising an error, so that the fetch can also be used as a query: "if self.fetchNeuron(id):", etc.
	

	def fetchSynapse(self, neuron0, neuron1):
		if type(neuron0) == int and type(neuron1) == int:
			return self.synapses[neuron0][neuron1]
		else: # if they are Neurons
			return self.synapses[neuron0.ID][neuron1.ID]

	def makeFullyConnectedPerceptron(self):
		
		for n0 in self.in_neurons:
			for n1 in self.out_neurons:
				syn = self.setSynapse(n0,n1)
				self.addWeight(syn, rand=True)


	def setSynapse(self, n0, n1):
		synapse = Synapse(n0, n1)
		self.synaptic_array[n0.ID][n1.ID] = synapse
		self.synapses.append(synapse)
		
		return synapse

	def updateFeeds(self):
		for neuron0 in self.neurons:
			neuron0.feeds = 0
			neuron0.eats = 0
			for neuron1 in self.neurons:
				if self.synaptic_array[neuron0.ID][neuron1.ID] is not None and self.synaptic_array[neuron0.ID][neuron1.ID].enabled:
					neuron0.feeds += 1
				if self.synaptic_array[neuron1.ID][neuron0.ID] is not None and self.synaptic_array[neuron1.ID][neuron0.ID].enabled:
					neuron0.eats += 1	

	def enlargeArray(self, n=1):
		for i in xrange(n):
			self.synaptic_array.append([None for i in xrange(len(self.neurons))])
			for column in self.synaptic_array:
				column.append(None)

	def addWeight(self, synapse, adj=1.0, rand=False):
		"""If rand is set to True, then the function will add a random float in between 0.0 and adj to the existing weight of the synapse."""
		
		if rand:
			polarity = random.choice((-1,1)) 
			adj      = polarity * random.random() * adj

		synapse.weight += adj


	def mutate(self, probability=1.0, probWeightVsTopology=0.75, probSynapseVsNeuron=0.5):
		if random.random() < probability:
			if random.random() < probWeightVsTopology:
				self.nudgeWeights()
			else:
				if random.random() < probSynapseVsNeuron and len(self.neurons) > len(self.in_neurons)+len(self.out_neurons):
					self.addSynapse()
					self.updateFeeds()
				else:
					self.addNeuron()
					self.addSynapse()
					self.updateFeeds()
		# redirect to addSynapse or addNeuron

	def nudgeWeights(self, nudge = 0.001):
		"""Nudges all the weights of the synapses in the net by a random positive or negative magnitude, whose absolute value is a float less than nudge."""
		for syn in self.synapses:
			if syn.enabled == True:
				polarity = random.choice((-1,1)) 
				adj = polarity * random.random() * nudge
				syn.weight += adj
		# the idea being that a weight of 0.0 means that the synapse is "Disabled". 

	def addSynapse(self):
		print "Has",len(self.synapses),"synapses..."
		print"TRYING TO ADD SYNAPSE..."
		tries = 10
		while tries:
			n0 = random.choice(self.neurons)
			
			N = [neuron for neuron in self.neurons if neuron.IN == False and neuron != n0 and self.synaptic_array[n0.ID][neuron.ID] is None]
			print N
			if N != []:
				n1 = random.choice(N)
				newSynapse = self.setSynapse(n0, n1)
				self.addWeight(newSynapse, rand=True)
				break
			tries -= 1
			
		print "Now has",len(self.synapses),"synapses."
		


	def addNeuron(self, neuron=None, rand=True):
		print "Has",len(self.neurons),"neurons..."
		print"ADDING NEURON..."
		print"Current array:",len(self.synaptic_array),"x",len(self.synaptic_array)
		self.enlargeArray(1)
		if neuron == None:
			newNeuron = Neuron(ID=len(self.neurons))
			# make room in the synaptic_array!
			
		else:
			newNeuron = neuron

		oldSynapse = random.choice(self.synapses)
		oldSynapse.enabled = False
		
		newSynapse0 = self.setSynapse(oldSynapse.source, newNeuron)
		self.addWeight(newSynapse0, rand=True)

		newSynapse1 = self.setSynapse(newNeuron, oldSynapse.target)
		self.addWeight(newSynapse1, rand=True)

		self.neurons.append(newNeuron)
		print "Now has",len(self.neurons),"neurons..."
		print"New Array:",len(self.synaptic_array),"x",len(self.synaptic_array)
	


	def speciesDistance(self, other, EX, DIS, W):
		
		motherSynapses = self.synapses[:]
		fatherSynapses = other.synapses[:]

		if len(fatherSynapses) > len(motherSynapses):
			fatherSynapses, motherSynapses = motherSynapses, fatherSynapses
		
		union_synapses = motherSynapses+fatherSynapses

		sources_targets = set([syn.strip() for syn in union_synapses])

		child_links = list(sources_targets)
		mother_links = [syn.strip() for syn in motherSynapses]
		father_links = [syn.strip() for syn in fatherSynapses]

		

		child_links.sort()
		mother_links.sort()
		father_links.sort()


		#comboSynapses = mergeListsNoRep(motherSynapses, fatherSynapses)
		# use procedure implemented in crossover func
		# measure excess and disjoint values to gauge whether this is a suitable pairing.
		disjoint, excess = 0,0
		for link in child_links:
			if link not in mother_links or link not in father_links:
				if child_links.index(link) >= len(father_links):
					excess += 1
				else:
					disjoint += 1

		
		weightDiffSum = 0.0
		matchingSynapses = 0
		for syn in motherSynapses:
			for apse in fatherSynapses:
				if syn.strip() == apse.strip():
					matchingSynapses += 1
					weightDiffSum += abs(syn.weight - apse.weight)


		avgWeightDiffs = weightDiffSum / float(matchingSynapses)

		SD = EX*excess + DIS*disjoint + W*avgWeightDiffs

		return SD

	def crossover(self, other):
		motherSynapses = self.synapses[:]
		fatherSynapses = other.synapses[:]
		
		
		child_synapses = []
		
		union_synapses = motherSynapses+fatherSynapses

		sources_targets = set([(syn.source.ID, syn.target.ID) for syn in union_synapses])
	
		for st in sources_targets:
			parents = [self,other]
			P = random.choice(parents)
			while not any(syn.source.ID == st[0] and syn.target.ID == st[1] for syn in P.synapses):
				parents.remove(P)
				P = parents[0]
			apse = [syn for syn in P.synapses if syn.source.ID == st[0] and syn.target.ID == st[1]][0]
			S = P.fetchNeuron(st[0])
			T = P.fetchNeuron(st[1])
			child_synapses.append(Synapse(source=S,target=T, weight=apse.weight))

		assert len(child_synapses) == len(sources_targets)


		for syn in child_synapses:
			if syn in motherSynapses and syn in fatherSynapses:
				syn.enabled = motherSynapses[motherSynapses.index(syn)].enabled and fatherSynapses[fatherSynapses.index(syn)].enabled
		# we now need to compile a list of the neurons included in these links
		child_neurons = []
		for syn in child_synapses:
			if syn[0] not in child_neurons:
				child_neurons.append(syn[0])
			if syn[1] not in child_neurons:
				child_neurons.append(syn[1])

		for i in xrange(self.inOut[0]):
			for neuron in child_neurons:
				if neuron.ID == i:
					neuron.IN = True
		for i in xrange(self.inOut[0], self.inOut[0]+self.inOut[1]):
			for neuron in child_neurons:
				if neuron.ID == i:
					neuron.OUT = True
		# now to set and weight the synapses of the child
		self.synapses.sort()
		other.synapses.sort()
		child_synapses.sort()
		try:
			print"\nmother: "+self.name
			for s in self.synapses: print s,
			print"\nfather: "+other.name
			for s in other.synapses: print s,
			print"\nchild:"
			for s in child_synapses: print s,
			print"\n"
		except:
			pass
	
		# if we want to build the child inside this module, use the following statement. Otherwise we'll assume we're calling the child's subphenotype from outside this module, in the main body of whatever programme this gets used in. 

		#child = NEATnet(inOut=[len(self.in_neurons),  len(self.out_neurons)], neurons=child_neurons, synapses=child_synapses,  initial_pop=False)

		return {"neurons":child_neurons, "synapses":child_synapses}

	def feedForward(self, inputs):
		# the idea here is that each call of the feedForward function passes each signal along at least one synapse. In order for the net to operate as expected, this function needs to be called repeatedly. 
		for neuron in self.out_neurons:
			neuron.state = neuron.decay(neuron.state)

		if len(self.neurons) > len(self.synaptic_array) or len(self.neurons) > len(self.synaptic_array[-1]):
			print "SOMETHING'S FISHY WITH THE ARRAY..."
			self.enlargeArray()

		#print inputs

		for i in xrange(len(inputs)):
			self.in_neurons[i].state = inputs[i]

		outputs = []
		# let's try reversing this
		#old version"
		# for i in xrange(len(self.neurons)):
		# 	# self[i] is the neuron in this net with ID == i.
		# 	for j in xrange(len(self.neurons)):
		# 		if self.synaptic_array[i][j] != None:
		# 			if self.synaptic_array[i][j].enabled == True:
		# 				signal = self[i].threshold(self[i].state) * self.synaptic_array[i][j].weight
		# 				self[j].state += signal
		# 	if self.neurons[i] in self.out_neurons:
		# 		outputs.append(self.neurons[i].state)
		# 	self.neurons[i].state = self.neurons[i].decay(self.neurons[i].state)
		for neuron in self.neurons:
			neuron.fired = 0
			neuron.gathered = False
		for neuron in self.in_neurons:
			neuron.gathered = True
		self.updateFeeds()
		looped = 0
		# new backwards version
		while any(neuron.fired < neuron.feeds for neuron in self.neurons):
			looped += 1
			if looped > 2:

				print "feedforward looped",looped,"times"
				print self.name," of generation",self.gen,"has",len(self.neurons),"neurons and",len(self.synapses),"synapses"
				for n in self.neurons:
					if n.fired < n.feeds:
						print self.name+"'s neuron",n.ID,"feeds into",n.feeds,"neurons but has only fired",n.fired,"time(s)"
				if looped > 100:
					break
			for neuron0 in self.neurons:
				feeders = (neuron for neuron in self.neurons if neuron.gathered and self.synaptic_array[neuron.ID][neuron0.ID] is not None and self.synaptic_array[neuron.ID][neuron0.ID].enabled)
				# self[i] is the neuron in this net with ID == i.
				for neuron1 in feeders:			
					signal = neuron1.threshold(neuron1.state) * self.synaptic_array[neuron1.ID][neuron0.ID].weight
					neuron0.state += signal
					neuron1.fired += 1
				neuron0.gathered = True # not sure if this is enouggh. instead try something dual to .feeds? like .eats? an int instead of a bool? how do we ensure thoroughness?
					
				if neuron0 in self.out_neurons:
					outputs.append(neuron0.state)

				if neuron.fired == neuron.feeds:
					neuron0.state = neuron0.decay(neuron0.state)
			# not going to work for augmented nets. some neurons will decay before they get to fire.
		# if looped > 1:
		# 	print "feedforward looped",looped,"times"
		# 	print self.name,"has",len(self.neurons)-(len(self.in_neurons)+len(self.out_neurons)),"hidden neurons"
		# there, that should do the trick, with no need for layers! 
		# outputs = [neuron.state for neuron in self.out_neurons]
		#print outputs
		return outputs

	def __getitem__(self, index):

		for neuron in self.neurons:
			if neuron.ID == index:
				return neuron

		raise IndexError


