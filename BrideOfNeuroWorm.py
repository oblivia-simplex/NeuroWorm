from NEAT import * # mine
#from bitarray import bitarray
import random
import pylab 
import numpy as np
import os
from operator import itemgetter, attrgetter, methodcaller

import pygame, sys
from pygame.locals import *

import networkx as nx

##################################

############# here's a big one

class WormyPhenotype(NEATnet):

	def __init__(self, x=0, y=0, inOut=[5,1], starting_length=7,
                     maturity=0, segments=[], gen=0, max_stamina=0.0,
                     mutation_rate=0.3, neurons=[], synapses=[],
                     initial_pop=True, brain=None, predator_ratio=0.0):

		self.mutation_rate = mutation_rate
		self.mutations = 0
		self.inOut = inOut
		#self.dropout = dropout
		self.stamina = max_stamina
		self.max_stamina = max_stamina
		self.maturity = maturity
		self.specimen = False
		self.inclination = 0.0
		self.newborn = False
		self.belly = []
		self.gen = 0
		self.offspring = 0
		self.propulsion = 0
		self.moved = False
		self.odour = [0.0,0.0]
		self.obstructed = False
		self.human = False
		self.pulse = 0.0
		self.blink = 0
		self.in_heat = False
		self.bitten = 0
		if random.random() < predator_ratio:
			self.predator = True
		else:
			self.predator = False
		super(WormyPhenotype, self).__init__(inOut, neurons=neurons,
		                                     synapses=synapses,
                                                     initial_pop=initial_pop)

		# if the worm is not one of the initial pop, then it must be a
		# child, and so it's already been passed through the super class
		# (NEAT)

		if not segments:
			self.segments = [[x-i,y] for i in xrange(starting_length)]
		else: 
			self.segments = segments
		self.direction = random.choice(((0,1),(0,-1),(1,0),(-1,0)))
		self.makeEyes()

	
	def makeEyes(self):
		self.eyes =[[],[]]
		if self.direction[0] == 0: # worm vertical
			self.eyes[0] = [self.segments[0][0]+1, self.segments[0][1]]
			self.eyes[1] = [self.segments[0][0]-1, self.segments[0][1]]
		else:
			self.eyes[0] = [self.segments[0][0], self.segments[0][1]-1]
			self.eyes[1] = [self.segments[0][0], self.segments[0][1]+1]

		if -1 in self.direction:
			self.eyes[0],self.eyes[1] = self.eyes[1],self.eyes[0]

	def sensorimotor(self):
		obstructed = float(self.obstructed)
		drone = min(0.5, abs(self.pulse))
		inputs = [self.odour[0], max(self.odour[1],obstructed),
		          self.odour[2], drone]
		# switch 1 for self.pulse
		output = self.feedForward(inputs)
		self.inclination += np.tanh(output[0])/8 #np.tanh(output[0]/100)/4
		# if self.specimen:
		# 	print self.inclination
		if self.predator:
			self.propulsion += np.tanh(output[1])/2
		else:
			self.propulsion += np.tanh(output[1])
		#print inputs,"======>",out
		threshold = 0.5

		if self.inclination < -threshold:
			rudder = -1
			self.inclination = 0.0
		elif -threshold <= self.inclination <= threshold:
			rudder = 0
		elif threshold < self.inclination:
			rudder = 1
			self.inclination = 0.0

		# EXPERIMENTAL

		if abs(self.propulsion) >= threshold:
			self.go = True
			self.propulsion = 0.0
		else:
			self.go = False

		# if random.random() < 0.05 and abs(out)<0.1:
		# 	rudder = random.choice((-1,0,1))

		# if self.specimen: print str(inputs[0])[:6]+("
		# 	"*10)+str(inputs[1])[:7]+" =========>
		# 	"+(str(self.inclination)+"
		# 	")[:6]+(str(self.propulsion)+" ")[:6]+" --->
		# 	"+str(rudder),str(self.go),"
		# 	GEN:",self.gen,f"OFFSPRING:",self.offspring,"CON:",
                # self.connectivity

		return rudder

	def mate(self, other, predator_ratio=0.0):
		# parameters for excess, disjoint, and weight difference
		# sex_sound.play()
		EX  = 1.0
		DIS = 1.0
		W   = 1.0
		 
		# if self.speciesDistance(other, EX, DIS, W) > return we'll fill
		# 	all this in later. Not sure what the values should look
		# 	like yet. For the time being, every worm can mate with
		# 	every other worm. No speciation yet.
		print"With EX = DIS = W = 1.0, the species distance", \
                "between",self.name,"and",other.name,"=", \
                self.speciesDistance(other, EX, DIS, W)
		child_schemata = {}
		child_schemata[self] = self.crossover(other)
		child_schemata[other] = other.crossover(self)
		# uncomment above line to have each mating pair make 2 children 
		children = []
		for parent in child_schemata.keys():
			breakpoint = len(parent.segments)/2
			
			child = WormyPhenotype(inOut=parent.inOut,
                                               neurons=child_schemata[parent]["neurons"],
                                               synapses=child_schemata[parent]["synapses"],
                                               initial_pop=False,
                                               maturity=self.maturity,
                                               segments=parent.segments[breakpoint:] )
			del parent.segments[breakpoint:]
			
			_rate = parent.mutation_rate
			
			if random.random() < child.mutation_rate:
				child.mutate(probWeightVsTopology=0.7)
				child.mutations += 1

			parent.offspring += 1
			child.direction = (lambda d: (d[0] * -1, d[1] * -1))(parent.direction)
			child.newborn = 5000
		
			if random.random()< predator_ratio:
				child.predator = True
			else:
				child.predator = False
			if child.speciesDistance(parent, EX, DIS, W) > 1:
				child.max_stamina = parent.max_stamina*1.2
			else:
				child.max_stamina = parent.max_stamina
			children.append(child)
			

		self.in_heat = False
		other.in_heat = False
		return children

class Grid:

	def __init__(self, grid_size):
		
		self.window_width 		= grid_size[0]*grid_size[2]
		self.window_height 		= grid_size[1]*grid_size[2]
		self.cell_size			= grid_size[2]
		self.cell_width			= grid_size[0]
		self.cell_height		= grid_size[1]
		self.hypoteneuse		= np.sqrt(self.cell_width**2 + self.cell_height**2)

		
		self.occupying = [[[] for i in xrange(self.cell_width)] for j in
		                  xrange(self.cell_height)]
		# being an array of lists that contain everything at every cell of the grid. may or may not be useful. 


	def distance(self, coord1, coord2):
		x_delta = abs(coord1[0]-coord2[0])
		y_delta = abs(coord1[1]-coord2[1])
		h = np.sqrt(x_delta**2 + y_delta**2)
		return h

	def whereIs(self, item):
		"""Probably not very efficient (O(n) for n = # of cells),
                but a function for searching the grid for various items.
                Returns a list of coordinates at which the item sought can
                be found."""
		coords = []
		for w in xrange(self.width):
			for h in xrange(self.height):
				if item in self.occupying[w][h]:
					coords.append([w,h])
		return coords
        # consider using push instead of append, for slightly better efficiency

	def randomCell(self, near=False, how_near=10):
		if not near:
			x = random.randrange(self.cell_width)
			y = random.randrange(self.cell_height)
		else:
			x = near[0]+random.randrange(-how_near, how_near)
			y = near[1]+random.randrange(-how_near, how_near)
			if x < 5:
				x = 5
			elif x > self.cell_width-5:
				x = self.cell_width-5
			if y < 0:
				y = 5
			elif y > self.cell_height-5:
				y = self.cell_height-5

		return [x,y]

class Wormosphere:

	def __init__(self, population_size=0, max_pop=30, min_pop=10,
	             mutation_rate=0.1, cell=10, population=[], starting_length=5,
	             apple_count=12, max_stamina=0.0, maturity=0, inOut=[5,2],
	             apple_move=0.01, predator_ratio=0.0):

		pygame.init()
		grid_size = [700/cell, 700/cell, cell]
		# get names
		with open("data/names.txt") as f:
			name_list = f.read().splitlines()
		name_list = [name.rstrip() for name in name_list]
		with open("data/demonNames2.txt") as f:
			name_list2 = f.read().splitlines()
		name_list2 = [name.rstrip().upper() for name in name_list2]
		name_list += name_list2
		self.name_list = [name for name in name_list if 3 <= len(name) <= 10]
		self.name_list.sort()
		#print self.name_list
		self.population_size 	= population_size
		self.max_pop			= max_pop
		self.min_pop			= min_pop
		self.popIter			= xrange(self.population_size)
		self.grid_size 			= grid_size # WIDTH by HEIGHT by CELLSIZE
		#self.genome_size		= 300
		self.apple_move			= apple_move
		self.inOut				= inOut
		self.mutation_rate		= mutation_rate
		self.starting_length  	= starting_length
		self.fps_clock 			= pygame.time.Clock()
		self.fps 				= 60
		self.compass 			= [(0,-1),(1,0),(0,1),(-1,0)]
		self.grid = Grid(self.grid_size)
		self.max_stamina 		= max_stamina
		self.predator_ratio		= predator_ratio
		self.maturity			= maturity
		self.population 		= population
		self.populate()
		self.apple_count		= apple_count
		self.min_apples			= max(apple_count/2,1)
		self.max_apples			= apple_count *2
		self.apples 			= []
		self.makeApples()
		self.theSpecimen 		= False
		self.humanWorm			= False

		self.basicfont = pygame.font.SysFont('monospace', 12)
		self.displaySurf = pygame.display.set_mode((self.grid.window_width, self.grid.window_height))

	def drawGrid(self):
		#             R    G    B
		WHITE     = (255, 255, 255)
		BLACK     = (  0,   0,   0)
		RED       = (255,   0,   0)
		PINK 	  = (255, 150, 170)
		DARKPINK  = (155, 75,   80)
		GREEN     = (  0, 255,   0)
		BLUE 	  = (  0,   0,  150)
		DARKGREEN = (  0, 100,   0)
		GRAYGREEN = (40,  100, 40)
		DARKGRAY  = ( 40,  40,  40)
		GRAY 	  = (100, 100, 100)
		PURPLE    = (80,   0,  100)
		DARKPURPLE = (50,  0,   80)
		BGCOLOR   = BLACK

		self.displaySurf = \
		pygame.display.set_mode((self.grid.window_width,
		                         self.grid.window_height))
		self.displaySurf.fill(BGCOLOR)

		for x in xrange(0, self.grid.window_width, self.grid.cell_size):
			# draw vertical lines
			pygame.draw.line(self.displaySurf,
                                         DARKGRAY, (x,0), (x, self.grid.window_height))
		for y in xrange(0, self.grid.window_height, self.grid.cell_size):
			# draw vertical lines
			pygame.draw.line(self.displaySurf, DARKGRAY, (0,y),
			                 (self.grid.window_width,y))

		# draw the worm
		for worm in self.population:
			for segment in worm.segments:
				wormRect = \
				           pygame.Rect(segment[0]*self.grid.cell_size,
				                segment[1]*self.grid.cell_size,
				                self.grid.cell_size, self.grid.cell_size)
			
				hidden_neurons = \
				                 (len(worm.neurons)-(len(worm.in_neurons)+ \
                                                    len(worm.out_neurons)))
				colour = (100, min(255, 10*hidden_neurons),
                                          min(255, len(worm.belly)*10))
				if not worm.predator:
					colour = (min(255, 10*hidden_neurons),
                                                  100, min(255, len(worm.belly)*10))
					EYEWHITE = WHITE
					if worm.specimen:
						colour = GREEN
					if worm.newborn:
						colour = PINK
						if segment == worm.segments[-1]:
							worm.newborn -= 1
					elif worm.in_heat:
						colour = PURPLE
					if worm.human:
						colour = (0,255,255)
					if worm.bitten > 0:
						if worm.bitten % 2 == 0:
							colour = RED
						else:
							colour = DARKPINK
						worm.bitten -= 1
				else:
					colour = (80, min(255, 10*hidden_neurons),
                                                  min(255, len(worm.belly)*10))
					EYEWHITE = PINK
					if worm.specimen:
						colour = GREEN
					if worm.newborn:
						colour = DARKPINK
						if segment == worm.segments[-1]:
							worm.newborn -= 1
					elif worm.in_heat:
						colour = PURPLE
					if worm.human:
						colour = (0,255,255)


				pygame.draw.rect(self.displaySurf, colour, wormRect)
			eyeRects = [0,0]
			pupilRects = [0,0]
			eyeWhiteRects = [0,0]
			eyeInd = 0
			pupilColour = (min(worm.mutations*10, 255), 0, 0)
			for eye in worm.eyes:
				eyeRects[eyeInd] = \
				pygame.Rect(eye[0]*self.grid.cell_size,
				            eye[1]*self.grid.cell_size,
                                            self.grid.cell_size,
				            self.grid.cell_size)

				eyeWhiteRects[eyeInd] = \
				pygame.Rect(eye[0]*self.grid.cell_size+
                                            (self.grid.cell_size/5.0),
				            eye[1]*self.grid.cell_size+
                                            (self.grid.cell_size/5.0),
				            self.grid.cell_size-
                                            (self.grid.cell_size/3.),
				            self.grid.cell_size-
                                            (self.grid.cell_size/3.))

                                pupilRects[eyeInd] = pygame.Rect(eye[0]*self.grid.cell_size+(self.grid.cell_size/3.), eye[1]*self.grid.cell_size+(self.grid.cell_size/3.), self.grid.cell_size-2*(self.grid.cell_size/3.), self.grid.cell_size-2*(self.grid.cell_size/3.))
				pygame.draw.rect(self.displaySurf, colour,
                                                 eyeRects[eyeInd])

				if random.random() < 0.003:
					worm.blink = 2
				if worm.pulse > 0.9 and worm.blink > 0:
					worm.blink -= 1

				if worm.blink <= 0:
					pygame.draw.rect(self.displaySurf,
                                                         EYEWHITE,
                                                         eyeWhiteRects[eyeInd])
					pygame.draw.rect(self.displaySurf,
                                                         pupilColour,
                                                         pupilRects[eyeInd])
				eyeInd += 1
		for apple in self.apples:
			appleRect = pygame.Rect(apple[0]*self.grid.cell_size,
			                        apple[1]*self.grid.cell_size,
                                                self.grid.cell_size,
			                        self.grid.cell_size)
			pygame.draw.rect(self.displaySurf, RED, appleRect)

	def makeApples(self):

		applesNeeded = max(0, self.apple_count - len(self.apples)) 
		for a in xrange(applesNeeded):
			if self.apples == []:
				near = False
				how_near = False
			else:
				near = self.apples[-1]
				how_near = 10
			self.apples.append(self.grid.randomCell(near=near,
                                                                how_near=how_near))
		for apple in self.apples:
			if random.random() < self.apple_move:
				c = random.randrange(2)
				d = random.choice((-1,1))
				allsegs = [] 
				for w in self.population:
					allsegs.extend(w.segments)
				prospective = apple[c]+d
				if prospective not in allsegs:
					apple[c] += d
			if apple[0] < 5:
				apple[0] = 5
			elif apple[0] > self.grid.cell_width-5:
				apple[0] = self.grid.cell_width-5
			if apple[1] < 0:
				apple[1] = 5
			elif apple[1] > self.grid.cell_height-5:
				apple[1] = self.grid.cell_height-5
		## intelligent apples:
		# sniff = {}
		# for d in self.compass:
			# 	sniff[self.scentsAt(apple, d, seeking="worms")] = d
			# best = sniff[min(sniff.keys())]
			# print best
			# print apple
			# apple = [apple[0]+best[0], apple[1]+best[1]]
			# print apple

	def populate(self):

		for i in xrange(self.population_size - len(self.population)):

			while 1:
				x, y = random.randrange(self.grid_size[0]), \
                                       random.randrange(self.grid_size[1])
				goodtogo = True
				for i in xrange(self.starting_length):
					goodtogo = goodtogo and \
                                                   self.grid.occupying[x-i][y] == []
				if goodtogo:
					break
			# make the Genome
			# genome = bitarray()
			# for i in xrange(self.genome_size+1):
			# 	genome.append(random.randint(0,1))
			newWorm = WormyPhenotype(x=x, y=y,
			                         starting_length=self.starting_length,
			                         max_stamina=self.max_stamina,
                                                 maturity=self.maturity,
			                         inOut=self.inOut,
                                                 mutation_rate=self.mutation_rate,
			                         predator_ratio=self.predator_ratio)
			newWorm.name = (random.choice(self.name_list))
			self.population.append(newWorm)
			newWorm =0

			# for i in xrange(self.starting_length):
			# 	grid.occupying[x-i][y].append("WORM")

	def scentsAt(self, coords, direction, seeking="apples"):
		"""Returns the potency of worm odours and the potency of apple odours at a given cell, at the time that the function is called."""
		#wormScent = 0.0
		scent = 0.0
		N, E, S, W = self.compass

		# for worm in self.population:
		# 	for segment in worm.segments:
		# 		wormScent += self.grid.distance(segment, coords)**EXP
		totalScent = 0.0
		count = 0
		canSmell = False

		target = []
		if seeking=="apples":
			target = self.apples 
		elif seeking in ["mate", "predmate","worm", "prey", "predator"]:
			if seeking == "mate":
				marker = (lambda x: x.in_heat)
			elif seeking == "prey":
				marker = (lambda x: not x.predator)
			elif seeking == "predator":
				marker = (lambda x: x.predator)
			else:
				marker = (lambda x: True)
			for worm in self.population:
				if marker(worm) and coords not in worm.segments:
					target.extend(worm.segments)

		for cell in target:
			if target == []:
				break
			if direction == N:
				if cell[1] <= coords[1] and abs(coords[0] - cell[0]) \
                                   <= abs(coords[1] - cell[1]):
					canSmell = True
					totalScent += self.grid.distance(cell,
                                                                         coords)\
                                                                         /self.grid.hypoteneuse
					#print direction, canSmell
			elif direction == E:
				if cell[0] >= coords[0] and abs(coords[0] -
				                                cell[0]) >= abs(coords[1] - cell[1]):
					canSmell = True
					totalScent += self.grid.distance(cell,
					                                 coords)/self.grid.hypoteneuse
					#print direction, canSmell
			elif direction == S:
				if cell[1] >= coords[1] and abs(coords[0] -
				                                cell[0]) <= abs(coords[1] - cell[1]):
					canSmell = True
					totalScent += self.grid.distance(cell,
					                                 coords)/self.grid.hypoteneuse
					#print direction, canSmell
			elif direction == W:
				if cell[0] <= coords[0] \
                                   and abs(coords[0] - cell[0]) >= abs(coords[1]
                                                                       - cell[1]):
					canSmell = True
					totalScent += self.grid.distance(cell,
                                                                         coords)/self.grid.hypoteneuse
		if target == [] or not canSmell:
			scent = 0.0
		else:
			scent = 1-(totalScent / len(target))
		return scent


	def popRegulate(self):
		if len(self.population) > self.max_pop:
			self.apple_count = self.min_apples
		elif len(self.population) < self.min_pop:
			self.apple_count = self.max_apples
		else:
			self.apple_count = self.max_apples/2
		if len(self.population) <= 3:
			self.populate()

		# # rapture mode:
		# if len(self.population) >= self.max_pop:
		# 	for w in self.population:
		# 		if len(w.segments) <= self.starting_length:
		# 			self.population.remove(w)


	def moveWorm(self, worm):
		N, E, S, W = self.compass
		# if worm.direction == N:
		#     newHead = [worm.segments[0][0], worm.segments[0][1] - 1]
		# elif worm.direction == S:
		#     newHead = [worm.segments[0][0], worm.segments[0][1] + 1]
		# elif worm.direction == W:
		#     newHead = [worm.segments[0][0] - 1,worm.segments[0][1]]
		# elif worm.direction == E:
		#     newHead = [worm.segments[0][0] + 1,worm.segments[0][1]]
		#collision
		# if [w for w in self.population if newHead in w.segments] == []:
		worm.segments.insert(0, newHead)
		# 	collision = True
		# else:
		# 	collision = False

		worm.makeEyes()

		worm.moved = True

	def neuroGraph(self, specimen ,node_size=2000, node_color='green',
	               node_alpha=0.6, node_text_size=12, edge_color='black',
                       edge_alpha=0.6, edge_tickness=1, edge_text_pos=0.3,
                       text_font='ubuntumono'):
		pylab.clf()
		# create networkx (directed) graph
		G=nx.DiGraph()
		print"\nSpecimen:",specimen.name
		for synapse in specimen.synapses:
			print synapse,
		print "\n"
		# extract nodes from graph
		nodes = set(neuron.ID for neuron in specimen.neurons)
		#set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

		# add nodes
		for node in nodes:
			G.add_node(node)

		words = ["L.EYE","NOSE", "TAIL",
		         "R.EYE","PULSE","RUDDER","SPEED"]+range(7,len(nodes))
		node_labels = {}
		for i in xrange(len(nodes)):
			node_labels[i] = str(words[i])
		nodelist = list(nodes)
		nodelist.sort()
		# add edges
		for synapse in specimen.synapses:
			if synapse.enabled:
				G.add_edge(synapse.source.ID, synapse.target.ID,
				           w=float(str(synapse.weight)[:10]))

		edge_labels = nx.get_edge_attributes(G,'w')
		#print edge_labels
		# draw graph
		pos = nx.shell_layout(G)
		nx.draw_networkx_edge_labels(G, pos, label_pos=0.5,
		                             font_weight="bold", bbox={"facecolor":"none",
		                                                       "edgecolor":"none"})
		nx.draw_networkx_labels(G, pos, node_labels, font_size=10,
		                        font_color='k', alpha=edge_alpha)
		nx.draw_networkx_nodes(G, pos, nodelist[:5], alpha=node_alpha,
		                       node_size=node_size, node_color='g', node_shape='h')
		nx.draw_networkx_nodes(G, pos, nodelist[5:7],alpha=node_alpha,
		                       node_size=node_size, node_color='b', node_shape='h') 
		nx.draw_networkx_nodes(G, pos, nodelist[7:], alpha=node_alpha,
		                       node_size=node_size, node_color='c', node_shape='h')  # so^>v<dph8
		nx.draw_networkx_edges(G, pos, alpha=edge_alpha)

		#nx.draw(G, pos, node_size=node_size)
		# show graph
		pylab.axis("off")
		pylab.title(specimen.name)
		pylab.draw()


	def specimenData(self):
		"""Displays some information about the currently selected worm,
                at the top of the gamespace window."""

		specimenSurf = self.basicfont.render("%(name)s SENSE: %(odr)s  [IN: %(IN)s  HID: %(HID)s  OUT: %(OUT)s  SYN: %(syn)s]   GEN: %(gen)s SPAWN: %(kids)s   FOOD: %(food)s POP: %(pop)s" %{'name':(self.theSpecimen.name[:11]+":"+" "*9)[:12],
				'gen': self.theSpecimen.gen, 
				'kids': self.theSpecimen.offspring, 
				'pop': len(self.population), 
				'odr':("("+(str(self.theSpecimen.odour[0])+" ")[:4])+", "+(str(self.theSpecimen.odour[1])+" ")[:4]+", "+(str(self.theSpecimen.odour[2])+" ")[:4]+", "+(str(self.theSpecimen.odour[3])+" ")[:4]+")", 
				'food':len(self.apples), 
				'HID':len(self.theSpecimen.neurons)-(len(self.theSpecimen.in_neurons+self.theSpecimen.out_neurons)), 
				'IN':len(self.theSpecimen.in_neurons), 
				'OUT':len(self.theSpecimen.out_neurons), 
				'syn':len([syn for syn in self.theSpecimen.synapses if syn.enabled])
					}, True, (255,255,255))
		specimenRect = specimenSurf.get_rect()
		specimenRect.topleft = (10,10)
		self.displaySurf.blit(specimenSurf, specimenRect)

	def savePopulation(self, filename):
		self.population.sort(key=attrgetter('name')) # sort the pop first.
		description = "%(line)s\n# NEUROWORM POPULATION OF %(POP)s AT CYCLE %(CYC)s # \n%(line)s\n"%{"POP":len(self.population), "CYC":self.timer, "line":("#"*(40+len(str(self.timer))))}
		path = "./populations/"
		saving = open(path+filename+".pop", 'w')
		saving.write(description+"\n")
		for creature in self.population:

			bitstring = str(creature.genome)[10:-2]
			saving.write("# %(nm)s OF GENERATION %(gen)s: \n"%{"nm":creature.name, "gen":creature.gen})
			saving.write(bitstring+"\n")
		saving.close()

	def infancy(self, worm):
		if worm.belly == []:
			worm.mutate(probability=1.0,probWeightVsTopology=1.00)
		else:
			worm.newborn = False

	def terminate(self):
		pygame.quit()
		filename = raw_input("Save population as: ")
		self.savePopulation(filename)
		sys.exit()

	def runWormosphere(self):
		pylab.ion()
		pygame.display.set_caption("BRIDE OF NEUROWORM")
		# initialize some variables
		spec = random.choice(self.population)

		spec.specimen = True
		self.theSpecimen = spec
		self.neuroGraph(self.theSpecimen)
		justDid = 0
		# some syntactic sugar:
		N, E, S, W = self.compass
		# i.e. [(0,-1),(1,0),(0,1),(-1,0)] represents the four
		# directions. The idea being that N is a negative movement on
		# the y axis, E is a positive movement on the x axis, etc...
		# This is handy because you can reverse directions by
		# multiplying any of these by (-1,-1). In fact:
		reverse = lambda d: (d[0] * -1, d[1] * -1)

		fertility = 0
		self.timer = 0
		#             R    G    B
		# WHITE     = (255, 255, 255)
		# BLACK     = (  0,   0,   0)
		# RED       = (255,   0,   0)
		# GREEN     = (  0, 255,   0)
		# DARKGREEN = (  0, 155,   0)
		# DARKGRAY  = ( 40,  40,  40)
		# BGCOLOR   = BLACK

		while 1:
			self.timer += 1
			if self.timer % 1000 == 0:
				neuron_count = [len(w.neurons) for w in self.population]
				avg_neuron_count = sum(neuron_count)/float(len(neuron_count))
				syn_count = [len(w.synapses) for w in self.population]
				avg_syn_count = sum(syn_count)/float(len(syn_count))
				max_syn_count = max(syn_count)
				max_neuron_count = max(neuron_count)
				print"Population:",len(self.population)
				print"Average number of neurons per worm:",avg_neuron_count
				print"Average number of synapses per worm:",avg_syn_count
				print"Highest neuron count:",max_neuron_count
				print"Highest synapse count:", max_syn_count

				brainy = [w for w in self.population if len(w.neurons) == max_neuron_count or len(w.synapses) == max_syn_count]
				spec = random.choice(brainy) 
				self.theSpecimen.specimen = False
				spec.specimen = True
				self.theSpecimen = spec
				self.neuroGraph(self.theSpecimen)

			if self.timer % (self.max_stamina/3) == 0:
				self.popRegulate()

			self.makeApples()

			for event in pygame.event.get(): # event handling loop
				if event.type == QUIT:
					self.terminate()

				elif event.type == KEYDOWN and event.key == K_ESCAPE:
					self.terminate()
		### for moving
			for worm in self.population:
				grow = False
				if self.timer % 500 == 0:
					if worm.newborn:
						self.infancy(worm)
				if pygame.mouse.get_pressed()[0] == True:
					mouse_x, mouse_y = pygame.mouse.get_pos()
					mouse_x, mouse_y = int(mouse_x /
					                       self.grid.cell_size),
                                        int(mouse_y / self.grid.cell_size)

					for w in self.population:
						if [mouse_x, mouse_y] in w.segments:
							if self.timer > justDid + 10:
								self.theSpecimen.specimen = False
								self.theSpecimen = w
								w.specimen = True
								self.neuroGraph(self.theSpecimen)
							justDid = self.timer

							break
				if pygame.mouse.get_pressed()[1] == True:

					mouse_x, mouse_y = pygame.mouse.get_pos()
					mouse_x, mouse_y = int(mouse_x /
					                       self.grid.cell_size), \
                                                               int(mouse_y /
                                                                   self.grid.cell_size)

					for w in self.population:
						if [mouse_x, mouse_y] in w.segments:
							if w.human:
								w.human = False
								self.humanWorm = False
							else:
								if self.humanWorm:
									self.humanWorm.human = False
								self.humanWorm = w
								w.human = True
							break

				if self.humanWorm:
					for event in pygame.event.get(): # event handling loop
						if event.type == QUIT:
							self.terminate()
						if event.type == KEYDOWN:
							if (event.key == K_LEFT or event.key == K_a) and self.humanWorm.direction != E:
								self.humanWorm.direction = W
							elif (event.key == K_RIGHT or event.key == K_d) and self.humanWorm.direction != W:
								self.humanWorm.direction = E
							elif (event.key == K_UP or event.key == K_w) and self.humanWorm.direction != S:
								self.humanWorm.direction = N
							elif (event.key == K_DOWN or event.key == K_s) and self.humanWorm.direction != N:
								self.humanWorm.direction = S
							elif event.key == K_ESCAPE:
								self.terminate()

				if pygame.mouse.get_pressed()[2]:

					mouse_x, mouse_y = pygame.mouse.get_pos()
					mouse_x, mouse_y = int(mouse_x / self.grid.cell_size), int(mouse_y / self.grid.cell_size)
					if self.timer > justDid + 10:
						self.apples.append([mouse_x, mouse_y])
					justDid = self.timer



				if worm.moved and not worm.newborn:
					worm.stamina -= 1-(len(worm.neurons)/float(len(worm.synapses))) #(worm.connectivity[0]+worm.connectivity[1])/2
				else:
					worm.stamina -= (1-(len(worm.neurons)/float(len(worm.synapses))))/3 #(worm.connectivity[0]+worm.connectivity[1])/6

				still_prey = len([w for w in self.population if not w.predator]) > 0


				if worm.in_heat and len(worm.segments) >= worm.maturity - 1:
					food = "mate"
				# elif worm.in_heat and len(worm.segments) >= worm.maturity - 1 and worm.predator:
				# 	food = "predmate"
				elif not worm.predator:
					food = "apples"
				elif worm.predator and still_prey and not worm.in_heat:
					food = "prey"
				else:
					food = "worm"

				worm.odour = [self.scentsAt(worm.eyes[0],worm.direction, seeking=food), self.scentsAt(worm.segments[0], worm.direction, seeking="worm"), self.scentsAt(worm.segments[-1], reverse(worm.direction), seeking="predator"), self.scentsAt(worm.eyes[1], worm.direction,seeking=food)]
				#print worm.odour

				## THE SENSORIMOTOR FUNCTION ##
				worm.pulse = np.sin(self.timer)
				if not worm.human:
					rudder = worm.sensorimotor()
				else:
					worm.go = True

				#print "rudder:",rudder

				directionIndex = self.compass.index(worm.direction)
				directionIndex = (directionIndex+rudder)%4
				worm.direction = self.compass[directionIndex]

				# move the worm by adding a segment in the direction it is moving
				worm.moved = False

				nextStep = [worm.segments[0][0]+worm.direction[0], worm.segments[0][1]+worm.direction[1]]
				worm.obstructed = False

				# obstruction and mating and predation
				for w in self.population:
					if w != worm and nextStep in w.segments:
						worm.obstructed = True
						# maybe make them have to touch for a few cycles before they pool their stamina...
						stampool = (worm.stamina+w.stamina)/2
						worm.stamina = stampool
						w.stamina = stampool


						if worm.in_heat and w.in_heat:

							children = worm.mate(w, self.predator_ratio)
							for child in children:
								child.name = random.choice(self.name_list)
								self.population.append(child)
						# PREDATION! 
						if worm.predator and (not w.predator or not still_prey) and not w.in_heat and not worm.in_heat and not w.newborn:
							worm.belly.append(w.segments.pop())
							w.bitten = 100
							#bite_sound.play()

							print worm.name,"BIT",w.name
							grow = True
							worm.obstructed = False
						break

				if worm.go and not worm.obstructed:
					worm.segments.insert(0, nextStep)
					worm.makeEyes()
					worm.moved = True

				# elif worm.obstructed:
				# 	worm.direction = reverse(worm.direction)

				# the world is round:
				if worm.segments[0][0] == -1:
					worm.segments[0][0] = self.grid.cell_width-1
				elif worm.segments[0][0] == self.grid.cell_height:
					worm.segments[0][0] = 0
				if worm.segments[0][1] == -1:
					worm.segments[0][1] = self.grid.cell_height-1
				elif worm.segments[0][1] == self.grid.cell_height:
					worm.segments[0][1] = 0


				# did the worm eat an apple?
				for bite in [worm.segments[0], worm.eyes[0], worm.eyes[1]]:
					if bite in self.apples:
						# apple_sound.play()
						self.apples.remove(bite)
						worm.stamina = worm.max_stamina
						grow = True
						if not worm.predator:
							worm.belly.append("a")

				# shrink end of worm
				if (not grow) and worm.moved: #and not collision:
					del worm.segments[-1]
					worm.moved = False

				# did the worm reach cloning size? Reproduction. 
				if len(worm.segments) >= worm.maturity and len(self.population) <= self.max_pop:
					worm.in_heat = True

				# is the worm starving?
				if worm.stamina <= 0. and len(worm.segments) > 1:
					worm.stamina = worm.max_stamina
					del worm.segments[-1]

				# death
				if len(worm.segments) <= 1 and len(self.population) > self.min_pop/2: #failsafe
					self.population.remove(worm)
					print worm.name,"HAS DIED."



				# stress-induced mutations
				# elif len(worm.segments) == 2 and worm.mutations < len(worm.belly)+3 :

				# 	worm.mutate(probability=0.5)

			######### make the picture

			# N = 10000
			# if timer > N:
			self.drawGrid()
			self.specimenData()

			pygame.display.update()
			# 	print timer
			# else:
			# 	print timer,"WILL DISPLAY WHEN THE TIMER REACHES", N
			#self.fps_clock.tick(self.fps)

###############################################

####################
## USER INTERFACE ##
####################


def getNumAnswer(nameOfValue, answerType=int):
	print nameOfValue+":"

	while 1:
		answer = raw_input(">> ")
		try:
			value = answerType(answer)
			break
		except ValueError:
			print"Unacceptable."
	return value


def getYesNoAnswer(nameOfValue):
	print nameOfValue+"? (Y/N)"
	yes = ["Y","YES","SURE","YEP"]
	no = ["N", "NO","NOPE","NO THANKS"]

	while 1:
		answer = raw_input(">> ")	
		if answer.upper() in yes:
			value = True
			break
		elif answer.upper() in no:
			value = False
			break
		else:
			print"Unacceptable."
			pass
	return value
############################################

def main():
	query = not getYesNoAnswer("Use default settings")

	if query:
		population_size = getNumAnswer("Initial population size", int)
		max_pop	= getNumAnswer("Maximum population size")
		min_pop = getNumAnswer("Minimum population size")
		predator_ratio = getNumAnswer("Predator ratio (float between 0.0 and 1.0)", float)
		apple_count = getNumAnswer("Food supply")
		apple_move = getNumAnswer("Food restlessness (float between 0.0 and 1.0)", float)
		max_stamina = getNumAnswer("Maximum stamina per segment")
		starting_length = getNumAnswer("Starting length of initial population of worms")
		maturity = getNumAnswer("Length of worm at which reproduction becomes possible")
		mutation_rate = getNumAnswer("Mutation rate (float between 0.0 and 1.0)", float)
		music = getYesNoAnswer("Music")


	else:
		population_size = 20
		predator_ratio=0.05
		max_pop=40
		min_pop=6
		apple_count=2
		max_stamina=150
		mutation_rate=.20
		starting_length=6
		maturity=12
		apple_move=0.3
		music = True
	
	#pygame.mixer.init()
	#global bite_sound, sex_sound, apple_sound
	#bite_sound = pygame.mixer.Sound("../../soundEffects/Collision8-Bit.ogg")
	#sex_sound = pygame.mixer.Sound("../../soundEffects/laser5.ogg")
	#apple_sound = pygame.mixer.Sound("../../soundEffects/Bloob8Bit.wav")
	#if music:
		
	#	pygame.mixer.music.load("../../soundEffects/PixiesWhereIsMyMind8BitRemix.ogg")
	#	pygame.mixer.music.play(-1,0.0)
	W = Wormosphere(population_size=population_size,
	                predator_ratio=predator_ratio,
                        max_pop=max_pop,
                        min_pop=min_pop,
	                apple_count=apple_count,
                        max_stamina=max_stamina,
	                mutation_rate=mutation_rate,
                        starting_length=starting_length,
	                maturity=maturity,
                        inOut=[5,2],
                        cell=7,
                        apple_move=apple_move)
        # any size cell will do, so long as it's a factor of 700 (5,7,10,20,25).
	# The smaller the cell, the more room the worms will have to roam.
	while 1:
		W.runWormosphere()

# stamina sharing by touch should also be optional. interesting to compare its
# effects with the effects of its absence.
main()
