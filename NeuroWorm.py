from neurophenotype import * # mine

from bitarray import bitarray
import random
import pylab 
import numpy as np
import cPickle
import os
from operator import itemgetter, attrgetter, methodcaller

import pygame, sys
from pygame.locals import *

##################################

############# here's a big one

class WormyPhenotype(NeuroPhenotype):

	def __init__(self, genome, x=0, y=0, layers=[4,5,2], dropout=4, length=7, maturity=16, stamina=300.0, segments=[], gen=0, mutation_rate=0.1):
		# we're going to have the phenotype take a bitarray rather than a genome type as genome. no need to use most of genome's attributes or functions here. we'll rebuild them to better suit the task.
		
		self.genome = genome
		self.mutation_rate = mutation_rate
		self.mutations = 0
		self.layers = layers
		self.dropout = dropout
		self.stamina = stamina
		self.max_stamina = stamina*2
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

		super(WormyPhenotype, self).__init__(self.genome)


		if not segments:
			self.segments = [[x-i,y] for i in range(length)]
		else: 
			self.segments = segments
		self.direction = random.choice(((0,1),(0,-1),(1,0),(-1,0)))
		self.makeEyes()

	
	def makeEyes(self):
		self.eyes =[[],[]]
		if self.direction[0] == 0: # worm vertical
			self.eyes[0] = [self.segments[0][0]-1, self.segments[0][1]]
			self.eyes[1] = [self.segments[0][0]+1, self.segments[0][1]]
		else:
			self.eyes[0] = [self.segments[0][0], self.segments[0][1]-1]
			self.eyes[1] = [self.segments[0][0], self.segments[0][1]+1]


	def sensorimotor(self):
		


		obstructed = float(self.obstructed)
		
		
		inputs = [self.odour[0], self.odour[1], abs(self.pulse), obstructed, 1.0]
		# switch 1 for self.pulse
		output = self.feedForward(inputs)
		self.inclination += np.tanh(output[0] * 200)/4
		self.propulsion += np.tanh(output[1] * 200)*2
		# for a while, I'd accidentally set both nerves to the same output!
		if self.specimen:
			print self.inclination

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

		# if self.specimen:
		# 	print str(inputs[0])[:6]+(" "*10)+str(inputs[1])[:7]+" =========> "+(str(self.inclination)+"   ")[:6]+(str(self.propulsion)+"   ")[:6]+" ---> "+str(rudder),str(self.go),"  GEN:",self.gen,"OFFSPRING:",self.offspring,"CON:",self.connectivity

		return rudder


class Grid:

	def __init__(self, grid_size):
		
		self.window_width 		= grid_size[0]*grid_size[2]
		self.window_height 		= grid_size[1]*grid_size[2]
		self.cell_size			= grid_size[2]
		self.cell_width			= grid_size[0]
		self.cell_height		= grid_size[1]
		self.hypoteneuse		= np.sqrt(self.cell_width**2 + self.cell_height**2)

		
		self.occupying	= [[[] for i in range(self.cell_width)] for j in range(self.cell_height)]
		# being an array of lists that contain everything at every cell of the grid. may or may not be useful. 


	def distance(self, coord1, coord2):
		x_delta = abs(coord1[0]-coord2[0])
		y_delta = abs(coord1[1]-coord2[1])
		h = np.sqrt(x_delta**2 + y_delta**2)
		return h

	def whereIs(self, item):
		"""Probably not very efficient, but a function for searching the grid for various items. Returns a list of coordinates at which the item sought can be found."""
		coords = []
		for w in range(self.width):
			for h in range(self.height):
				if item in self.occupying[w][h]:
					coords.append([w,h])
		return coords

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

	def __init__(self, population_size=0, max_pop=30, min_pop=10, mutation_rate=0.1, cell=10, population=[], starting_length=5, apple_count=12, max_stamina=300, maturity=8, layers=[5,4,2]):
		pygame.init()
		synapses = layers[0]*layers[1] + layers[1]*layers[2]
		dropoutbits = synapses
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
		self.popIter			= range(self.population_size)
		self.grid_size 			= grid_size # WIDTH by HEIGHT by CELLSIZE
		self.genome_size		= 300
		self.layers				= layers
		self.mutation_rate		= mutation_rate
		self.starting_length  	= starting_length
		self.fps_clock 			= pygame.time.Clock()
		self.fps 				= 60
		self.compass 			= [(0,-1),(1,0),(0,1),(-1,0)]
		self.grid = Grid(self.grid_size)
		self.max_stamina 		= max_stamina
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

		self.basicfont = pygame.font.SysFont('ubuntumono', 14)
		self.displaySurf = pygame.display.set_mode((self.grid.window_width, self.grid.window_height))
		

		

	def drawGrid(self):
		#             R    G    B
		WHITE     = (255, 255, 255)
		BLACK     = (  0,   0,   0)
		RED       = (255,   0,   0)
		PINK 	  = (255, 150, 170)
		GREEN     = (  0, 255,   0)
		BLUE 	  = (  0,   0,  150)
		DARKGREEN = (  0, 100,   0)
		GRAYGREEN = (100,  180, 100)
		DARKGRAY  = ( 40,  40,  40)
		BGCOLOR   = BLACK

		self.displaySurf = pygame.display.set_mode((self.grid.window_width, self.grid.window_height))
		self.displaySurf.fill(BGCOLOR)

		for x in range(0, self.grid.window_width, self.grid.cell_size):
			# draw vertical lines
			pygame.draw.line(self.displaySurf, DARKGRAY, (x,0), (x, self.grid.window_height))

		for y in range(0, self.grid.window_height, self.grid.cell_size):
			# draw vertical lines
			pygame.draw.line(self.displaySurf, DARKGRAY, (0,y), (self.grid.window_width,y))

		# draw the worm
		for worm in self.population:
			for segment in worm.segments:
				wormRect = pygame.Rect(segment[0]*self.grid.cell_size, segment[1]*self.grid.cell_size, self.grid.cell_size, self.grid.cell_size)
				
				
				# G = int(200/max(1,getGrayCodeInteger(worm.genome[0:4])))+50
				# B = int(200/max(1,getGrayCodeInteger(worm.genome[4:8])))+50
				# R = int(200/max(1,getGrayCodeInteger(worm.genome[8:12])))+50

				colour = (min(255, 10*worm.gen), 100, min(255, len(worm.belly)*10))
				if worm.specimen:
					colour = GREEN
				if worm.newborn:
					colour = PINK
					if segment == worm.segments[-1]:
						worm.newborn -= 1
				if worm.human:
					colour = (0,255,255)

				pygame.draw.rect(self.displaySurf, colour, wormRect)	
			eyeRects = [0,0]
			pupilRects = [0,0]
			eyeWhiteRects = [0,0]
			eyeInd = 0
			pupilColour = (min(worm.mutations*10, 255), 0, 0)
			for eye in worm.eyes:
				eyeRects[eyeInd] = pygame.Rect(eye[0]*self.grid.cell_size, eye[1]*self.grid.cell_size, self.grid.cell_size, self.grid.cell_size)
				eyeWhiteRects[eyeInd] = pygame.Rect(eye[0]*self.grid.cell_size+(self.grid.cell_size/5.0), eye[1]*self.grid.cell_size+(self.grid.cell_size/5.0), self.grid.cell_size-(self.grid.cell_size/3.), self.grid.cell_size-(self.grid.cell_size/3.))
				pupilRects[eyeInd] = pygame.Rect(eye[0]*self.grid.cell_size+(self.grid.cell_size/3.), eye[1]*self.grid.cell_size+(self.grid.cell_size/3.), self.grid.cell_size-2*(self.grid.cell_size/3.), self.grid.cell_size-2*(self.grid.cell_size/3.))
				pygame.draw.rect(self.displaySurf, colour, eyeRects[eyeInd])
				if random.random() < 0.003:
					worm.blink = 2
				if worm.pulse > 0.9 and worm.blink > 0:
					worm.blink -= 1
					
				if worm.blink <= 0:
					pygame.draw.rect(self.displaySurf, WHITE, eyeWhiteRects[eyeInd])
					pygame.draw.rect(self.displaySurf, pupilColour, pupilRects[eyeInd])
				
				eyeInd += 1
					

		for apple in self.apples:
			appleRect = pygame.Rect(apple[0]*self.grid.cell_size, apple[1]*self.grid.cell_size, self.grid.cell_size, self.grid.cell_size)
			pygame.draw.rect(self.displaySurf, RED, appleRect)
		
	def makeApples(self):

		applesNeeded = max(0, self.apple_count - len(self.apples)) 
		for a in range(applesNeeded):
			if self.apples == []:
				near = False
				how_near = False
			else:
				near = self.apples[-1]
				how_near = 10
			self.apples.append(self.grid.randomCell(near=near, how_near=how_near))
		for apple in self.apples:
			if random.random() < 0.01:
				c = random.randrange(2)
				d = random.choice((-1,1))
				apple[c] += d
			if apple[0] < 5:
				apple[0] = 5
			elif apple[0] > self.grid.cell_width-5:
				apple[0] = self.grid.cell_width-5
			if apple[1] < 0:
				apple[1] = 5
			elif apple[1] > self.grid.cell_height-5:
				apple[1] = self.grid.cell_height-5

	def populate(self):

		for i in range(self.population_size - len(self.population)):

			while 1:
				x, y = random.randrange(self.grid_size[0]), random.randrange(self.grid_size[1])
				goodtogo = True
				for i in range(self.starting_length):
					goodtogo = goodtogo and self.grid.occupying[x-i][y] == []
				if goodtogo:
					break
			# make the Genome
			genome = bitarray()
			for i in range(self.genome_size+1):
				genome.append(random.randint(0,1))
			newWorm = WormyPhenotype(genome=genome, x=x, y=y, length=self.starting_length, stamina=self.max_stamina, maturity=self.maturity, layers=self.layers, mutation_rate=self.mutation_rate)
			newWorm.name = (random.choice(self.name_list))
			self.population.append(newWorm)



			# for i in range(self.starting_length):
			# 	grid.occupying[x-i][y].append("WORM")

	def scentsAt(self, coords, direction):
		"""Returns the potency of worm odours and the potency of apple odours at a given cell, at the time that the function is called."""
		#wormScent = 0.0
		appleScent = 0.0
		N, E, S, W = self.compass


		# for worm in self.population:
		# 	for segment in worm.segments:
		# 		wormScent += self.grid.distance(segment, coords)**EXP
		totalScent = 0.0
		count = 0
		canSmell = False
		for apple in self.apples:
			

			if direction == N:
				if apple[1] <= coords[1] and abs(coords[0] - apple[0]) <= abs(coords[1] - apple[1]):
					canSmell = True
					totalScent += self.grid.distance(apple, coords)/self.grid.hypoteneuse
					#print direction, canSmell
			elif direction == E:
				if apple[0] >= coords[0] and abs(coords[0] - apple[0]) >= abs(coords[1] - apple[1]):
					canSmell = True
					totalScent += self.grid.distance(apple, coords)/self.grid.hypoteneuse
					#print direction, canSmell
			elif direction == S:
				if apple[1] >= coords[1] and abs(coords[0] - apple[0]) <= abs(coords[1] - apple[1]):
					canSmell = True
					totalScent += self.grid.distance(apple, coords)/self.grid.hypoteneuse
					#print direction, canSmell
			elif direction == W:
				if apple[0] <= coords[0] and abs(coords[0] - apple[0]) >= abs(coords[1] - apple[1]):
					canSmell = True
					totalScent += self.grid.distance(apple, coords)/self.grid.hypoteneuse
					#print direction, canSmell
		
		if len(self.apples)==0 or not canSmell:
			appleScent = 0.0
		else:
			appleScent = 1-(totalScent / len(self.apples))
			
		

		return appleScent


	def popRegulate(self):
		if len(self.population) > self.max_pop:
			self.apple_count = self.min_apples
		elif len(self.population) < self.min_pop:
			self.apple_count = self.max_apples
		else:
			self.apple_count = self.max_apples/2
		if len(self.population) <= 3:
			self.populate()


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

	def specimenData(self):
		
		specimenSurf = self.basicfont.render("%(name)s GEN: %(gen)s  SPAWN: %(kids)s  CON: %(con)s  SMELLS: %(odr)s  FOOD: %(food)s POP: %(pop)s" %{'name':(self.theSpecimen.name[:11]+":"+" "*9)[:12], 'gen': self.theSpecimen.gen, 'kids': self.theSpecimen.offspring, 'pop': len(self.population), 'odr':("("+(str(self.theSpecimen.odour[0])+" ")[:4]+", "+(str(self.theSpecimen.odour[1])+" ")[:4]+")"),'con': ("("+(str(self.theSpecimen.connectivity[0])+" ")[:4]+", "+(str(self.theSpecimen.connectivity[1])+" ")[:4]+")"), 'food':str(len(self.apples))+"/"+str(self.apple_count)}, True, (255,255,255))
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


	def terminate(self):
		pygame.quit()
		filename = raw_input("Save population as: ")
		self.savePopulation(filename)
		sys.exit()

	def runWormosphere(self):
		pygame.display.set_caption("NEUROWORMS")
		
		# initialize some variables
		spec = random.choice(self.population)
		spec.specimen = True
		self.theSpecimen = spec
		justDid = 0
		# some syntactic sugar:
		N, E, S, W = self.compass
		# i.e. [(0,-1),(1,0),(0,1),(-1,0)] represents the four directions. The idea being that N is a negative movement on the y axis, E is a positive movement on the x axis, etc... This is handy because you can reverse directions by multiplying any of these by (-1,-1). In fact:
		reverse = lambda d: (d[0] * -1, d[1] * -1)

		fertility = 0
		self.timer = 0
		#             R    G    B
		WHITE     = (255, 255, 255)
		BLACK     = (  0,   0,   0)
		RED       = (255,   0,   0)
		GREEN     = (  0, 255,   0)
		DARKGREEN = (  0, 155,   0)
		DARKGRAY  = ( 40,  40,  40)
		BGCOLOR   = BLACK
		
		while 1:


			
			self.timer += 1

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


				if pygame.mouse.get_pressed()[0] == True:

					mouse_x, mouse_y = pygame.mouse.get_pos()
					mouse_x, mouse_y = int(mouse_x / self.grid.cell_size), int(mouse_y / self.grid.cell_size)

					for w in self.population:
						if [mouse_x, mouse_y] in w.segments:
							
							self.theSpecimen.specimen = False
							self.theSpecimen = w
							w.specimen = True
							# print "\n NEW SPECIMEN \n"

							break
				if pygame.mouse.get_pressed()[1] == True:

					mouse_x, mouse_y = pygame.mouse.get_pos()
					mouse_x, mouse_y = int(mouse_x / self.grid.cell_size), int(mouse_y / self.grid.cell_size)

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



				if worm.moved:
					worm.stamina -= (worm.connectivity[0]+worm.connectivity[1])/2
				else:
					worm.stamina -= (worm.connectivity[0]+worm.connectivity[1])/6

				worm.odour = [self.scentsAt(worm.eyes[0],worm.direction), self.scentsAt(worm.eyes[1], worm.direction)]
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
				for w in self.population:
					if w != worm and nextStep in w.segments:
						worm.obstructed = True
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

				grow = False
				# did the worm eat an apple?
				for bite in [worm.segments[0], worm.eyes[0], worm.eyes[1]]:
					if bite in self.apples:
						self.apples.remove(bite)
						worm.stamina = worm.max_stamina
						grow = True
						worm.belly.append("a")

				# shrink end of worm
				if (not grow) and worm.moved: #and not collision:
					del worm.segments[-1]
					worm.moved = False

				# did the worm reach cloning size? Reproduction. 
				if len(worm.segments) >= worm.maturity and len(self.population) <= self.max_pop:
					breakpoint = worm.maturity/2
					child_segments = worm.segments[breakpoint:]
					del worm.segments[breakpoint:]
					child = WormyPhenotype(genome=worm.genome, segments=child_segments, maturity=worm.maturity)
					child.gen = worm.gen +1
					child.name = random.choice(self.name_list)
					child.mutations = worm.mutations
					child.mutate()
					worm.offspring += 1
					child.direction = reverse(worm.direction)
					child.newborn = 50
					self.population.append(child)

				# is the worm starving?
				if worm.stamina <= 0. and len(worm.segments) > 1:
					worm.stamina = worm.max_stamina
					del worm.segments[-1]

				# death
				if len(worm.segments) <= 1 and len(self.population) > self.min_pop/2: #failsafe
					self.population.remove(worm)

				# stress-induced mutations
				elif len(worm.segments) == 2 and worm.mutations < len(worm.belly)+3 :
					worm.mutation_rate += 20
					worm.mutate()

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

def main():
	music = False
	if music:
		pygame.mixer.init()
		pygame.mixer.music.load("../soundEffects/PixiesWhereIsMyMind8BitRemix.ogg")
		pygame.mixer.music.play(-1,0.0)
	W = Wormosphere(population_size=10, max_pop=35, min_pop=6, apple_count=2, max_stamina=50, mutation_rate=.30, starting_length= 6,maturity=16, layers=[5,5,2], cell=10)
	while 1:
		W.runWormosphere()


main()