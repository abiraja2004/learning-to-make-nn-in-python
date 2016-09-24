#	Back-Propagation of Classifier Neural Networks
#
#	Written in Python.  See http://www.python.org/
#	Placed in the public domain.
#	Ihor Menshykov <ihor.ibm@gmail.com>

import math
import random
import string
import numpy as np
np.set_printoptions(precision=5,	suppress=True)
#from pylab import *
#	<-- for orthonormal init

random.seed(0)

#	Orthonormal/Orthogonal initialisation
def svd_orthonormal(shape):
	if len(shape) < 2:
		raise RuntimeError("Only shapes of length 2 or more are supported.")
	flat_shape = (shape[0], np.prod(shape[1:]))
	# print "shape:",					shape
	# print "np.prod(shape[1:]):",	np.prod(shape[1:])
	# print "flat_shape:",			flat_shape
	#a = standard_normal(flat_shape)
	a = np.random.standard_normal(flat_shape)
	u, _, v = np.linalg.svd(a, full_matrices = False)
	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return q


# Calculate a random number between a and b
def rand(a, b):
	return a + (b-a)*random.random()

#	Kind of like usual objects in JavaScript
class Bunch:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

# You can expiriment with different activation functions
# and write your own easily

class activation:
	@staticmethod
	def act(read_numpy_array, type="relu"):
		numpy_array = np.copy(read_numpy_array)
		for x in np.nditer( numpy_array , op_flags=['readwrite'] ):
			#	Plain linear
			if		type=="" or type==0:
				# x[...]=	x
				continue

			#	Hyperbolic tangent
			elif	type=="tanh":
				x[...]=	math.tanh(x)

			elif	type=="relu":
				if x>0:
					# x[...]=	x
					continue
				else:
					x[...]=	0

			elif	type=="lrelu":
				if x>0:
					# x[...]=	x
					continue
				else:
					x[...]=	x*0.1

			elif	type=="necroRelu":
				if x>0:
					# x[...]=	x
					continue
				else:
					x[...]=	0

			elif	type=="abs":
				if x>0:
					# x[...]=	x
					continue
				else:
					x[...]=	-x

			# no luck with this one
			elif	type=="wierdAbs":
				if		x> 1:
					# x[...]=	x
					continue
				elif	x>-1:
					x[...]=	0
				else:
					x[...]=	-x

			# no luck with this one
			elif	type=="sin":
				x[...]=	math.sin(x)

			# no luck with this one
			elif	type=="custom1":
				x[...]=	1 / ( x*x + 2 )

		return numpy_array


	@staticmethod
	def error_activation(read_numpy_array, type="relu"):
		#	"Derivatives" don't actually have to be mathematically correct derivatives
		numpy_array = np.copy(read_numpy_array)
		for x in np.nditer( numpy_array , op_flags=['readwrite'] ):
			#	Plain linear
			if	type=="" or type==0:
				x[...]=	1

			elif	type=="tanh":
				x[...]=	1 - math.tanh(x)**2
				#	also works, but isn't a correct derivative:
				# return	1 - x**2
				#	fact that such things work is an interesting topic

			elif	type=="relu":
				if x>0:
					x[...]=	1
				else:
					x[...]=	0

			elif	type=="lrelu":
				if x>0:
					x[...]=	1
				else:
					x[...]=	0.1

			elif	type=="necroRelu":
				if x>0:
					x[...]=	1
				else:
					x[...]=	0.1

			elif	type=="abs":
				if x>0:
					x[...]=	1
				else:
					x[...]=	-1

			elif	type=="wierdAbs":
				if		x> 1:
					x[...]=	1
				elif	x>-1:
					x[...]=	0
				else:
					x[...]=	-1

			elif	type=="sin":
				x[...]=	math.cos(x)

			elif	type=="custom1":
				x[...]=	-2*x / ( ( x*x+2 )**2 )

		return numpy_array


# activation	=	activation()



class ClassifierNet:

	def __init__(self, layers):
		self.layers	=	layers
		#	parameter [layers] consists of
		#		arrays:
		#			sizes_of_hidden
		#			act			--	activation layer types
		#		integers:
		#			n_inputs	--	number of input neurons
		#			n_outputs	--	number of possible classification outputs
		#		mixed string/0
		#			final_act	--	function for final layer activation

		self.layers.sizes				=		[ self.layers.n_inputs ]		\
											+	  self.layers.sizes_of_hidden	\
											+	[ self.layers.n_outputs ]

		#	We don't want to distort ("activate") input values
		self.layers.act					=		[0]								\
											+	  self.layers.act				\
											+	[ self.layers.final_act ]		\

		self.layers.num_of_layers		=	len(	self.layers.sizes )	#includes ins and outs

		self.layers.neuron_activations	=	[0] *	self.layers.num_of_layers
		self.layers.neuron_biases		=	[0] *	self.layers.num_of_layers
		#	<- input layer doesn't have biases, but anyway..

		self.layers.weights_from_to		=	[0] * (	self.layers.num_of_layers -1 )	#	output layer doesn't have any

		self.layers.error				=	[0] * 	self.layers.num_of_layers
		self.layers.targets				=	[0] * 	self.layers.num_of_layers


		for	current_layer, current_layer_size	in	enumerate(self.layers.sizes):
			# print	current_layer
			# print	"current_layer_size:", current_layer_size

			self.layers.neuron_activations	[ current_layer ]	=	\
				np.zeros(		current_layer_size )

			if	current_layer	>	0	:
				#	Biases are only for non-input layers
				self.layers.neuron_biases	[ current_layer ]	=	\
					np.zeros(	current_layer_size )


			if	current_layer	!=	len( self.layers.sizes ) -1	:
				#	If it ain't the last layer, it should have some weights
				next_layer_size	=	self.layers.sizes[ current_layer+1 ]
				# print	'next_layer_size:', next_layer_size

				#	Initialising weights with orthonormal vectors (a cool maths-based trick).
				#	Better than pure random, or random from normal distribution.
				#	"weights_from_to", because weights_from_to[layer][from_neuron][to_neuron]
				self.layers.weights_from_to[ current_layer ]	=	\
					svd_orthonormal( ( current_layer_size, next_layer_size ) )

				# print	"weights_from_to shape:"
				# print							self.layers.weights_from_to[ current_layer ].shape


		# print	"\n"
		# print	"neuron_activations:"
		# print							self.layers.neuron_activations
		# print	"\n"
		# print	"neuron_biases:"
		# print							self.layers.neuron_biases
		# print	"\n"
		print	"INITIALIZATION DONE"
		#	we're not going to use/calculate any momentum for this example



	def forward(self, inputs):

		if len(inputs) != len( self.layers.neuron_activations[0] ):
			raise ValueError('FORWARD method SAYS:	wrong number of inputs')

		# input activations
		# for i in range(self.input_neurons-1):	# bias always 1, we keep it as is
		# 	self.input_neuron_activations[i] = inputs[i]
		self.layers.neuron_activations[0] =	np.array( inputs )

		for	current_layer	in	xrange(	self.layers.num_of_layers -1 ):
			self.layers.neuron_activations[ current_layer+1 ] =							\
				activation.act															\
					(																	\
						np.dot															\
							(															\
								self.layers.neuron_activations	[ current_layer   ]		\
							,	self.layers.weights_from_to		[ current_layer   ]		\
							)															\
						+																\
						self.layers.neuron_biases				[ current_layer+1 ]		\
					,																	\
						type=	self.layers.act					[ current_layer+1 ]		\
					)


	def backward(self, target, mode="each", learning_rate=0.01,	weight_decay=0):

		if target > self.layers.n_outputs	:
			raise ValueError('BACKWARD method SAYS:	target should be <= # of possible outputs')

		if target < 1	:
			raise ValueError('BACKWARD method SAYS:	target should be > 0')

		target	-=	1
		#	We pass as		1-based		--- "human-friendly" way,
		#	We process as	0-based		--- "computer-friendly" way


		#	Setting learning targets

		if		mode	==	"each"	:
			#	Calculate resulting activity error for each output neuron
			self.layers.targets[ -1 ]			=	np.zeros( self.layers.n_outputs )
			self.layers.targets[ -1 ][ target ]	=	1.
			#	In Python
			#	[-1]	selects the  last element of an array
			#	[-1:]	selects from last element of an array
			#			(select starting from last index and ending at the end)

		elif	mode	==	"max and target"	:
			#	Only fiddle with the layer that outputs largest value and target value
			#	and only if the net didn't make a correct guess
			if	np.argmax( self.layers.neuron_activations[ -1 ] )	==	target	:
				# print	"Correct guess! Lets keep it that way."
				return	"no change"

			#	implicitly, else:
			self.layers.targets							[ -1 ]			=			\
				np.copy(self.layers.neuron_activations	[ -1 ])

			self.layers.targets							[ -1 ]						\
				[																	\
				np.argmax(self.layers.neuron_activations[ -1 ])						\
				]														=	0.

			self.layers.targets							[ -1 ][ target ]=	1.


		self.layers.error								[ -1 ]			=			\
			activation.error_activation												\
					(																\
						self.layers.neuron_activations	[ -1 ]						\
					,	type= self.layers.act			[ -1 ]						\
					)																\
				*	(																\
						self.layers.targets				[ -1 ]						\
					-	self.layers.neuron_activations	[ -1 ]						\
					)


		#	Beginning at the pre-last layer, backpropagate errors
		#	We have a more complicated syntax in this example,
		#	but we're doing the same thing as in the previous.
		for		i_layer		in	reversed(xrange( self.layers.num_of_layers-1 ))		:
			self.layers.error[ i_layer ] =											\
					activation.error_activation										\
						(															\
							self.layers.neuron_activations	[ i_layer   ]			\
						,	type= self.layers.act			[ i_layer   ]			\
						)															\
				*	np.dot															\
						(															\
							self.layers.error				[ i_layer+1 ]			\
						,	self.layers.weights_from_to		[ i_layer   ].T			\
						)
			#	We used .T to transpose matrix weights_from_to
			#	-- turn weights[from][to] into weights[to][from] instead
			#	With dot product we've calculated [from] amount of sums of differences
			#	back-propagated from each of next layer's neuron to each in the current layer
			#	And we've * derivative of current activation type.
			#	Derivatives make us tweak neurons, activation values of which
			#	tend to change faster by more
			#	than those, slightly changing which wouldn't result in such a big change.
			#	Also, this provides us with a correct direction of update
			#	--	think about "abs" activation function.
			#	"Derivatives" --- error_activation functions --- don't actually have to be
			#	completely mathematically correct derivatives
			#	This and weight multiplication means we provide more influence
			#	to those neurons, which we have to alter the least
			#	That way we maximize
			#	[Resulting positive change] / [Required pre-activation neuron value tweaking]

		# print	"pre-error activations:"
		# print					self.layers.neuron_activations[-1]
		# print	"pre-error targets:"
		# print					self.layers.targets[-1]

		# print	"error:"
		# print					self.layers.error[-1]
		# print	"\n"
		# print	"sum error squared:"
		# print					np.sum(self.layers.error[-1] * self.layers.error[-1])
		# print	"\n"


		#	In order to actually tweak any activation values we have to change weights.
		#	The more activation change we need, the more we tweak the incoming weights.
		#	Another heuristic is that the higher the neuron's pre-activation value
		#	the more we tweak the weights coming from it.
		#	That way we maximize
		#	[Resulting positive change] / [Required weight tweaking]
		#
		#	All but the last layer have weights. We want to tweak those.
		for		i_layer		in	xrange( self.layers.num_of_layers-1 )		:
			weight_addition	=														\
					np.array_split													\
						(															\
							self.layers.neuron_activations		[ i_layer	]		\
						,	len	(													\
 								self.layers.neuron_activations	[ i_layer	]		\
								)													\
						)															\
				*	self.layers.error							[ i_layer+1	]		\
				*	learning_rate

			# weight_decay=0.000001
			weight_addition		-=														\
				weight_decay	*	weight_addition * weight_addition * np.sign( weight_addition )
				# weight_decay	*	weight_addition * weight_addition * weight_addition

			self.layers.weights_from_to							[ i_layer	]	+=	\
				weight_addition

			self.layers.neuron_biases	[ i_layer+1	]	+=		\
					self.layers.error	[ i_layer+1	]			\
				*	learning_rate

			#	We tweak weights by
 			#	learning rate *
			#	amount of change we want *
			#	amount of activation we have from neuron from which weight origins



	def tell_result(self, pattern):

		self.forward( pattern.i )

		# print	"\n"
		# print	"neuron_activations:"
		# print					self.layers.neuron_activations[-1]
		#
		# print	"max_act:"
		# print		np.argmax(	self.layers.neuron_activations[-1]	) +1
		# print	"target:"
		# print					pattern.o

		if	np.argmax(	self.layers.neuron_activations[-1]	) +1	==	pattern.o	:
			# print	'Correct',		pattern.o
			return 1
		else:
			# print	'Incorrect, need ',	pattern.o, 'got', np.argmax(	self.layers.neuron_activations[-1]	) +1
			return 0



	def tell_weights(self):
		print	"weights_from_to:"
		print					self.layers.weights_from_to[0]
		print	"\n"
		print					self.layers.weights_from_to[1]
		print	"\n"







import time

if __name__ == "__main__":

	pat = 	[
				Bunch(	i= [0,0],	o= 1	)
			,	Bunch(	i= [0,1],	o= 2	)
			,	Bunch(	i= [1,0],	o= 2	)
			,	Bunch(	i= [1,1],	o= 1	)
			,	Bunch(	i= [2,2],	o= 3	)
			]

	for		samples_per_output	in	xrange(10):
		for		out				in	xrange(20):
			pat.append											\
				(												\
				Bunch(	i=	np.random.randn(2)	,	o=out+1	)	\
				)


	cn	=											\
		ClassifierNet								\
			(										\
			Bunch(									\
					sizes_of_hidden	= [32,32]			\
				,	act				= ["lrelu","lrelu"]		\
				,	n_inputs		= 2				\
				,	n_outputs		= 20			\
				,	final_act		= 0				\
				)									\
			)
	# cn.forward(	pat[0].i )

	# cn.tell_result()
	# cn.tell_weights()

	t0 = time.clock()
	cn.forward(	pat[0].i )
	print time.clock() - t0, "seconds to forward"

	t0 = time.clock()
	cn.backward(	pat[0].o )
	print time.clock() - t0, "seconds to backprop"


	for	superepoch		in	xrange(5):
		for	i		in	xrange(100):
			for	p	in	xrange( len(pat) ):
			# for	p	in	xrange( 1 ):
				cn.forward(	pat[p].i )
				# cn.backward(	pat[p].o ,	mode	=	"max and target")
				cn.backward(	pat[p].o,	weight_decay=0.01)

			if	i%250	==	0	:
				print	i


		count	=	0
		Of		=	0
		for	p	in	xrange( len(pat) ):
			count	+=	cn.tell_result(	pat[p] )
			Of		+=	1
		# cn.tell_weights()c
		print	"\n"
		print	"Correct:"
		print	count
		print	"of"
		print	Of
		print	"\n"




	# cn.forward(	pat[0].i )
	# cn.backward(	pat[0].o ,	mode	=	"max and target")

	# cn.tell_result()

	# train it with some patterns
	# for i in range(20):
	# 	n.train(pat)
	# 	# test it
	# 	n.test(pat)
