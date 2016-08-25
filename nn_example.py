# Back-Propagation of Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
# Ihor Menshykov <ihor.ibm@gmail.com>

import math
import random
import string

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
	return a + (b-a)*random.random()

# Make a matrix
# we could use NumPy to speed this up,
# but we want things done as simple as possible
def makeMatrix(I, J, fill=0.0):
	m = []
	for i in range(I):
		m.append([fill]*J)
	return m

# You can expiriment with different activation functions
# and write your own easily
def activationFunction(x, type="relu"):
	#	hyperbolic tangent
	if	type=="tanh":
		return	math.tanh(x)

	if	type=="relu":
		if x>0:
			return x
		else:
			return 0

	#	Derivative isn't the actual derivative for this one,
	#	but it can sometimes work better than if it was
	if	type=="necroRelu":
		if x>0:
			return x
		else:
			return 0

	if	type=="abs":
		if x>0:
			return x
		else:
			return -x

	if	type=="lrelu":
		if x>0:
			return x
		else:
			return x*0.05

	# no luck with this one
	if	type=="wierdAbs":
		if		x> 1:
			return	x
		elif	x>-1:
			return	0
		else:
			return	-x

	# no luck with this one
	if	type=="sin":
		return math.sin(x)

	# no luck with this one
	if	type=="custom1":
		return 1 / ( x*x + 2 )


def activationDerivative(x, type="relu"):
	if	type=="tanh":
		return	1 - math.tanh(x)**2
		#	also works, but isn't a correct derivative:
		# return	1 - x**2
		#	fact that such things work is an interesting topic

	if	type=="relu":
		if x>0:
			return 1
		else:
			return 0

	if	type=="necroRelu":
		if x>0:
			return 1
		else:
			return 0.05

	if	type=="abs":
		if x>0:
			return 1
		else:
			return -1

	if	type=="lrelu":
		if x>0:
			return 1
		else:
			return 0.05

	if	type=="wierdAbs":
		if		x> 1:
			return	1
		elif	x>-1:
			return	0
		else:
			return	-1

	if	type=="sin":
		return math.cos(x)

	if	type=="custom1":
		return -2*x / ( ( x*x+2 )**2 )



class NN:
	def __init__(self, ni, nh, no):
		self.input_neurons	= ni + 1 # +1 for bias node
		self.hidden_neurons	= nh
		self.output_neurons	= no

		self.input_neuron_activations	= [1.0]*self.input_neurons
		self.hidden_neuron_activations	= [1.0]*self.hidden_neurons
		self.output_neuron_activations	= [1.0]*self.output_neurons

		self.weights_from_input		= makeMatrix(self.input_neurons,	self.hidden_neurons)
		self.weights_from_hidden	= makeMatrix(self.hidden_neurons,	self.output_neurons)
		# set them to random vaules
		for i in range(self.input_neurons):
			for j in range(self.hidden_neurons):
				self.weights_from_input[i][j]	= rand(-.01, .01)

		for j in range(self.hidden_neurons):
			for k in range(self.output_neurons):
				self.weights_from_hidden[j][k]	= rand(-.01, .01)

		# last change in weights for momentum
		self.momentum_input		= makeMatrix(self.input_neurons,	self.hidden_neurons)
		self.momentum_hidden	= makeMatrix(self.hidden_neurons,	self.output_neurons)


	def forwardPropagate(self, inputs):
		if len(inputs) != self.input_neurons-1:
			raise ValueError('wrong number of inputs')

		# input activations
		for i in range(self.input_neurons-1):	# bias always 1, we keep it as is
			self.input_neuron_activations[i] = inputs[i]

		# hidden activations
		for j in range(self.hidden_neurons):
			sum = 0.0
			for i in range(self.input_neurons):
				sum += self.input_neuron_activations[i] * self.weights_from_input[i][j]
			self.hidden_neuron_activations[j] = activationFunction(sum, "relu")

		# output activations
		for k in range(self.output_neurons):
			sum = 0.0
			for j in range(self.hidden_neurons):
				sum += self.hidden_neuron_activations[j] * self.weights_from_hidden[j][k]
			self.output_neuron_activations[k] = sum
			#	We don't want to constrain our outputs to activation-processed values,
			#	Because there's just no reason to anyway.
			#
			#	People currently often use SoftMax, or other activation of output layer when they are using
			#	NN to classify things as one OR the other.
			#	In such function each output
			#	depends on raw values of all the other outputs (-> "classification probabilities") too, not just one.
			#	I feel like things might get simpler as people will discover SoftMax usage
			#	to be as "funny" as such overcomplicated activations as archaic Sigmoid and Tanh were.
			#	This general overcomplicatedness is a side-effect of field being driven by too many PhDs
			#	and others taking them too dogmatically.
			#	I might be the one wrong, of course, but that's the beauty of doubt. (^.^)
			#	One good reason to use SoftMax might be that the learning would conventrate
			#	more on net actually telling which result is the best,
			#	rather than on forcing all of the others to stay at 0.
			#	If you have many things to classify, if you use a plain linear architecture,
			#	the net might decide that the least wrong it could get is by always telling
			#	there's a 0 probability that you the input is _any_ sort of classification result
			#	One other way to overcome this could, probably, be using plain max activation.
			#	That is, only calculating difference and doing backprop
			#	if maximum linear value of output wasn't the one of	target-neuron
			#	and if so, only calculating change of this wrongfully-maximum result (should have been 0.)
			#	and the correct result (should have been 1.).
			#	This should probably cut the noize from all of the other neurons
			#	vs the plain way - calculating change for every neuron from the last layer.
			#
			#	Let's explore this concept in ClassifierNet

		return self.output_neuron_activations[:]


	def backPropagate(self, targets, N, M):
		if len(targets) != self.output_neurons:
			raise ValueError('wrong number of target values')

		# calculate error terms for output
		output_deltas = [0.0] * self.output_neurons
		for k in range(self.output_neurons):
			output_deltas[k] = (targets[k] - self.output_neuron_activations[k])
			#  current derivative is 1, because we have a linear (none) activation here
			#
			#  derivative(=1) * difference
			# (derivative = gradient at that activation point)
			# (
			#   ?= how far we would have to move pre-activation value
			#	  in order to get target result
			#  --- but shouldn't we then DIVIDE, not MULTIPLY by activation derivative?
			# )

		# calculate error terms for hidden
		hidden_deltas = [0.0] * self.hidden_neurons
		for j in range(self.hidden_neurons):
			error = 0.
			for k in range(self.output_neurons):
				error += output_deltas[k] * self.weights_from_hidden[j][k]
				# (difference * derivative) of neuron * our weight towards it

				# ?=   how far we should move that value
				#	* abs(how important it is)
				#	* sign(us growing =?= result growing)

			hidden_deltas[j] = activationDerivative(self.hidden_neuron_activations[j], "relu") * error
			#   sum(difference * derivative of neurons) * our weight towards them
			# * derivative

			# (
			#   ?= how far we would have to move pre-activation value
			#	  in order to get target result
			#  --- but shouldn't we then DIVIDE, not MULTIPLY by activation derivative?
			# )

			#print error

		# update output weights
		for From	in range(self.hidden_neurons):
			for To	in range(self.output_neurons):
				change = output_deltas[To]*self.hidden_neuron_activations[From]
				self.weights_from_hidden[From][To]+= N*change + M*self.momentum_hidden[From][To]
				self.momentum_hidden[From][To] = change
				#print N*change, M*self.momentum_hidden[j][k]

		# update input weights
		for From	in range(self.input_neurons):
			for To	in range(self.hidden_neurons):
				change = hidden_deltas[To]*self.input_neuron_activations[From]
				self.weights_from_input[From][To]+= N*change + M*self.momentum_input[From][To]
				self.momentum_input[From][To] = change

		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5*(targets[k]-self.output_neuron_activations[k])**2
		return error


	def test(self, patterns):
		for p in patterns:
			print(p[0], '->', self.forwardPropagate(p[0]))


	def weights(self):
		print('Input weights:')
		for i in range(self.input_neurons):
			print(self.weights_from_input[i])
		print()
		print('Output weights:')
		for j in range(self.hidden_neurons):
			print(self.weights_from_hidden[j])


	def train(self, patterns, epochs=500, N=0.005, M=0.):
		# N: learning rate
		# M: momentum factor
		for i in range(epochs):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.forwardPropagate(inputs)
				error = error + self.backPropagate(targets, N, M)
			if i % 1000 == 0:
				print('error %-.5f' % error)


def demo():
	# Lets teach our network XOR function and something more
	pat = [
		[[0,0], [0,	1	,	10,	-0.9]],
		[[0,1], [1,	1	,	20,	-0.8]],
		[[1,0], [1,	1	,	30,	-0.7]],
		[[1,1], [0,	-1	,	40,	-0.6]]
	]

	# create a network with two input, two hidden, and one output nodes
	n = NN(2, 7, 4)
	# train it with some patterns
	for i in range(20):
		n.train(pat)
		# test it
		n.test(pat)

if __name__ == '__main__':
	demo()
