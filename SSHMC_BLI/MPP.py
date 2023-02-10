"""
This code belongs to Semi-Supervised Hierarchical Multi-label Classifier Based on Local Information
	PGM_PyLib: https://github.com/jona2510/SSHMC-BLI/tree/master

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

from SSHMC_BLI.hStructure import hierarchy
import numpy as np
import copy

# A parent of B:
# Then the output for f+(A)  = f(A)
# and the output for f+(B) = min( f(A), f(B) )		# in the general case, the minimum of B and its parents.
class limitProb:

	"""
	Constructor
		baseClassifier: it is a hierarchical classifier [or a multilabel classifier] that has the function 'predict_proba'		
		TL : the partial_fit
	"""
	def __init__(self,H,baseClassifier,TL="from_scratch"):

		self.baseClassifier = copy.deepcopy(baseClassifier)#baseClassifier
		if( not (( "fit" in dir(self.baseClassifier) ) and ( "predict_proba" in dir(self.baseClassifier) )) ):
			raise NameError("ERROR: you has to provide a valid classifier (with fit and predict_proba functions)!")				
	
		# validate structure 	
		if(isinstance(H,hierarchy)):
			self.H = H
		else:
			raise NameError("Error!!, H has to be an hierarchy object")


	"""
		fit the classifier
	"""
	def fit(self, trainSet, cl):	
		self.baseClassifier.fit(trainSet, cl)

	"""
		partial_fit the classifier
	"""
	def partial_fit(self, trainSet, cl):	
		if( not ( "parttial_fit" in dir(self.baseClassifier) )  ):
			raise NameError("ERROR: the base classifier does not have the method: partial_fit!")				
		self.baseClassifier.partial_fit(trainSet, cl)

	def predict(self,testSet):
		print("WARNING: Method unavailable, returning -1")
		return -1


	"""
	return the probabilities of each instance being associated to the labels
		while guarantees that the probability of a node is not greater than the probability of its parent(s)
	"""
	def predict_proba(self,testSet):
		probs = self.baseClassifier.predict_proba(testSet)		
		
		return self._restrictChild(probs,True)


	# podria ser solo la funcion de limitar las predicciones
	"""
	Restricts the score of the child with respect to its parent(s)
	It only requieres: 
		probs: ndarray of shape(n_instances,m_labels)
		overwrite: if True then it overwrite the results in probs, else it creates a new matrix to return the values
	"""
	def _restrictChild(self, probs, overwrite=False):		
		sh = np.shape(probs)

		if(overwrite):
			res = probs
		else:
			res = probs.copy() #np.zeros(sh)
		# too much slow
		#for i in range(sh[0]):		# for each instance
		#	for x in self.H.iteratePF():	# iteration parents first
		#		if(x not in self.H.roots()):		# 
		#			val = min( res[i, H.getParents()[x] ] )
		#			if(res[i,x] > val):
		#				res[i,x] = val

		# faster than previous
		for x in self.H.iteratePF():	# iteration parents first
			#if(x not in self.H.roots()):		# 
			for z in self.H.getParents()[x]:
				cv = np.where( res[:,x] > res[:,z] )[0]
				res[cv,x] = res[cv,z]	#change the values by the values of its parents	

		return res
		
	"""
	Restricts the score of the parent with respect to its children
	It only requieres: 
		probs: ndarray of shape(n_instances,m_labels)
		overwrite: if True then it overwrite the results in probs, else it creates a new matrix to return the values
	"""
	def _restrictParent(self, probs, overwrite=False):		
		sh = np.shape(probs)

		if(overwrite):
			res = probs
		else:
			res = probs.copy() #np.zeros(sh)

		# fast 
		for x in self.H.iterateCF():	# iteration parents first
			#if(x not in self.H.roots()):		# 
			for z in self.H.getChildren()[x]:
				cv = np.where( res[:,x] > res[:,z] )[0]
				res[cv,x] = res[cv,z]	#change the values by the values of its parents	

		return res		














