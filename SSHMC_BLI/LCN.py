"""
This code belongs to Semi-Supervised Hierarchical Multi-label Classifier Based on Local Information
	PGM_PyLib: https://github.com/jona2510/SSHMC-BLI/tree/master

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np 
import copy
#from collections import deque
from SSHMC_BLI.hStructure import hierarchy
from SSHMC_BLI.hStructure import policiesLCN as pol
#from PGM_PyLib.naiveBayes import naiveBayes as nb
from sklearn.ensemble import RandomForestClassifier as rfc
from SSHMC_BLI.evaluationHC import MCC

"""
(L)ocal (C)lassifier per (N)ode aproach
Trains a binary classifier for each node (except the root node)
"""
class LCN:

	"""
	constructor of LCN
	H: hierarchy object, contains the hierarchical structure
	baseClassifier: base classifier to be trained for each node, default, NaiveBayes
	policy: policy to select the positive and negative instances at each node, default None (siblings policy)
	"""
	def __init__(self,H,baseClassifier=rfc(),policy=None,TL="from_scratch"):

		self.baseClassifier = baseClassifier	# base classifier, it will be copied for each class		

		# validate structure 	
		if(isinstance(H,hierarchy)):
			self.H = H
		else:
			raise NameError("Error!!, H has to be an hierarchy object")

		if(policy is None):
			self.policy = pol(H)	# default parameters (balanced=False,policy="siblings",seed=0)
		elif( "getInstances" in dir(policy) ):	
			self.policy = policy			# policy to select instances positives and negatives to train a node/label
		else:
			raise NameError("Error!!, policy does not have the requiered method 'getInstances'")

		self.H.initialize()
		self.n = self.H.n #shs[0]

		# Variables estimated by the method. These HAVE NOT to be modified except by the classifier itself
		self.isfit = False
		self.dCl = np.empty(self.n, dtype = object)		#contains the classifiers

		# variables useful for retrain a classifier
		#self.first = True
		self.retrain = False
		self.TL = TL


	# train: is a matrix (n x m)  where n is the number of instances and m the number of classes
	# cl: is a matrix (n x l)  where n is the number of instances and l the number of classes
	def fit(self, trainSet, cl):	
		# analize Process in package multiprocessing
		# To lanch multiple process and fit each classifier on them

		shTr = np.shape(trainSet)	# shape of the train set
		if(len(shTr) != 2):
			raise NameError("The train set has to be a two-dimension numpy matrix (intances x attributtes)")			

		shCl = np.shape(cl)	# shape of the classes to which the instances are associated
		if(len(shCl) != 2):
			raise NameError("The classes has to be a two-dimension numpy matrix (intances x classes)")			

		if(shTr[0] != shCl[0]):
			raise NameError("The number of elements in data is different from classes (cl)")	

		if( not (( "fit" in dir(self.baseClassifier) ) and ( "predict" in dir(self.baseClassifier) )) ):
			raise NameError("ERROR: you has to provide a valid classifier!")				

		# Begin the training
		for i in range(self.n):
			#copy the base classifier
			#print("train ",i," classifier")			
			self.dCl[i] = copy.deepcopy(self.baseClassifier)
			#dLocal, clLocal = self.policy.getInstances(i,trainSet,cl)
			#print("classes local: ",i,"\n",clLocal)		
			#self.dCl[i].fit( dLocal, clLocal )

			ind = self.policy.getInstances(i,trainSet,cl)
			#print("ind: \n",ind)
			self.dCl[i].fit( trainSet[ind], cl[ind,i] )
			#print("classes_: ",self.dCl[i].classes_)
		self.isfit = True


	def predict_proba(self, testSet):	
		self.checkIfFit()	# first check if the classifier is already trained	
		if( "predict_proba" not in dir(self.baseClassifier) ):
			raise NameError("ERROR: the provided base classifier does NOT have the method 'predict_proba'!")
		
		# check if testSet has the correct dimensions
		#pr_prob = [ ] # [] for x in range(len(self.structure)) ]
		pr_prob = np.zeros((len(testSet),self.n))
		# Begin the prediction
		for x in range(self.n):			
			#print("local prediction: ",x)
			#print("len test: ", len(testSet))
			# obtain the probabilities for every node
			#pr_prob[x] = self.dCl[x].predict_proba( testSet )
			p = self.dCl[x].predict_proba( testSet )
			#print("p shape: ",np.shape(p))
			#print("p: \n",p)
			#exit()
			#pr_prob.append( p[:, np.where( self.dCl[x].classes_ == 1 ) ] )	# save only the positive probabilities, negatives are (1-positives)
			#print("p+\n",p[:, np.where( self.dCl[x].classes_ == 1 )[0] ])
			#print("p+ reshape\n",p[:, np.where( self.dCl[x].classes_ == 1 )[0] ].reshape(1,-1))
			#print("np.where: ",np.where( self.dCl[x].classes_ == 1 )[0])
			pcl = np.where( self.dCl[x].classes_ == 1 )[0]	# position of the positive class in classes_
			if(len(pcl)<1):	# if there is no positive class, it is set the probabilities to 0
				pr_prob[:,x] = 0
			else:
				pr_prob[:,x] = p[:, pcl[0] ]#.reshape(1,-1)
		#return predictions
		#return np.column_stack(pr_prob)
		#print("pr_prob:")
		#print(pr_prob)
		return pr_prob

	def predict(self, testSet):
		"""
		return the individual predictions of each classifier
		does NOT consider the hierarchy
		"""
		print("Call predict independent LCN")
		self.checkIfFit()	# first check if the classifier is already trained	
		if( "predict" not in dir(self.baseClassifier) ):
			raise NameError("ERROR: the provided base classifier does NOT have the method 'predict'!")
		
		# check if testSet has the correct dimensions

		pred = np.zeros((len(testSet),self.n),dtype=bool)
		# Begin the prediction
		for x in range(self.n):
			#p = self.dCl[x].predict(testSet)
			pred[:,x] = self.dCl[x].predict(testSet) 	#p[:, np.where( self.dCl[x].classes_ == 1 )[0] ].reshape(1,-1)
		return pred

	def predictIndependent(self, testSet):			
		"""
		return the individual predictions of each classifier
		does NOT consider the hierarchy
		"""
		print("Call independent")
		self.checkIfFit()	# first check if the classifier is already trained	
		if( "predict" not in dir(self.baseClassifier) ):
			raise NameError("ERROR: the provided base classifier does NOT have the method 'predict'!")
		
		# check if testSet has the correct dimensions

		pred = np.zeros((len(testSet),self.n),dtype=bool)
		# Begin the prediction
		for x in range(self.n):
			#p = self.dCl[x].predict(testSet)
			pred[:,x] = self.dCl[x].predict(testSet) 	#p[:, np.where( self.dCl[x].classes_ == 1 )[0] ].reshape(1,-1)
		return pred


	"""
	Check is the classifiers is already trained, 
	if not, then raises a exeption
	"""
	def checkIfFit(self):
		if(not self.isfit):
			raise NameError("Error!: First you have to train ('fit') the classifier!!")


			

class TopDown(LCN):

	#keeps the same constructur than LCN

	#keeps the same method fit than LCN

	"""
	This function follows faithfully the Top-Down prediction, nevetheless the individuals predictions make it too slow
	"""
	def predictLowMemory(self, testSet):
		self.checkIfFit()	# first check if the classifier is already trained
		# check if testSet has the correct dimensions
		shte = np.shape( testSet )

		pr = np.zeros((shte[0],self.n),dtype=int) #	[ [] for x in range(len(self.structure)) ]

		pos_cl = np.zeros(self.n,dtype=int)		# position of 'positive' in classes_
		for i in range(self.n):
			pos_cl[i]=np.where( self.dCl[i].classes_ == 1 )[0]

		# Begin the prediction
		# this could be parall, may be, a copy of all classifier will be required for each process
		for i in range(shte[0]):	# for each intance
			#print("p: ",i)
			children = self.H.getRoots()	 #self.H.getChildren()[nl]
			nl = -1	# node with maximmum probability
			while(len(children) > 0):
				pmax = -np.inf	# maximum probability	( restart at each level of hierarchy)
				for x in children:
					#p = self.dCl[x].predict_proba( np.array([testSet[i]]) )
					p = self.dCl[x].predict_proba( testSet[i].reshape(1,-1) ) 
					p = p[0,pos_cl[x] ]	# get the probabilities and store the psotive probability
					if(p > pmax):	
						pmax = p
						nl = x
				children = self.H.getChildren()[nl]	# advance towards the node with highest probability
			pr[i] = self.H.getSinglePaths()[nl].copy()
			
		return pr

	def predict(self, testSet):
		print("Call predicti TD")
		self.checkIfFit()	# first check if the classifier is already trained

		# check if testSet has the correct dimensions

		shte = np.shape( testSet )

		pr = np.zeros((shte[0],self.n),dtype=int) #	[ [] for x in range(len(self.structure)) ]

		#pos_cl = np.zeros(self.n,dtype=int)		# position of 'positive' in classes_
		#for i in range(self.n):
		#	pos_cl[i]=np.where( self.dCl[i].classes_ == 1 )[0]

		lpr = self.predict_proba(testSet)		# get all the probabilities of each instance for each node

		# Begin the prediction
		# this could be parall, may be, a copy of all classifier will be required for each process
		for i in range(shte[0]):	# for each intance
			#print("p: ",i)
			children = self.H.getRoots()	 #self.H.getChildren()[nl]
			nl = -1	# node with maximmum probability
			while(len(children) > 0):
				pmax = -np.inf	# maximum probability	( restart at each level of hierarchy)
				for x in children:
					##p = self.dCl[x].predict_proba( np.array([testSet[i]]) )
					#p = self.dCl[x].predict_proba( testSet[i].reshape(1,-1) ) 
					p = lpr[i,x]	#p[0,pos_cl[x] ]	# get the probabilities and store the psotive probability
					if(p > pmax):	
						pmax = p
						nl = x
				children = self.H.getChildren()[nl]	# advance towards the node with highest probability
			pr[i] = self.H.getSinglePaths()[nl].copy()
			
		return pr



