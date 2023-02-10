"""
This code belongs to Semi-Supervised Hierarchical Multi-label Classifier Based on Local Information
	PGM_PyLib: https://github.com/jona2510/SSHMC-BLI/tree/master

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np 
import copy
#from PGM_PyLib.hierarchicalClassification.LCN import TopDown as td
#from SSHMC_BLI.LCN import TopDownMPP as tdm
from SSHMC_BLI.LCN import LCN
from SSHMC_BLI.hStructure import hierarchy
from SSHMC_BLI.hStructure import policiesLCN as pol
#import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree as bt
from sklearn.neighbors import DistanceMetric as dm
from sklearn.semi_supervised import SelfTrainingClassifier as stc
#from PGM_PyLib.naiveBayes import naiveBayes as nb
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.cluster import OPTICS 
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier as rfc
from SSHMC_BLI.LCN import TopDown as td
#from hStructure import policiesLCN as pol
from SSHMC_BLI.utils import SISI
from sklearn.neighbors import NearestNeighbors as NN
from math import pi
"""
(S)emi (S)upervised (L)ocal (C)lassifier per (N)ode aproach 
Trains a 'hierarchical' classifier which make use of LCN
"""
class SS_LCN:

	"""
	constructor of LCN
	H: hierarchy object, contains the hierarchical structure
	baseClassifier: base classifier to be trained for each node, default, NaiveBayes
	policy: policy to select the positive and negative instances at each node, default None (siblings policy)
	"""
	#def __init__(self):#,H,Hclassifier=td(),policy=None,classifiers_strategy="from_scratch",unlabeled_strategy="all",strategy_bestU="",threshold=0.7):
	def __init__(self,H,Hclassifier=None,policy=None,classifiers_strategy="from_scratch",unlabeled_strategy="all",strategy_bestU="naiveThreshold",threshold=0.7,minUdata=10,maxIterations=-1,seed=0):

		self.classifiers_strategy = classifiers_strategy
		self.unlabeled_strategy = unlabeled_strategy
		self.strategy_bestU = strategy_bestU
		self.threshold = threshold
		self.minUdata = minUdata	# minimum unalbelled data to continue the iteration process, lower or equal than 0 until all unlabeled data is pseudo-labeled 
		self.maxIterations = maxIterations	# -1: without limit
		self.seed = seed


		# validate structure 	
		if(isinstance(H,hierarchy)):
			self.H = H
		else:
			raise NameError("Error!!, H has to be an hierarchy object")

		if(Hclassifier is None):	# if the hclassifier is not provided, generate a TD
			self.Hclassifier = td(H)	# instanciate TD with default parameter: baseClassifier=nb(),policy=None,TL="from_scratch"
		else:
			self.Hclassifier = Hclassifier	# base classifier, it will be used for the semi-supervised learning

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
		#self.dCl = np.empty(self.n, dtype = object)		#contains the classifiers		
		self.HC = None		#the hierarchical classifier
		self.unIadded = np.zeros(self.n,dtype=int)		# number of unlabeled instances added to each node


	




	
	def predict(self, testSet):
		return self.HC.predict(testSet)

	def predict_proba(self, testSet):
		return self.HC.predict_proba(testSet)


class SSHMCBLI(SS_LCN):
	"""
	(S)emi-(S)upervised (H)ierarchical (M)ulti-label (C)lassifier (B)ased on (L)ocal (I)nformation (SSHMC-BLI)
	"""

	"""
	constructor of LCN
	H: hierarchy object, contains the hierarchical structure
	baseClassifier: base classifier to be trained for each node, default, NaiveBayes
	policy: policy to select the positive and negative instances at each node, default None (siblings policy)
	"""	
	#def __init__(self,H,Hclassifier=None,policy=None,classifiers_strategy="from_scratch",unlabeled_strategy="all",strategy_bestU="naiveThreshold",threshold=0.7,minUdata=10,maxIterations=-1,seed=0):
	def __init__(self,H,Hclassifier=None,policy=None,classifiers_strategy="from_scratch",variant="v1",threshold=0.7,minUdata=10,maxIterations=-1,seed=0,k=3,t2considerNode=0,converge=0,metric='euclidean'):

		self.classifiers_strategy = classifiers_strategy
		self.variant = variant
		self.threshold = threshold
		self.minUdata = minUdata	# minimum unalbelled data to continue the iteration process, lower or equal than 0 until all unlabeled data is pseudo-labeled 
		self.maxIterations = maxIterations	# -1: without limit
		self.seed = seed
		
		# added knn
		self.t2considerNode = t2considerNode
		self.converge = converge
		self.metric = metric
		self.k = k
		self.iniSP = False # If true, first pseudo-label each unlabeled instance with the path of its nearest labeled neighbor

		if(self.variant not in ["v1","v2","v3"]):
			raise NameError("Error!!, variant unavailable: ",self.variant)

		# validate structure 	
		if(isinstance(H,hierarchy)):
			self.H = H
		else:
			raise NameError("Error!!, H has to be an hierarchy object")

		if(Hclassifier is None):	# if the hclassifier is not provided, generate a TD
			self.Hclassifier = td(H)	# instanciate TD with default parameter: baseClassifier=nb(),policy=None,TL="from_scratch"
		else:
			self.Hclassifier = Hclassifier	# base classifier, it will be used for the semi-supervised learning

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
		#self.dCl = np.empty(self.n, dtype = object)		#contains the classifiers		
		self.HC = None		#the hierarchical classifier
		self.unIadded = np.zeros(self.n,dtype=int)		# number of unlabeled instances added to each node


	def fit(self,trainSet, cl, unlabeled): 
		#self.drawInstances(trainSet,cl,unlabeled)
		

		#self.metric = 'euclidean'
		dist = dm.get_metric(self.metric)

		shTr = np.shape(trainSet)	# shape of the train set
		if(len(shTr) != 2):
			raise NameError("The train set has to be a two-dimension numpy matrix (intances x attributtes)")			

		shCl = np.shape(cl)	# shape of the classes to which the instances are associated
		if(len(shCl) != 2):
			raise NameError("The classes has to be a two-dimension numpy matrix (intances x classes)")			

		if(shTr[0] != shCl[0]):
			raise NameError("The number of elements in data is different from classes (cl)")	

		shU = np.shape(unlabeled)

		#if( not (( "fit" in dir(self.baseClassifier) ) and ( "predict" in dir(self.baseClassifier) )) ):
		#	raise NameError("ERROR: you has to provide a valid classifier!")				
		opdDraw = "all"
		draw2d = False	#True 	#False
		#if(draw2d):
		#	self.drawLandU(trainSet,cl,unlabeled,self.H,opdDraw)
		#	#self.drawInternalsLandU(trainSet,cl,unlabeled,self.H)



		# unlabeled
		##copy_u = unlabeled.copy() 		# esto puede consumir mucha memoria

		self.HC = copy.deepcopy(self.Hclassifier)
		retrain = False		# for classifiers that can be retrained
		first = True		# first time for being trained
		bestU = []			# list with the best Unlabeled instances	# DO NOT SORT
		clBU = []			# list with classes associated to each unlabeled instance, follows the order in bestU
		
		trainLocal = trainSet
		clLocal = cl

		# 1 select the classifier:

		# this part is fulfilled in constructor, when the Hclassifier is provided
		# 			sustituir esto por un metodo:
		#				self.classifier_strategy.updateClassifiers(self.H, self.dCl), method consider first and retrain variables
		flag = True

		c=0
		addU = [] # indices of unlabeled instances added to the labeled set
		#print("shU: ",shU)
		#print("shCl: ",shCl)
		pseudo = np.zeros( (shU[0] ,shCl[1]), dtype=bool)	# pseudo-labels			# predictions in T-1
		lpseudo = np.zeros( (shU[0] ,shCl[1]), dtype=bool)	# local pseudo-labels	# prediction in T
		pseudoT2 = np.zeros( (shU[0] ,shCl[1]), dtype=bool)							# predictions in T-2 
		#print("\npseudo:\n",pseudo)


		# variant first pseudo label all the information
		if(self.iniSP):		# if True, pseudo-label each unlabeled instance with the path of its nearest labeled neighbour
			print("# unlabeled data: ", shU[0])
			##knn = bt(trainSet, leaf_size=40, metric=self.metric )		#knn			
			#di, ind = knn.query(unlabeled,k=1, dualtree=False)	# the nearest neighbour

			knn = NN(n_neighbors=1,metric=self.metric,algorithm='auto',n_jobs=10)
			knn.fit(trainSet)
			di, ind = knn.kneighbors(unlabeled, return_distance=True)

			for i in range(shU[0]):
				pseudo[i] = cl[ ind[i,0] ]
			print("added labeled data for node:\n")
			print(sum(pseudo))
			addU = [ i for i in range(shU[0]) ]
			#print("addU:\n",addU)	
			#exit()
			del knn

		flagk = False

		while(True):	#, there are unlabeled instances, or, new instances were not added
			c=c+1
			print("c: ",c)
			# 2 training

			# Begin the training
			"""
			if(self.classifiers_strategy == "from_scratch"):
				# copy new Hclassifier
				print("len trainlocal: ", len(trainLocal))		
				self.HC = copy.deepcopy(self.Hclassifier)
				self.HC.fit(trainLocal,clLocal)
				if(flag):
					if(draw2d):
						self.drawLandU(trainLocal,clLocal,copy_u,self.H,opdDraw)
					flag=False
			elif(self.classifiers_strategy == "partial_fit"):
				self.HC.partial_fit(trainLocal,clLocal)
			else:
				raise NameError("ERROR: invalid 'classifiers_strategy'!")				
			"""
			#print("addU:\n",addU)
			trainLocal = np.concatenate([trainSet, unlabeled[addU] ])
			#print(trainLocal)		
			#print("\ncl:\n",cl)
			#print("\npseudo:\n",pseudo)
			clLocal = np.concatenate([cl, pseudo[addU] ])

			#if(flag):
			#if(draw2d):
			#	sun = np.array( list( set( [i for i in range(shU[0])] ) - set(addU) ),dtype=int )	# set index unlabeled data
			#	self.drawLandU(trainLocal,clLocal,unlabeled[sun],self.H,opdDraw)
			#	#flag=False


			#knn = bt(trainLocal, leaf_size=40, metric=self.metric )	# get the tree to apply knn, with labeled and pseudo labeled data
			#knn = NN(n_neighbors=self.k,metric=self.metric,algorithm='auto',n_jobs=10)
			#knn.fit(trainLocal)
			#di, ind = knn.kneighbors(unlabeled, return_distance=True)

			# 3 classify unlabeled instances
			#if(len(copy_u) == 0):	# if all the unlabeled data were classified, then finish the loop

			"""
			lencu = len(copy_u)
			if( (lencu < self.minUdata) or (lencu == 0)):
				#print("finish 2")
				if(draw2d):
					self.drawLandU(trainLocal,clLocal,copy_u,self.H,opdDraw)
					#self.drawInternalsLandU(trainLocal,clLocal,copy_u,self.H)
				break

			if(self.maxIterations >= 0):
				if(c >= self.maxIterations ):	# maximum number iterations			
					print("Broken loop!, maximum number of iterations reached: " + str(c))
					break

			up = self.HC.predict_proba( copy_u )	# unlabeled predictions
			"""

			# for each unlabeled instance (even the already classified)
			# get the nearest points and the distances
			# *********analize delete that the same unlabelled point does not contribute to its new path

			if(self.variant == "v1"):
				#di, ind = knn.query(unlabeled,k=self.k, dualtree=False)
				knn = NN(n_neighbors=self.k,metric=self.metric,algorithm='auto',n_jobs=10)
				knn.fit(trainLocal)
				di, ind = knn.kneighbors(unlabeled, return_distance=True)
			elif((self.variant == "v2") or (self.variant == "v3") ):
				if(flagk):
					if(self.variant == "v3"):
						if((c%10)==0):
							self.k += 1
					#di, ind = knn.query(unlabeled,k=self.k+1, dualtree=False)		# the parameter dualtree=True, could speed the performance if there are a lot of points
					knn = NN(n_neighbors=self.k+1,metric=self.metric,algorithm='auto',n_jobs=10)
					knn.fit(trainLocal)
					di, ind = knn.kneighbors(unlabeled, return_distance=True)
				else:
					#di, ind = knn.query(unlabeled,k=self.k, dualtree=False)		# the parameter dualtree=True, could speed the performance if there are a lot of points
					knn = NN(n_neighbors=self.k,metric=self.metric,algorithm='auto',n_jobs=10)
					knn.fit(trainLocal)
					di, ind = knn.kneighbors(unlabeled, return_distance=True)

				if(flagk):
					#print("di:\n",di)
					#print("ind:\n",ind)
					# for each unlabeled point, delete the point that points to the same point
					di2 = np.zeros( (shU[0],self.k), dtype=float)
					ind2 = np.zeros( (shU[0],self.k), dtype=int)
					for i in range(shU[0]):	#for each unlabeled point
						if(i in addU):	# if i was added to the trainlocal
							# estimate the position in train local
							estp = shTr[0] + addU.index(i)												
							#print("estp: ",estp)
							# this should not fail if there are multiple equal points
							pidi = np.where(ind[i]==estp)[0]
							if(len(pidi)==0):
								#print("multiple points are equal!")
								di2[i,:] = di[i,:self.k]	#add the k nearest neighbours
								ind2[i,:] = ind[i,:self.k]	#add the k nearest neighbours													
							else:
								#print("pidi: ",pidi)
								di2[i,:pidi[0]] = di[i,:pidi[0]]
								di2[i,pidi[0]:] = di[i,(pidi[0]+1):]

								ind2[i,:pidi[0]] = ind[i,:pidi[0]]
								ind2[i,pidi[0]:] = ind[i,(pidi[0]+1):]
						else:
							di2[i,:] = di[i,:self.k]	#add the k nearest neighbours
							ind2[i,:] = ind[i,:self.k]	#add the k nearest neighbours						
					# keep the procesed distances and indices
					di = di2
					ind = ind2
					#print("di mod:\n",di)
					#print("ind mod:\n",ind)
				else:
					# add one k to the neigbours
					#self.k = self.k + 1	
					flagk = True
			
			else:
				raise NameError("Error!!, variant not available: ",self.variant)

			ll = [[] for i in range(shU[0])]	# list with the index that contribute to the path
			laddU = [] 	# (local) indices of unlabeled instances added to the labeled set
			addU = []
			# process the pseudo labeles for each unlabeled point
			for i in range(shU[0]):	
				#la linea de abajo no tiene mucho sentido
				#sl = np.sum( self.H.getSinglePaths()[ ind[i] ] , axis=0) / self.k
				# I must get the paths of the labeled points, add them, and divide among the number of k nearest neighbors
				sl = np.sum( clLocal[ ind[i] ] , axis=0) / self.k
				lp = sl >= self.t2considerNode	# generate a vector with true's and false's			
				if(np.any(lp)):		# if there are nodes with 'probability' higher than self.t2considerNode,
					# then recuperate the points that cooperate to the probability
					lc = np.where(lp)[0]	# index of positive labels
					for j in range(self.k):
						#if( np.any( self.H.getSinglePaths()[ ind[i,j] ] & lp ) ):	# if j instance contribute, then add it							
						if( np.any( clLocal[ ind[i,j] ] & lp ) ):	# if j instance contribute, then add it							
							ll[i].append(j)	# add the index 
					laddU.append(i)	# the (index) instance was 'modified'
					# save its path
					lpseudo[i] = lp[:]

					## analize SSI inside the first cicly
					lprob = SISI(unlabeled[i], trainLocal[ ind[i, ll[i] ] ],self.metric)						
					#if(lprob > self.threshProb):	# this decision will be combined in future with the prediction of the classifier
					if(lprob > self.threshold):	# this decision will be combined in future with the prediction of the classifier
						addU.append(i)	# the (index) instance was 'modified'					

			# due that in each iteration all the instances are considered
			# the 'correct' way to determine that the procedure converged is 
			#	the associated classes to the unlabeled data does not change
			
			#print("iteration: ",c)
			#print("number of unlabeled instances added: ",len(addU))
			#print("useless instances: ",shU[0]-len(addU))
			#print("unlabeled instances added for node:\n", lpseudo[addU].sum(axis=0))

			# all the parameters are equal:
			if(self.converge == 0):
				#print("\n\nentra")
				#print("sh lpseudo: ",np.shape(lpseudo))
				#print("sh pseudo: ",np.shape(pseudo),"\n")
				#print("type lpesudo: ",type(lpseudo))
				#print("type pseudo: ",type(pseudo),"\n")

				xx = np.all( lpseudo == pseudo ) 			
				diff = np.where(lpseudo!=pseudo)
				#print("\ndiff:\n",diff)
				#print("lpseudo, pseudo:\n")
				#print(lpseudo[diff])
				#print(pseudo[diff])

				#print("xx: ",xx)
				#print("sum lpseudo: ", np.sum(lpseudo))
				#print("sum pseudo: ", np.sum(pseudo))

				if( np.all( lpseudo == pseudo ) ):		# 
					print("successfully converged by T-1!")
					#self.drawLandU(trainLocal,clLocal,copy_u,self.H,opdDraw)
					break

				elif(np.all( lpseudo == pseudoT2 )):
					print("successfully converged by T-2!")
					#self.drawLandU(trainLocal,clLocal,copy_u,self.H,opdDraw)
					break
			else:
				# percentage of aceptable error
				# evaluation measure exact match:
				if( np.all( lpseudo == pseudo,axis=1).sum()/shU[0] < self.converge):
					print("converged by threshold!")
					break

			if(self.maxIterations != -1):
				if(c >= self.maxIterations):
					print("Maximum number of iteration reached!: " +str(c) )
					break

			#for new iteration copy new data to old data
			pseudoT2[:,:] = pseudo[:,:]		# T-2 receives info from T-1
			pseudo[:,:] = lpseudo[:,:]		# T-1 receives info from T
			lpseudo.fill(False)				# T is empty
			continue
			

		self.isfit = True
		print("******finished cylce***********")
		#print("iteration: ",c)
		#print("number of unlabeled instances added: ",len(bestU))
		#print("useless instances: ",len(copy_u))
		#print("unlabeled instances added for node:\n",self.unIadded)
		print("iteration: ",c)
		#print("number of unlabeled instances added: ",len(addU))
		#print("useless instances: ",shU[0]-len(addU))
		self.unIadded = pseudo[addU].sum(axis=0)
		#print("unlabeled instances added for node:\n", self.unIadded)

		#training 
		print("\n\nTraining local classifiers ...")
		if(self.classifiers_strategy == "from_scratch"):
			# copy new Hclassifier
			#print("len trainlocal: ", len(trainLocal))		
			self.HC = copy.deepcopy(self.Hclassifier)
			self.HC.fit(trainLocal,clLocal)

			sun = np.array( list( set( [i for i in range(shU[0])] ) - set(addU) ),dtype=int )	# set index unlabeled data
			#self.drawInstances(trainLocal,clLocal,unlabeled[sun])
			#if(flag):
			#if(draw2d):
			#	sun = np.array( list( set( [i for i in range(shU[0])] ) - set(addU) ),dtype=int )	# set index unlabeled data
			#	self.drawLandU(trainLocal,clLocal,unlabeled[sun],self.H,opdDraw)
			#	flag=False
		elif(self.classifiers_strategy == "partial_fit"):
			self.HC.partial_fit(trainLocal,clLocal)
		else:
			raise NameError("ERROR: invalid 'classifiers_strategy'!")
		print("Finish training")

		#self.c = c
		#self.addedInstances = len(bestU)
		#self.uselessInstances = len(copy_u)
		self.c = c
		self.addedInstances = len(addU)
		self.uselessInstances = shU[0] - self.addedInstances

	
