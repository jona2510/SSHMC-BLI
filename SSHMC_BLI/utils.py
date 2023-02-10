"""
This code belongs to Semi-Supervised Hierarchical Multi-label Classifier Based on Local Information
	PGM_PyLib: https://github.com/jona2510/SSHMC-BLI

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#libraries
import numpy as np

#from scipy.stats import chi2 as c2
#from scipy.stats import chi2_contingency as c2c
from sklearn.neighbors import DistanceMetric as dm




def SISI(p,s,metric="euclidean"):
	"""
	(S)imilitude of an (I)nstance with a (S)et of (I)nstances
	p: nd_array of size(n_attributes)
	s: nd_array of size(m_instances,n_attributes)
	"""

	shP = np.shape(p)
	shS = np.shape(s)

	if(len(shP) != 1):
		print("ERROR! p has to be a nd_array of size(n_attributes)")
		exit()
	if(len(shS) != 2):		
		print("ERROR! s has to be a nd_array of size(m_instances,n_attributes)")
		#print(s)
		#print("shape: ",shS)
		exit()
	if(shP[0] != shS[1]):
		print("ERROR! number of attributes is different (p,s)")
		exit()

	if(shS[0] == 1):
		print("WARNING!!! there is only one instance in s, returning score: 0")
		return 0.0
		#print("WARNING!!! there is only one instance in s, returning score: 1")
		#return 1.0
	#print("********* s ********")
	#print("s shape: ",shS)
	#print(s)
	dist = dm.get_metric(metric)

	md = dist.pairwise( s )		# all vs all (labeled)
	lavg = np.sum( np.triu(md,k=1) ) / shS[0]		#  distances average of labeled points
	# get the distances among the unlabeled and the nearest labeled points (knn)	
	uavg = np.average( dist.pairwise([p],s) )

	score = 0.0			
	# estimate pseudo-probability	(score)
	if(uavg <= lavg):	# if uavg is lower or equal to lavg, then probability 1
		# that is, the unlabeled point is 'inside' the labeled points
		score = 1.0
	elif(uavg >= (3*lavg)):	# if the unlabeled point is 'too' far, then probability 0
		# that is, the unlabeled point is 'outside' the labeled points
		score = 0.0
	else:
		# in other case, estimate a simple linear score 
		score = 1.5 - uavg / (2 * lavg)
	return score
	

