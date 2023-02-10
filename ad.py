"""
This code belongs to Semi-Supervised Hierarchical Multi-label Classifier Based on Local Information
	PGM_PyLib: https://github.com/jona2510/SSHMC-BLI/tree/master

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""


#import matplotlib.pyplot as plt
from SSHMC_BLI.hStructure import hierarchy
from SSHMC_BLI.HSSL import SSHMCBLI as ssbli
from SSHMC_BLI.hStructure import policiesLCN as pol
from SSHMC_BLI.LCN import LCN
from sklearn.ensemble import RandomForestClassifier as rfc
from SSHMC_BLI.MPP import limitProb as limp
import numpy as np
import copy
from sklearn.metrics import average_precision_score as aps
from scipy.stats import norm


# hierarchy:
dag = np.zeros((6,6),dtype=int)
dag[0,3] = 1
dag[0,4] = 1
dag[1,4] = 1
dag[1,5] = 1

H = hierarchy(dag)
H.initialize()


#def norm_set(n_samples,random_s=0):
rs = 1000
mx = 2
ni = 5+55+50
x4 = np.zeros((ni,2))
x4[:,0] = norm.rvs(loc=4, scale=1, size=ni, random_state=rs+0)
x4[:,1] = norm.rvs(loc=4, scale=1, size=ni, random_state=rs+1)

x3 = np.zeros((ni,2))
x3[:,0] = norm.rvs(loc=0, scale=1, size=ni, random_state=2)
x3[:,1] = norm.rvs(loc=4, scale=1, size=ni, random_state=3)

x5 = np.zeros((ni,2))
x5[:,0] = norm.rvs(loc=7, scale=1, size=ni, random_state=4)
x5[:,1] = norm.rvs(loc=4, scale=1, size=ni, random_state=5)

x2 = np.zeros((ni,2))
x2[:,0] = norm.rvs(loc=4, scale=1, size=ni, random_state=6)
x2[:,1] = norm.rvs(loc=7, scale=1, size=ni, random_state=7)

x0 = np.zeros((ni,2))
x0[:,0] = norm.rvs(loc=0, scale=2, size=ni, random_state=8)
x0[:,1] = norm.rvs(loc=4, scale=1, size=ni, random_state=9)

x1 = np.zeros((ni,2))
x1[:,0] = norm.rvs(loc=8, scale=2, size=ni, random_state=10)
x1[:,1] = norm.rvs(loc=4, scale=1, size=ni, random_state=11)

# x: instances, y: classes associated to each intenances; u: unlabeled instances
x = np.concatenate([ x0[:mx], x1[:mx], x2[:mx], x3[:mx], x4[:mx], x5[:mx] ])
xt = np.concatenate([ x0[5:55], x1[5:55], x2[5:55], x3[5:55], x4[5:55], x5[5:55] ])
u = np.concatenate([ x0[55:], x1[55:], x2[55:], x3[55:], x4[55:], x5[55:] ])

y = np.zeros((mx*6,6),dtype=bool)
y[:mx,0] = 1 	
y[mx:mx*2,1] = 1
y[mx*2:mx*3,2] = 1
y[mx*3:mx*4,3] = 1
y[mx*4:mx*5,4] = 1
y[mx*5:mx*6,5] = 1
for i in range(len(y)):
	y[i] = H.combinePaths( np.where( y[i] == 1 )[0] )	# make consistent paths

yt = np.zeros((300,6),dtype=bool)
yt[:50,0] = 1 	
yt[50:100,1] = 1
yt[100:150,2] = 1
yt[150:200,3] = 1
yt[200:250,4] = 1
yt[250:300,5] = 1
for i in range(len(y)):
	yt[i] = H.combinePaths( np.where( yt[i] == 1 )[0] )	# make consistent paths	





sspol = pol(H,balanced=False,policy="balancedBU")	# policy for selecting positive and negative instances
bc = rfc(n_estimators=100,n_jobs=5,random_state=0)	# base classifier for each binary classifier
supc = LCN(H,baseClassifier=bc,policy=sspol,TL="from_scratch")	# hierarchical multi-label classifier baseod on local classifier per node
bhc = limp(H,baseClassifier=supc,TL="from_scratch")		# post-processing output of the LCN


vrs = ["v1","v2","v3"]
res = []
for variant in vrs:
	th = 0.5
	kk = 3
	# initialize the proposed method:
	hc = ssbli(H,Hclassifier=bhc, policy=None,classifiers_strategy="from_scratch",variant=variant,minUdata=10,maxIterations=50,threshold=th,seed=0,k=kk,t2considerNode=0.5,converge=0,metric='euclidean')	#t2considerNode was 0.5
	# training:
	hc.fit(x,y,u)
	p = hc.predict_proba(xt)
	res.append( aps(yt, p, average="micro")  )
	#print("sshmc-bli," + str(aps(yt, p, average="micro") ))

#supervised classifier
cl = copy.deepcopy(bhc)	# for supervised hierarchical multi-label classifier
cl.fit(x,y)
p = cl.predict_proba(xt)


print("***********************************")
print("Results:")
print("LNC: " + str(aps(yt, p, average="micro") ))
for x,y in zip(vrs,res):
	print(x+": "+str(y))

#SSHCKN


