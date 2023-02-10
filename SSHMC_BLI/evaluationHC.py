"""
This code belongs to Semi-Supervised Hierarchical Multi-label Classifier Based on Local Information
	PGM_PyLib: https://github.com/jona2510/SSHMC-BLI

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

# all this metrics are for Hierarchical Classification
#	that is, the labels/classes are arranged in a predefined structure
#	and the instances are associated to a subset of the labels while complain the hierarchical constraint
#
# matrix = rows x coloumns
# matrix = instances x labels



#libraries
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf

def check_error(shapeR, shapeP):
	if( (len(shapeR) != 2) or (len(shapeP) != 2) ):
		raise NameError( "Error: you has to provide two two-dimensional numpy matrices!" )

	if( (shapeR[0] != shapeP[0]) or (shapeR[1] != shapeP[1]) ):
		raise NameError( "Error: The dimensions of the matrices are different!" )		



# Exact match
def exactMatch(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)		

	c=0.0
	for i in range(shR[0]):
		if( np.all(real[i] == prediction[i]) ):		# np.all returns true if all the elements are true
			c += 1

	return (c/shR[0])

# hierarchical accuracy 
def hAccuracy(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)	
	
	acc = 0.0		

	for i in range(shR[0]):			# walks over the intances
		yr = set(np.where(real[i] == 1)[0])
		yp = set(np.where(prediction[i] == 1)[0])
		acc += len(yr&yp)/len(yr|yp)	#simmetric difference/union
		#for j in range(shR[1]):		# walks over the labels
		#	if(real[i,j] != prediction[i,j]):	#symmetric difference
		#		acc += 1
	return (acc / shR[0] )


#	 WARNING!!!
# THIS IS: Sensitivity, recall, true positive rate
# IT IS NOT ACCURACI
# Accuracy
def accuracy_BAD(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)
	check_error(shR, shP)		
	acc = 0.0		
	for i in range(shR[0]):			# walks over the intances
		union = 0.0				# sum of predicted and real
		intersection = 0.0		# correctly predicted
		for j in range(shR[1]):		# walks over the labels
			#if(real[i,j] == prediction[i,j]):
			if(real[i,j] == 1):
				if(prediction[i,j] == 1):
					intersection += 1
				union += 1
			else:
				if(prediction[i,j] == 1):
					union += 1						
		acc += intersection / union
	return ( acc / shR[0] )

# it is threat score (TS) or critical success index (CSI)
# 	TP/(TP + FN + FP)
def csi(real, prediction):	# rename to csi, before accuracy (wrong name)
	shR = np.shape(real)
	shP = np.shape(prediction)
	check_error(shR, shP)
		
	acc = np.zeros(shR[1])
	for i in range(shR[1]):
		realL = real[:,i] == 1			# real labels
		predL = prediction[:,i] == 1		# predicted labels
		acc[i] = len(np.where( realL & predL )[0]) / len(np.where( realL | predL )[0])	# TP/(TP + FN + FP)

	return np.average(acc)



# Hamming Loss
#	it is used in Hamming accuracy
def hammingLoss(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)	

	acc = 0.0		

	for i in range(shR[0]):			# walks over the intances
		yr = set(np.where(real[i] == 1)[0])
		yp = set(np.where(prediction[i] == 1)[0])
		acc += len(yr^yp)	#simmetric difference
		#for j in range(shR[1]):		# walks over the labels
		#	if(real[i,j] != prediction[i,j]):	#symmetric difference
		#		acc += 1
	return (acc / (shR[0] * shR[1]) )


# Hamming accuracy
def hammingAccuracy(real, prediction):
	hLoss = hammingLoss(real, prediction)
	return (1 - hLoss)



# hierarchical recall h R
#
def hRecall(real, prediction, check=True):	

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	nReal = 0.0				# number of "real" labels
	intersection = 0.0		# correctly predicted

	for i in range(shR[0]):			# walks over the intances
		for j in range(shR[1]):		# walks over the labels
			if(real[i,j] == 1):
				if(prediction[i,j] == 1):
					intersection += 1
				nReal += 1

	if(nReal == 0):
		print("WARNING: the number of real associations in the whole dataset is zero")
		return 0
	return ( intersection / nReal )


# hierarchical precision hP
#
def hPrecision(real, prediction, check=True):

	shR = np.shape(real)
	shP = np.shape(prediction)

	if(check):
		check_error(shR, shP)

	nPred = 0.0				# number of "predicted" labels
	intersection = 0.0		# correctly predicted

	for i in range(shR[0]):			# walks over the intances
		for j in range(shR[1]):		# walks over the labels
			if(prediction[i,j] == 1):
				if(real[i,j] == 1):
					intersection += 1
				nPred += 1

	if(nPred == 0.0):
		print("WARNING: the number of predictions in the whole dataset was zero")
		return 0
	else:
		return ( intersection / nPred )

# hierarchical F measure (hF)
def hFmeasure(real, prediction, check=True):

	hP = hPrecision(real, prediction, True)
	hR = hRecall(real, prediction, False)

	if(hP == hR == 0):
		print("WARNING: hierarchical Precision and hierarchical Recall were zero")
		return 0
	else:
		return ( (2 * hP * hR) / (hP + hR) )

"""
Matthews Correltaion Coeficient (MCC)
Binary Classification!!!, (True,False) or (1,0)
worst value: –1; best value: +1

paper: The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation
by: Davide Chicco and Giuseppe Jurman
"""
def MCC(real, prediction):
	#	Matrix:
	#					Predicted positive, Predicted negative
	#	Actual Positive		TP, 				FN
	#	Actual Negative		FP, 				TN

	def localMCC(TP,FN,FP,TN):
		return (  (TP*TN - FP*FN) / np.sqrt( (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) ) ) 
	
	shR = np.shape(real)
	shP = np.shape(prediction)

	if( (len(shR) != 1) or (len(shP) != 1) ):
		raise NameError( "Error: you has to provide two one-dimensional numpy arrays!" )

	if( shR[0] != shP[0] ):
		raise NameError( "Error: The size of the arrays are different!" )		

	realP = (real == 1)
	realN = (real == 0)
	predP = (prediction == 1)
	predN = (prediction == 0)
	
	cm = np.zeros(4)	# confusion matrix
	cm[0] = TP = float( len( np.where( realP & predP)[0] ) )	# True Positives
	cm[1] = FN = float( len( np.where( realP & predN)[0] ) )	# False Negatives
	cm[2] = FP = float( len( np.where( realN & predP)[0] ) )	# False Positives
	cm[3] = TN = float( len( np.where( realN & predN)[0] ) )	# True Negatives



	zeros = np.where(cm == 0)[0]	#cells of the cm where there are zeros values
	if(len(zeros) >= 3):
		raise NameError("Error!, at least 3 cells of the confusion matrix contain zero values.")
	elif(len(zeros) == 1):	# only one value is non-zero	# error it was ==3
		if((0 not in zeros) or (3 not in zeros)):	
			return (1.0)
		else:
			return (-1.0)
	elif(len(zeros) == 2):	# two values are zero
		# the four posible cases (row or column full of zeros):
		#if( ((0 in zeros) and (1 in zeros)) or ((0 in zeros) and (2 in zeros)) ):
		#	a = cm[zeros[0] ]
		#	b = cm[zeros[1] ]
		#elif( ((2 in zeros) and (3 in zeros)) or ((1 in zeros) and (3 in zeros)) ):	
		#	a = cm[zeros[1] ]
		#	b = cm[zeros[0] ]
		if( ((0 in zeros) and (1 in zeros)) or ((0 in zeros) and (2 in zeros)) or ((2 in zeros) and (3 in zeros)) or ((1 in zeros) and (3 in zeros)) ):	
			#print("Warning!! a row/column of the confusion matrix is full of zeros, 0 returned")
			return 0.0
		else:
			#compute normal MCC
			return localMCC(TP,FN,FP,TN)
		#print("Warning!! a row/column of the confusion matrix is full of zeros, so, epsilon value (",epsilon,") is used!!")
		#return ( np.sqrt(epsilon) * ( (a - b) / np.sqrt( 2*a*b*(b-a) ) )  )
	return localMCC(TP,FN,FP,TN)


"""
Multi-label classification,
for each label it estimates Matthews Correltaion Coeficient (MCC) and return the array
"""
def mlc_MCC(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)
	
	check_error(shR, shP)

	aMCC = np.zeros(shR[1])
	for i in range(shR[1]):			# estimate MCC for each label
		aMCC[i] = MCC(real[:,i],prediction[:,i])
	return aMCC
 



# weighted Hamming accuracy
def wHAccuracy(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)

	check_error(shR, shP)	
	
	acc = 0.0		

	for i in range(shR[0]):			# walks over the intances
		yr = set(np.where(real[i] == 1)[0])
		yp = set(np.where(prediction[i] == 1)[0])
		acc += len(yr^yp)/len(yr|yp)	#simmetric difference/union  
		#for j in range(shR[1]):		# walks over the labels
		#	if(real[i,j] != prediction[i,j]):	#symmetric difference
		#		acc += 1
	return (1 - (acc / shR[0] ))


# kind of micro accuracy for multilabel
def microAccuracy(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)
	check_error(shR, shP)
		
	union = 0.0				# sum of predicted and real
	intersection = 0.0
	for i in range(shR[1]):
		yr = set(np.where(real[i] == 1)[0])
		yp = set(np.where(prediction[i] == 1)[0])
		union += len(yr|yp)	
		intersection += len(yr&yp)
	if(union == 0):
		print("Warining!!, microAccuracy: sum of 'unions' is zero, return 0")
		return 0
	return (intersection/union)





#
#	BELOW ARE BINARY-BASED EVALUATION MEASURES!!!!
#

# accuracy binary
# (TP+TN)/(TP+TN+FP+FN)
def accBinary(real,prediction):
	if(len(real)!=len(prediction)):
		raise NameError("Error!!, sizes are different")
	return( len(np.where(real==prediction)[0])/len(real) )

"""
Multi-label classification,
for each label it estimates accuracy (binary) and return the array
"""
def mlc_accuracy(real, prediction):
	shR = np.shape(real)
	shP = np.shape(prediction)
	
	check_error(shR, shP)

	acc = np.zeros(shR[1])
	for i in range(shR[1]):			# estimate accuracy for each label
		acc[i] = accBinary(real[:,i],prediction[:,i])
	return acc	


# recall
# TP / (TP + FN)
def recallBinary(real,prediction):
	if(len(real)!=len(prediction)):
		raise NameError("Error!!, sizes are different")
	TP = len( np.where( np.logical_and(real, prediction) )[0] )		#0.0	# True positive
	#TN = len( np.where( np.logical_and( np.logical_not(real[:,i]), np.logical_not(prediction[:,i]) ) )[0] )		#0.0	# True negative
	#FP = len( np.where( np.logical_and( np.logical_not(real[:,i]), prediction[:,i]) )[0] )		#0.0	# False positive
	FN = len( np.where( np.logical_and(real, np.logical_not(prediction) ) )[0] )		#0.0	# False negative

	return (TP / (TP + FN))


# precision
# TP / (TP + FP)
def precisionBinary(real,prediction):
	if(len(real)!=len(prediction)):
		raise NameError("Error!!, sizes are different")

	TP = len( np.where( np.logical_and(real, prediction) )[0] )		#0.0	# True positive
	#TN = len( np.where( np.logical_and( np.logical_not(real[:,i]), np.logical_not(prediction[:,i]) ) )[0] )		#0.0	# True negative
	FP = len( np.where( np.logical_and( np.logical_not(real), prediction) )[0] )		#0.0	# False positive
	#FN = len( np.where( np.logical_and(real[:,i], np.logical_not(prediction[:,i]) ) )[0] )		#0.0	# False negative

	return (TP / (TP + FP))

# f1
#  (2*precision*recall)/(precision+recall)
def f1FEL(real,prediction,returnPR=False):		# f1 fFor Each Label
	shR = np.shape(real)
	shP = np.shape(prediction)
	
	check_error(shR, shP)
	R = np.zeros(shR[1])
	P = np.zeros(shR[1])
	f1 = np.zeros(shR[1])

	for i in range(shR[1]):
		aux = prf(real[:,i],prediction[:,i],average='binary')
		P[i] = aux[0]		#precisionBinary(real[:,i],prediction[:,i])
		R[i] = aux[1]		#recallBinary(real[:,i],prediction[:,i])
		f1[i] = aux[2]	#(2*P*R)/(P+R)

	if(returnPR):
		return (f1,P,R)
	else:
		return f1














