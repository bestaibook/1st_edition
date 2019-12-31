from __future__ import division
from builtins import len
import numpy as np



#----------------------------------------
class ADABOOST:

	#------------------------------
	def __init__(self, weakLearners):

		self.weakLearners = weakLearners
		self.strongLearner = []
	#------------------------------
	
	#------------------------------
	def train(self, samples, labels):

		#--------------------
		def getNewWeights(samples, weakLearner, oldWeights):

			N = len(samples)

			#----------			
			#[POINT 2.1]
			predictResults = list()

			for index in range(0,N):
				predictLabel = weakLearner(samples[index])
				label = labels[index]

				if(predictLabel != label):
					predictResults.append(False)
				else:
					predictResults.append(True)

			predictResults = np.array(predictResults)
			
			error = 0

			for i in range(len(predictResults)):
				if(predictResults[i] == False):
					error = error + oldWeights[i]
			#----------
			#----------
			#[POINT 2.2]
			alpha = 0.5 * np.log((1-error)/error)
			#----------		
			#----------
			#[POINT 2.3]
			newWeights = np.zeros(N)

			for i in range(N):
				data = samples[i]

				if predictResults[i] == False:
					newWeights[i] = oldWeights[i] * np.exp( alpha)
				else:
					newWeights[i] = oldWeights[i] * np.exp(-alpha)
					
			Z = newWeights.sum()

			for i in range(0, len(newWeights)):	
				preNewWeight = newWeights[i]
				newWeights[i] = newWeights[i]/Z
			#----------

			return alpha, newWeights
		#--------------------
		
		self.classes = list(set(labels))

		#--------------------
		#[POINT 1]
		N = len(samples)
		weights = np.ones(N)/N		
		#--------------------		
		#--------------------
		#[POINT 2]	
		alphaList = list()

		for weakLearner in self.weakLearners:
			#----------	
			alpha, weights = getNewWeights(samples, weakLearner, weights)
			#----------
			
			alphaList.append(alpha)
		#--------------------
		#--------------------
		#[POINT 3]
		totalProportion = sum(alphaList)

		for i in range(0,len(alphaList)):			
			proportion = int(np.round((alphaList[i]/totalProportion)*100.0))
			self.strongLearner.append((proportion, weakLearners[i]))
		#--------------------

		return
	#------------------------------

	#------------------------------
	def predict(self, inputs):

		predictResults = list()

		#--------------------
		#[POINT 4]	
		for input in inputs:

			hist = dict({label:int(0) for label in self.classes})

			#----------	
			for proportion, weakLearner in self.strongLearner:
				weakPredictLabel = weakLearner(input)
				hist[weakPredictLabel] = hist[weakPredictLabel] + proportion
			
			strongPredictLabel = max(hist, key=lambda k: hist[k])	
			predictResults.append((input, strongPredictLabel))
			#----------
		#--------------------

		return predictResults
	#------------------------------
#----------------------------------------



#----------------------------------------
if __name__ == '__main__':

	#------------------------------
	#--------------------
	samples = []
	samples.append((2, 3))
	samples.append((3, 3))
	samples.append((2, 2))
	samples.append((4, 2))
	samples.append((4, 1))

	samples.append((1, 3))
	samples.append((1, 2))
	samples.append((1, 1))
	samples.append((2, 1))
	samples.append((3, 1))
	#--------------------
	#--------------------
	labels = []
	labels.append('Accept')
	labels.append('Accept')
	labels.append('Accept')
	labels.append('Accept')
	labels.append('Accept')

	labels.append('Reject')
	labels.append('Reject')
	labels.append('Reject')
	labels.append('Reject')
	labels.append('Reject')
	#--------------------
	#------------------------------



	#------------------------------
	#--------------------
	def weakLearner1(data):
		if(data[0] > 3): return 'Accept'
		else: return 'Reject'
	#--------------------
	#--------------------
	def weakLearner2(data):
		if(data[0] > 1): return 'Accept'
		else: return 'Reject'
	#--------------------
	#--------------------
	def weakLearner3(data):
		if(data[1] > 1): return 'Accept'
		else: return 'Reject'
	#--------------------
	#--------------------
	weakLearners = []
	weakLearners.append( weakLearner1 )
	weakLearners.append( weakLearner2 )
	weakLearners.append( weakLearner3 )

	adaBoost = ADABOOST(weakLearners)
	adaBoost.train(samples, labels)
	#--------------------
	#------------------------------

	#------------------------------
	samples = []
	samples.append((3, 2)) #Jack
	predictResults = adaBoost.predict(samples)
	for input, predictLabel in predictResults:
		print(predictLabel)
	#------------------------------