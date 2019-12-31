import ast
import drawing
import pandas as pd 
from collections import Counter



#--------------------------------------------------
class NODE:
	def __init__(self):
		self.rightFeatures = list()
		self.rightLabels = list()
		self.rightAttributes = set()
		self.leftFeatures = list()
		self.leftLabels = list()
		self.leftAttributes = set()
#--------------------------------------------------



#--------------------------------------------------
class CART:

	#------------------------------
	def loadSamples(self, samplePath):

		df = pd.read_csv(samplePath)  

		self.samples = df.values[:,:].tolist() 
		self.titles = df.columns.values[:-1].tolist()
		self.features = [sample[:-1] for sample in self.samples]
		self.labels = [sample[-1] for sample in self.samples]	
		self.classes = self.labels

		return
	#------------------------------

	#------------------------------
	def getGini(self, labels):

		labelCounts = dict(Counter(labels))
		score = 0.0
		for label, count in labelCounts.items():
			probability = float(count) / len(labels)
			score += probability * probability

		gini = 1.0 - score

		return gini
	#------------------------------

	#------------------------------
	def getSubsets(self, features, labels, featureIndex, targetAttribute):			
		node = NODE()

		for sampleIndex in range(len(features)):
			attribute = features[sampleIndex][featureIndex]

			if (attribute == targetAttribute):
				node.leftFeatures.append(features[sampleIndex])
				node.leftLabels.append(labels[sampleIndex])
				node.leftAttributes.add(attribute)
			else:
				node.rightFeatures.append(features[sampleIndex])
				node.rightLabels.append(labels[sampleIndex])
				node.rightAttributes.add(attribute)

		return node
	#------------------------------

	#------------------------------
	def creatDecisionTree(self, features, labels, titles):

		#--------------------
		#----------
		if(labels == None): return None
		#----------
		#----------
		#判斷傳入的dataset中是否只有一種類別，是，返回該類別
		classes = set(labels)
		if(len(classes) == 1): return labels[0]
		#----------
		#--------------------
		
		#--------------------
		baseGini = self.getGini(labels)
		#--------------------

		#--------------------
		maxGain = 0.0
		splitIndex = -1
		splitTitle = 'Other'
		splitNode = NODE()

		for featureIndex in range(len(features[0])):	
			attributes = set([feature[featureIndex] for feature in features])

			for attribute in attributes:			
				#----------
				node = self.getSubsets(features, labels, featureIndex, attribute)	

				leftGini = self.getGini(node.leftLabels)
				rightGini = self.getGini(node.rightLabels)
				#----------
				#----------
				leftProbability = float(len(node.leftLabels)) / len(labels)
				rightProbability = 1.0 - leftProbability

				expectation = leftProbability*leftGini + rightProbability*rightGini			
				#----------
				#----------
				gain = baseGini - expectation

				if (gain>maxGain and len(node.leftLabels)>0 and len(node.rightLabels)>0):
					maxGain = gain
					splitIndex = featureIndex
					splitTitle = titles[featureIndex]
					splitNode = node
				#----------
		#--------------------

		#--------------------	
		decisionTree = {splitTitle:{}}

		if (maxGain > 0):
			leftBranch = self.creatDecisionTree(splitNode.leftFeatures, splitNode.leftLabels, titles)
			rightBranch = self.creatDecisionTree(splitNode.rightFeatures, splitNode.rightLabels, titles)
			decisionTree[splitTitle][str(splitNode.leftAttributes)] = leftBranch
			decisionTree[splitTitle][str(splitNode.rightAttributes)] = rightBranch
		else:
			labelCounts = dict(Counter(labels))
			decisionTree[splitTitle][''] = str(labelCounts)
		#--------------------

		return decisionTree
	#------------------------------

	#------------------------------
	def train(self):

		self.decisionTree = self.creatDecisionTree(self.features, self.labels, self.titles)	

		return 
	#------------------------------

	#------------------------------	
	def classify(self, decisionTree, input):
	
		root = list(decisionTree.keys())[0]
		subTree = decisionTree[root]
		rootTitleIndex = self.titles.index(root)

		#--------------------
		#[POINT 3]
		#遞迴在決策樹中尋找輸入資料的對應的類別
		for subRoot in subTree.keys():
			branchElements = ast.literal_eval(subRoot)
			inputElement = input[rootTitleIndex]

			if (inputElement in branchElements):
				if type(subTree[subRoot]).__name__ == 'dict':
					classLabel = self.classify(subTree[subRoot], input)
				else: classLabel = subTree[subRoot]
		#--------------------

		return classLabel
	#------------------------------

	#------------------------------
	def predict(self, inputs):
		
		predictResults = list()

		#--------------------
		for input in inputs:
			predictLabel = self.classify(self.decisionTree, input)
			predictResults.append([input,predictLabel])
		#--------------------

		return predictResults
	#------------------------------

	#------------------------------
	def drawDecisionTree(self):

		drawing.createPlot(self.decisionTree)
	#------------------------------
#--------------------------------------------------



if __name__ == '__main__':
	cart = CART()
	cart.loadSamples('weather.csv')
	cart.train()
	cart.drawDecisionTree()