import drawing
import pandas as pd 

from math import log
from collections import Counter



#--------------------------------------------------
class ID3:

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
	def getEntropy(self, labels):

		labelCounts = dict(Counter(labels))
		entropy = 0.0

		for label, count in labelCounts.items():
			probability = float(count) / len(labels)
			entropy -= probability * log(probability,2)

		return entropy
	#------------------------------

	#------------------------------
	def getSubsets(self, features, labels, featureIndex, attribute):

		subFeatures,subLabels = list(),list()

		for featureNumber in range(len(features)):
			#--------------------
			#選定feature後，以它為Root之取出子樹所需要的資料
			#也就是除了它本身以外剩下來其他的特徵
			feature = features[featureNumber]
			if (feature[featureIndex] == attribute):
				#----------
				#取出除了featureIndex以外其他的特徵
				subFeature = feature[:featureIndex]
				#剩下的特徵以延伸的方式加入既有特徵的資料集中 
				subFeature.extend(feature[featureIndex+1:])
				#----------
				#----------
				subFeatures.append(subFeature)
				subLabels.append(labels[featureNumber])
				#----------
			#--------------------

		return subFeatures, subLabels
	#------------------------------

	#------------------------------
	def getSplitIndex(self, features, labels, titles):
		#--------------------
		#[POINT 1]
		#計算全部特徵下的信息熵
		baseEntropy = self.getEntropy(labels)
		#--------------------		
		#--------------------
		#[POINT 2]
		#計算個別特徵下的條件熵
		maxInfoGain = 0.0
		splitIndex = -1
		for featureIndex in range(len(features[0])):
			#----------
			#統計這個特徵下所有可能的的屬性
			attributes = set([feature[featureIndex] for feature in features])
			featureEntropy = 0.0
			for attribute in attributes:
				#取得這個特徵下特定屬性的屬性值(屬性值就是該屬性對應到了類別)
				subFeatures, subLabels = self.getSubsets(features, labels, featureIndex, attribute)
				#-----
				#[POINT 2.1]
				#計算條件熵
				probability = float(len(subFeatures)) / len(features)
				featureEntropy += probability * self.getEntropy(subLabels)
				#-----
			#----------
			#----------
			#[POINT 2.2]
			#計算信息增益
			infoGain = baseEntropy - featureEntropy
			
			if (infoGain > maxInfoGain):
				maxInfoGain = infoGain
				splitIndex = featureIndex
			#----------
		#--------------------

		return splitIndex
	#------------------------------

	#------------------------------
	def creatDecisionTree(self, features, labels, titles):
		#--------------------
		#判斷傳入的類別中是否只剩下一種
		#如果是就傳回該類別
		classes = set(labels)
		if(len(classes) == 1): return labels[0]
		#--------------------
		#--------------------
		#找出最佳的分割特徵	
		splitIndex = self.getSplitIndex(features, labels, titles)
		#--------------------
		#--------------------
		#找出最佳的分割特徵	對應的標頭
		#因為用過的特徵不會再繼續使用，所以找出後將其刪除
		splitTitle = titles[splitIndex]
		titles.remove(splitTitle)
		#--------------------
		#--------------------
		#建立決策樹
		decisionTree = {splitTitle:{}}
		#統計這個特徵下所有可能的的屬性
		attributes = set([feature[splitIndex] for feature in features])
		for attribute in attributes:
			#取得這個特徵下特定屬性的屬性值(屬性值就是該屬性對應到了類別)
			subFeatures, subLabels = self.getSubsets(features, labels, splitIndex, attribute)
			subTitles = titles[:]
			#遞迴建立決策樹
			decisionTree[splitTitle][attribute] = self.creatDecisionTree(subFeatures, subLabels, subTitles)
		#--------------------

		return decisionTree
	#------------------------------

	#------------------------------	
	def classify(self, decisionTree, input):
	
		root = list(decisionTree.keys())[0]
		subTree = decisionTree[root]
		rootTitle = self.titles.index(root)
		#--------------------
		#[POINT 3]
		#遞迴在決策樹中尋找輸入資料的對應的類別
		for subRoot in subTree.keys():
			if (input[rootTitle] == subRoot):
				if type(subTree[subRoot]).__name__ == 'dict':
					classLabel = self.classify(subTree[subRoot], input)
				else: classLabel = subTree[subRoot]
		#--------------------

		return classLabel
	#------------------------------

	#------------------------------
	def drawDecisionTree(self):

		drawing.createPlot(self.decisionTree)
	#------------------------------

	#------------------------------
	def train(self):

		self.decisionTree = self.creatDecisionTree(self.features, self.labels, self.titles.copy())	

		return 
	#------------------------------

	#------------------------------
	def predict(self, inputs):
		
		predictResults = list()

		#--------------------
		for input in inputs:
			predictLabel = self.classify(self.decisionTree, input)
			predictResults.append( [input,predictLabel])
		#--------------------

		return predictResults
	#------------------------------
#--------------------------------------------------



if __name__ == '__main__':

	id3 = ID3()
	id3.loadSamples('weather.csv')
	id3.train()
	id3.drawDecisionTree()