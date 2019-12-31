import drawing
import pandas as pd 
from math import log
from collections import Counter



#---------------------------------------------
class C45:

	#------------------------------
	def creatDecisionTree(self, titles, featureVectors, labels):

		#--------------------
		# 計算熵
		def getEntropy(labels):

			#----------
			labelCounts = dict(Counter(labels))
			entropy = 0.0
			for label, count in labelCounts.items():
				probability = float(count) / len(labels)
				entropy -= probability * log(probability,2)
			#----------

			return entropy
		#--------------------
		
		#--------------------
		# 收集目標特徵值的子資料集與子標籤列表
		def getSubsets(featureVectors, labels, featureIndex, targetValue):

			subFeatureVectors = list()
			subLabels = list()

			for featureNumber in range(len(featureVectors)):

				featureVector = featureVectors[featureNumber]

				if (featureVector[featureIndex] == targetValue):
					#----------
					# 收集目標特徵左側的其他特徵索引
					subFeatureVector = featureVector[:featureIndex]
					# 收集目標特徵右側的其他特徵索引
					subFeatureVector.extend(featureVector[featureIndex+1:])

					# 收集目標特徵對應的資料
					subFeatureVectors.append(subFeatureVector)
					#----------
					#----------
					# 收集目標特徵對應的標籤
					subLabels.append(labels[featureNumber])
					#----------

			return subFeatureVectors, subLabels
		#--------------------

		#--------------------
		# 取得劃分資料的特徵索引
		def getSplitIndex(featureVectors, labels):

			#----------
			baseEntropy = getEntropy(labels)
			#----------
			#----------
			maxInfoGain = 0.0
			splitIndex = -1
			
			for featureIndex in range(len(featureVectors[0])):

				# 下邊這句實現抽取特徵i在資料集中的所有取值
				featureValues = set([featureVector[featureIndex] for featureVector in featureVectors])

				#-----
				featureEntropy = 0.0
				splitInfo = 0.0
				for featureValue in featureValues:
					subFeatureVectors, subLabels = getSubsets(featureVectors, labels, featureIndex, featureValue)

					# 計算對應資料的出現機率
					probability = float(len(subFeatureVectors)) / len(featureVectors)
					featureEntropy += probability * getEntropy(subLabels)

					# 計算對應資料的信息熵
					splitInfo -= probability * log(probability, 2)
				#-----
				#-----
				infoGain = 0
				if(splitInfo != 0): 
					# 計算信息增益
					gain = baseEntropy - featureEntropy
					# 計算信息增益率
					infoGain = gain / splitInfo
				
				if (infoGain > maxInfoGain):
					maxInfoGain = infoGain
					splitIndex = featureIndex
				#-----
			#----------

			return splitIndex
		#--------------------

		#--------------------
		# 判斷傳入的類別集中是否只有一種標籤，是，返回該標籤，代表遞迴終止
		classes = set(labels)
		if(len(classes) == 1): return labels[0]
		#--------------------
		#--------------------
		# 選定最佳的特徵劃分資料集，並且輸出這個特徵的索引
		splitIndex = getSplitIndex(featureVectors, labels)
		#--------------------
		#--------------------
		# 選定最佳特徵對應的標籤，輸出後刪除這個標籤
		# C45如同ID3，用過的特徵不再重複利用
		splitTitle = titles[splitIndex]
		titles.remove(splitTitle)
		#--------------------
		#--------------------
		# 初始化這一輪的決策樹
		decisionTree = {splitTitle:{}}
		# 收集最佳特徵下的特徵值，排序的目的是確保順序的一致性。
		featureValues = sorted(set([featureVector[splitIndex] for featureVector in featureVectors]))

		for featureValue in featureValues:
			# 收集最佳特徵值取下的子資料集與子標籤列表
			subFeatureVectors, subLabels = getSubsets(featureVectors, labels, splitIndex, featureValue)
			subTitles = titles[:]
			# 遞迴創建子樹
			decisionTree[splitTitle][featureValue] = self.creatDecisionTree(subTitles, subFeatureVectors, subLabels)
		#--------------------

		return decisionTree
	#------------------------------
	
	#------------------------------
	def load(self, samplePath):

		df = pd.read_csv(samplePath)  
		dataset = df.values[:,1:].tolist() 

		self.featureVectors = [example[:-1] for example in dataset]
		self.labels = [example[-1] for example in dataset]
		self.titles = df.columns.values[1:-1].tolist()

		return 
	#------------------------------

	#------------------------------
	def train(self):

		#--------------------
		titles = self.titles.copy()
		featureVectors = self.featureVectors.copy()
		labels = self.labels.copy()
		#--------------------

		self.decisionTree = self.creatDecisionTree(titles, featureVectors, labels)

		return
	#------------------------------

	#------------------------------	
	def classify(self, decisionTree, input):
	
		root = list(decisionTree.keys())[0]
		subTree = decisionTree[root]
		rootTitle = self.titles.index(root)

		#--------------------
		# 在決策樹中尋找輸入資料的對應的標籤
		for subRoot in subTree.keys():
			if (input[rootTitle] == subRoot):
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
	def paint(self):

		drawing.createPlot(self.decisionTree)
	#------------------------------
#---------------------------------------------



#---------------------------------------------
if __name__ == '__main__':

	c45 = C45()
	c45.load('weather.csv')
	c45.train()	
	c45.paint()
#---------------------------------------------