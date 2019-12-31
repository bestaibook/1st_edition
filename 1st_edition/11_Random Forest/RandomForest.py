import ast
import drawing
import numpy as np
import pandas as pd 

from collections import Counter



#--------------------------------------------------
class NODE:
	def __init__(self, name=None):

		self.name = name
		self.leftVectors, self.rightVectors = list(), list()
		self.leftLabels, self.rightLabels = list(), list()
		self.leftValues, self.rightValues = set(), set()
#--------------------------------------------------

#--------------------------------------------------
class RANDOMFOREST:

	#------------------------------
	def __init__(self, sampleRatio, forestSize, treeDepth, nodeSize, nSelected):

		self.sampleRatio = sampleRatio
		self.forestSize = forestSize
		self.treeDepth = treeDepth
		self.nodeSize = nodeSize
		self.nSelected = nSelected

		return
	#------------------------------

	#------------------------------
	# 建立決策樹
	def buildTree(self, titles, featureVectors, labels, treeDepth, nodeSize, nSelected):
		
		#--------------------
		# 隨機選擇索引值，選出後 "不放回"
		def randomSelectIndices(data, nSelected):
			
			#----------
			length = len(data)		
			#----------
			
			#----------
			indices = []
			if(length>=nSelected):
				#indices = np.random.sample(range(length), nSelected)
				indices = np.random.choice(range(length), nSelected)
			#----------

			return sorted(indices)
		#--------------------

		#--------------------
		# 計算Gini係數
		def getGini(labels):
			# 統計資料型別及每一種資料型別對應的數量
			labelCounts = dict(Counter(labels))
			score = 0.0
			for label, count in labelCounts.items():
				probability = float(count) / len(labels)
				score += probability * probability

			gini = 1.0 - score

			return gini
		#--------------------

		#--------------------
		# 計算信息增益值
		def giniGain(leftLabels, rightLabels):

			labels = leftLabels+rightLabels

			#----------
			# 計算所有資料的Gini
			baseGini = getGini(labels)	
			# 計算左左半部資料的Gini
			leftGini = getGini(leftLabels)
			# 計算右半部資料的Gini
			rightGini = getGini(rightLabels)
			#----------

			#----------
			leftProbability = float(len(leftLabels)) / len(labels)
			rightProbability = 1.0 - leftProbability					
			expectation = leftProbability*leftGini + rightProbability*rightGini		
			gain = baseGini - expectation
			#----------		
			

			return gain
		#--------------------

		#--------------------
		# 建立樹葉節點
		def getLeaf(results):

			# 因為不一定能將所有的資料分類完，
			# 所以一旦遇到此種情況，就對所有可能的結果採取投票，
			# 得票數最多者作為最終的label
			leaf = max(set(results), key=results.count)

			# 讀者可以嘗試使用這一行，會顯示投票前各種可能結果的比重
			#leaf = str(dict(Counter(results)))

			return leaf
		#--------------------

		#--------------------
		# 根據特徵值分割資料集		
		def getNode(titles, featureVectors, labels, featureIndex, targetValue):
	
			node = NODE(titles[featureIndex])

			#----------
			# 找尋這個特徵下可以讓Gini Gain最大的特徵值
			for vectorIndex in range(len(featureVectors)):

				# 取得對應的特徵值
				featureValue = featureVectors[vectorIndex][featureIndex]

				# 分成2群，相等的特徵值放在左子樹，否則放在右子樹
				if (featureValue == targetValue):
					node.leftVectors.append(featureVectors[vectorIndex])
					node.leftLabels.append(labels[vectorIndex])
					node.leftValues.add(featureValue)
				else:
					node.rightVectors.append(featureVectors[vectorIndex])
					node.rightLabels.append(labels[vectorIndex])
					node.rightValues.add(featureValue)
			#----------

			return node
		#--------------------

		#--------------------
		# 隨機選擇最佳的分割特徵
		def getRoot(titles, featureVectors, labels, nSelected):
	
			maxGain = 0.0
			root = NODE()	

			#----------
			# 第2次隨機:隨機選擇feature作為分割點
			# 當選出的分割點數量達到預設上限時就結束選擇
			indices = randomSelectIndices(featureVectors[0], nSelected)
			#----------
			#----------
			# 走訪每一個隨機選出的feature分割點
			# 與CART的不同之處在於:而Random Forest只會隨機選取幾個來試試
			for featureIndex in indices:

				# 取得這個特徵下所有的特徵值，並且以set將它們的種類變成唯一
				# (使用sorted是為了讓每次輸出的順序一致)
				featureValues = sorted(set([vector[featureIndex] for vector in featureVectors]))

				# 走訪每一個feature value
				for featureValue in featureValues:
					
					# 利用value來將輸入資料分成2群，然後利用gini係數計算混亂程度
					node = getNode(titles, featureVectors, labels, featureIndex, featureValue)

					#計算gini係數用於判別混亂的程度
					gain = giniGain(node.leftLabels, node.rightLabels)

					if (gain>maxGain and len(node.leftLabels)>0 and len(node.rightLabels)>0):
						maxGain = gain
						root = node		
		
			# 如果maxGain為0代表信息增益為0，只能分成1類，我們將這類資料放在左子樹中
			if(maxGain==0.0): root.leftLabels=labels
			#----------

			return root
		#--------------------

		#--------------------
		# 建立決策樹
		def getTree(titles, root, treeDepth, nodeSize, nSelected, depth):
		
			rootName = root.name
			left = root.leftVectors
			right = root.rightVectors
			leftBranchName = str(root.leftValues)
			rightBranchName = str(root.rightValues)

			decisionTree = {rootName:{}}
	
			#----------
			# Case 1: 未分割需要終止
			if (not left or not right):
				leaf = getLeaf(root.leftLabels+root.rightLabels)

				decisionTree = {str(leaf):{}}

				return leaf
			#----------
			#----------
			# Case 2: 超過深度需要終止
			if (depth >= treeDepth):
				leftLeaf = getLeaf(root.leftLabels)
				rightLeaf = getLeaf(root.rightLabels)

				decisionTree[rootName][leftBranchName] = leftLeaf
				decisionTree[rootName][rightBranchName] = rightLeaf
		
				return decisionTree
			#----------
			#----------
			# Case 3: 左子樹中擁有的資料數量還未超過上限，則繼續分割左子樹
			if (len(left) <= nodeSize):
				leaf = getLeaf(root.leftLabels)

				decisionTree[rootName][leftBranchName] = leaf
			else:
				leftSubTreeRoot = getRoot(titles, root.leftVectors, root.leftLabels, nSelected)
				leftSubTree = getTree(titles, leftSubTreeRoot, treeDepth, nodeSize, nSelected, depth+1)

				decisionTree[rootName][leftBranchName] = leftSubTree
			#----------
			#----------
			# Case 4: 右子樹中擁有的資料數量還未超過上限，則繼續分割右子樹
			if (len(right) <= nodeSize):
				leaf = getLeaf(root.rightLabels)

				decisionTree[rootName][rightBranchName] = leaf
			else:
				rightSubTreeRoot = getRoot(titles, root.rightVectors, root.rightLabels, nSelected)
				rightSubTree = getTree(titles, rightSubTreeRoot, treeDepth, nodeSize, nSelected, depth+1)

				decisionTree[rootName][rightBranchName] = rightSubTree
			#----------

			return decisionTree
		#--------------------

		root = getRoot(titles, featureVectors, labels, nSelected)
		tree = getTree(titles, root, treeDepth, nodeSize, nSelected, 1)

		return tree
	#------------------------------

	#------------------------------
	# 建立森林
	def buildForest(self, titles, featureVectors, labels, sampleRatio, forestSize, treeDepth, nodeSize, nSelected):

		#--------------------
		# 隨機選擇樣本，選出後 "放回"
		def randomSelectSample(dataset, ratio):

			#----------
			length = len(dataset)		
			#----------
			#----------
			sampleSize = round(len(dataset) * ratio)
			#----------
			#----------
			indices = []
			if(length>nSelected):
				indices = np.random.choice(range(length), sampleSize)
			#----------

			return sorted(indices)
		#--------------------

		self.forest = {'FOREST':{}}

		for i in range(forestSize):
			np.random.seed(i*1000)

			#----------
			# 隨機選取資料樣本建立決策樹
			indices = randomSelectSample(featureVectors, sampleRatio)
			subFeatureVectors = [featureVectors[index] for index in indices]
			subLabels = [labels[index] for index in indices]
			#----------			

			tree = self.buildTree(	titles, 
									subFeatureVectors, 
									subLabels, 
									treeDepth, 
									nodeSize, 
									nSelected	)

			self.forest['FOREST']['tree'+str(i+1)] = tree			

		return
	#------------------------------

	#------------------------------	
	# 利用每一棵tree進行預測，並收集結果
	def baggingPredict(self, input):


		#--------------------
		def classify(decisionTree, input):
			classLabel = None
			root = list(decisionTree.keys())[0]
			subTree = decisionTree[root]
			rootTitleIndex = self.titles.index(root)

			#----------
			# 遞迴在決策樹中尋找輸入資料的對應的類別
			for subRoot in subTree.keys():
				branchElements = ast.literal_eval(subRoot)
				inputElement = input[rootTitleIndex]

				if (inputElement in branchElements):
					if type(subTree[subRoot]).__name__ == 'dict':
						classLabel = classify(subTree[subRoot], input)
					else: classLabel = subTree[subRoot]
			#----------

			return classLabel
		#--------------------

		#--------------------
		# 取出forest中的每一棵tree
		trees = list()
		for i in range(forestSize):
			tree = self.forest['FOREST']['tree'+str(i+1)] 
			trees.append(tree)
		#--------------------

		#--------------------
		# 輸入資料進入每一棵樹中，紀錄預測結果
		predictions = [classify(tree, input) for tree in trees]
		#--------------------

		#--------------------
		# 投票選出最多的結果
		selected = max(set(predictions), key=predictions.count)
		#--------------------

		return selected
	#------------------------------



	#==============================
	# Framework
	#------------------------------
	def load(self, samplePath):

		df = pd.read_csv(samplePath)  

		self.samples = df.values[:,:].tolist() 
		self.titles = df.columns.values[:-1].tolist()
		self.featureVectors = [sample[:-1] for sample in self.samples]
		self.labels = [sample[-1] for sample in self.samples]	
		self.classes = self.labels

		return
	#------------------------------
	
	#------------------------------
	def train(self):
		
		self.buildForest(	self.titles, 
							self.featureVectors, 
							self.labels, 
							self.sampleRatio,
							self.forestSize, 
							self.treeDepth, 
							self.nodeSize, 
							self.nSelected	)

		return 
	#------------------------------

	#------------------------------
	def predict(self, inputs):
		
		predictResults = list()
		
		#--------------------
		for input in inputs:
			predictLabel = self.baggingPredict(input)
			predictResults.append([input,predictLabel])
		#--------------------

		return predictResults
	#------------------------------

	#------------------------------
	def paint(self, tree=None):
		if(tree==None):
			drawing.createPlot(self.forest)
		else:
			drawing.createPlot(tree)
	#------------------------------
	#==============================
#--------------------------------------------------



if __name__ == '__main__':
	
	sampleRatio = 0.8
	forestSize = 3
	treeDepth = 3
	nodeSize = 1
	nSelected = 2

	randomforest = RANDOMFOREST(sampleRatio, forestSize, treeDepth, nodeSize, nSelected)
	randomforest.load('admission.csv')
	randomforest.train()
	randomforest.paint()

	#--------------------
	tests = [
				['male','married','pop','Taipei']		#Accept
			]

	predictResults = randomforest.predict(tests)

	for result in predictResults:
		print(result)
	#--------------------