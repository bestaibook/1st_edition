from operator import itemgetter
from collections import OrderedDict
from pandas.core.indexes.frozen import FrozenList



#--------------------------------------------------
class NODE:

	#------------------------------
	def __init__(self, nameValue, count, parentNode):

		self.name = nameValue
		self.count = count
		self.nodeLink = None
		self.parent = parentNode
		self.children = {} 
	#------------------------------

	#------------------------------
	def counter(self, count):
		self.count += count
	#------------------------------
#--------------------------------------------------



#--------------------------------------------------
class FPGROWTH:

	#------------------------------
	def __init__(self, minSup):

		self.minSup = minSup
	#------------------------------

	#------------------------------
	# 創建FP Growth Tree
	def createTree(self, dataSet, minSup=1):

		#--------------------
		# 更新項目索引表中各個項目的連結，這個函數會在項目重複出現時被呼叫。
		def updateHeader(nodeToTest, targetNode):

			while (nodeToTest.nodeLink != None):
				nodeToTest = nodeToTest.nodeLink

			nodeToTest.nodeLink = targetNode

			return
		#--------------------

		#--------------------
		def updateTree(items, inTree, headerTable, count):

			# 一個新的項目進來後首先要探討它是否已經是子節點
			# (因為FP Growth Tree的第1個節點是ROOT，所以用探討的對象是從子節點開始，而不是本身。)
			if (items[0] in inTree.children):			
				# 如果這個項目已經出現，在既有的的支持度上加1。
				inTree.children[items[0]].counter(count) 
			else:
				# 如果這是全新的項目，則將這個項目建立成為子節點。
				inTree.children[items[0]] = NODE(items[0], count, inTree)
			
				# 全新的項目成為子節點後，需要將項目索引表鏈結到這個新的節點上。		
				if (headerTable[items[0]][1] == None):
					# 如果項目索引表中沒有這個項目的連結，就直接從表中鏈結到這個新的項目。
					headerTable[items[0]][1] = inTree.children[items[0]]
				else:
					# 如果項目索引表中有這個項目的連結，就與既有的項目產生鏈結。
					updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
						
			# 如果還有高頻出現項目沒有填入FPGrowth Tree中，
			# 那麼以遞迴的方式繼續對剩餘的高頻出現進行填入的工作。
			if (len(items) > 1):
				updateTree(items[1::], inTree.children[items[0]], headerTable, count)

			return
		#--------------------

		#--------------------
		# 建立項目索引表
		headerTable = {}

		#----------
		# 計算每一個項目的出現頻率
		for trans in dataSet:
			for item in trans:
				headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
		#----------
		#----------	
		newHeaderTable={}

		for k in headerTable.keys():
			# 移除出現頻率低於閥值的項目
			if (headerTable[k] >= minSup): 
				newHeaderTable.update({k:headerTable[k]})

		headerTable=newHeaderTable
		#----------
		#--------------------

		#--------------------
		# 建立頻繁集
		# (排序只是為了確保順序一致，不排序也可以。)
		freqItemSet = sorted(set(headerTable.keys()))

		# 沒有項目則終止
		if (len(freqItemSet) == 0): 
			return None, None
		#--------------------
	
		#--------------------
		# 初始化項目索引表中每一個項目的連結
		for k in headerTable:
			headerTable[k] = [headerTable[k], None]

		# 初始化FPGrowth Tree
		rootNode = NODE('ROOT', 1, None)
		#--------------------

		#--------------------
		for tranSet, count in dataSet.items():

			# 儲存待分析的高頻出現項目
			localD = {}

			#----------
			# 過濾掉每一筆交易紀錄中不屬於頻繁集的項目，
			# 再建立FPGrowth Tree的過程中，只分析屬於頻繁集的高頻出現項目。
			for item in tranSet:
				if (item in freqItemSet):
					localD[item] = headerTable[item][0]
			#----------

			#----------
			if (len(localD) > 0):
				# 根據出現頻率對高頻出現項目進行排序
				orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
				# 將這筆交易的高頻出現項目填入至FPGrowth Tree中
				updateTree(orderedItems, rootNode, headerTable, count)
			#----------
		#--------------------

		return rootNode, headerTable
	#------------------------------

	#------------------------------
	def mineTree(self, inTree, headerTable, minSup, preFix, freqItemList):
	
		#--------------------
		# "向上" 走訪節點	(由樹葉往樹根方向)
		def ascendTree(leafNode, prefixPath): #ascends from leaf node to root

			#----------
			if (leafNode.parent != None):
				# 走訪過程中紀錄每一個節點名稱
				prefixPath.append(leafNode.name)
				ascendTree(leafNode.parent, prefixPath)
			#----------

			return
		#--------------------

		#--------------------
		# "橫向" 走訪項目
		def findPrefixPath(basePat, treeNode): #treeNode comes from header table
			condPats = {}

			#----------
			while treeNode != None:
				prefixPath = []

				# 每一個項目都向上走訪節點，走訪過程中經過的節點形成路徑。
				ascendTree(treeNode, prefixPath)
			
				if (len(prefixPath) > 1): 
					# 路徑起始節點的支持度即為該路徑的支持度。	
					condPats[FrozenList(prefixPath[1:])] = treeNode.count

				# 換到下一個項目繼續走訪
				treeNode = treeNode.nodeLink
			#----------

			return condPats
		#--------------------

		#--------------------
		# 取出項目索引表中的每一個項目，並且根據支持度由小到大進行排序。
		bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]

		# 走訪每一種項目(根據支持度由小到大進行走訪)
		for basePat in bigL:

			newFreqSet = preFix.copy()
			newFreqSet.add(basePat)

			print ('finalFrequent Item: ',newFreqSet)	#append to set
		
			freqItemList.append(newFreqSet)
			condPattBases = findPrefixPath(basePat, headerTable[basePat][1])		

			# 創建FPGrowth Tree，並取得創建後的項目索引表
			myCondTree, myHead = self.createTree(condPattBases, minSup)	

			# 項目索引表不為空代表這個項目下走訪的高頻出現項目
			# 之數量足以再繼續創建FPGrowth Tree
			if (myHead != None): 
				print ('head from conditional tree: ', myHead.keys())
				print ('conditional tree for: ',newFreqSet)

				# 繼續往下創建子FPGrowth Tree
				self.mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
		#--------------------

		return
	#------------------------------

	#------------------------------
	def paint(self, rootNode, ind=1):
		print ('  '*ind, rootNode.name, ' ', rootNode.count)
		for child in rootNode.children.values():
			self.paint(child, ind+1)
	#------------------------------

	#------------------------------
	def run(self, dataSet):
	
		self.rootNode, self.headerTable = self.createTree(dataSet, self.minSup)

		self.paint(self.rootNode)

		myFreqList = []
		preFix = set([])
		self.mineTree(self.rootNode, self.headerTable, self.minSup, preFix, myFreqList)
	
		return
	#------------------------------

	#------------------------------
	def load(self, path):

		#--------------------
		samples = list()
		f = open(path, 'rU')
		for l in f:
			sample = list(map(str.strip, l.split(',')))
			 
			# (排序只是為了確保順序一致，不排序也可以。)
			samples.append(sorted(sample))
		#--------------------

		#--------------------
		retDict = {}
		for trans in samples:
			retDict[FrozenList(trans)] = 1

		# (排序只是為了確保順序一致，不排序也可以。)
		retDict = OrderedDict(sorted(retDict.items(), key=itemgetter(0)))
		#--------------------

		return retDict
	#------------------------------

	#------------------------------
	def report(self, ruleList, feqItems):

		return
	#------------------------------
#--------------------------------------------------




if __name__ == '__main__':
	minSup = 2
	fpgrowth = FPGROWTH(minSup)

	data = fpgrowth.load('commodity.csv')

	fpgrowth.run(data)