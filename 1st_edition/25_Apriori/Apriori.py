from itertools import chain, combinations



#--------------------------------------------------
class APRIORI:

	#------------------------------
	def __init__(self, minSupport=0.5, minConfidence=0.5):

		self.minSupport = minSupport
		self.minConfidence = minConfidence
	#------------------------------

	#------------------------------
	# 取出每一筆交易紀錄中的項目
	def getItems(self, data):

		itemSet = set()
		transactionList = list()

		#--------------------
		for row in data:
			transactionList.append(frozenset(row))
			for item in row:
				if (item):
					itemSet.add(frozenset([item]))
		#--------------------

		return itemSet, transactionList
	#------------------------------

	#------------------------------
	# 取得出現頻率高的項目
	def getFrequencyItems(self, transactionList, itemSet, minSupport):

		#--------------------
		# 產生候選集:創造各種長度的的集合
		def joinset(itemSet, length):

			joinSet = set()

			#----------
			for itemA in itemSet:
				for itemB in itemSet:
					if (len(itemA.union(itemB)) == length):
						joinSet.add(itemA.union(itemB))
			#----------

			return joinSet
		#--------------------
		
		#--------------------
		# 統計:計算每一種組合的support
		def getSupport(transactionList, itemSet, minSupport):
	
			transactionLength = len(transactionList)

			#----------
			#計算support
			supportList = list()
			for item in itemSet:
				#-----
				#判斷這一個子集合是否存在於其他集合中，若存在則紀錄數量
				sumOfSet = float(0)
				for row in transactionList :
					if (item.issubset(row)): sumOfSet = sumOfSet + 1.0
				#-----
				#-----
				# 一個集合存在其他集合中的數量除以集合總數即為support
				support = float(sumOfSet)/transactionLength
				supportList.append((item,support))
				#-----
			#----------

			#----------
			#篩選掉support不足的子集合
			supports = dict()
			for item, support in supportList:
				if (support >= minSupport):
					supports.update({item:support})
			#----------
	
			return supports
		#--------------------

		#--------------------
		feqItems = dict()

		setLength = 1
		while True:
			#----------
			if (setLength > 1): 
				itemSet = joinset(qualityItems, setLength)
			#----------
			#----------
			# 統計支持度
			qualityItems = getSupport(transactionList, itemSet, minSupport)
			#----------
			#----------
			# 沒有任何項目組合可以大於minSupport就結束
			if (not qualityItems): 
				break
			#----------
			#----------
			feqItems.update(qualityItems)
			setLength += 1
			#----------
		#--------------------


		return feqItems    
	#------------------------------

	#------------------------------
	# 根據confidence建立規則
	def getRules(self, feqItems, minConfidence):

		#--------------------
		# 產生候選集:
		# 在特定長度的集合中創造各種可能的組合
		def subsets(itemSet):

			iterableList = list()

			#----------
			for i, a in enumerate(itemSet):
				# 創造各種可能的組合
				iterableList.append(combinations(itemSet, i + 1))
			#----------

			subset = chain(*iterableList)

			return subset

		# 函數使用案例：
		# enumerate(('foo', 'bar', 'baz')) => (0, foo),(1, bar),(2, baz)
		# combinations([1,2,3], 2) => (1, 2), (1, 3), (2, 3)
		# chain('ABC', 'DEF') => A B C D E F
		#--------------------

		#--------------------
		ruleList = list()

		for item, support in feqItems.items():
			if (len(item) > 1):
				for setA in subsets(item):
					setB = item.difference(setA)
					if (setB):
						# 取得項目A
						setA = frozenset(setA)
						# 取得項目A與項目B
						setAB = setA | setB
						# 取得項目A發生的機率，也就是取得項目A的support
						probA = feqItems[setA] 
						# 取得項目A與項目B同時發生的機率，也就是取得項目A與項目B的support
						probAB = feqItems[setAB] 
						# 計算confidence，也就是項目A發生的前提下項目A與項目B同時發生的條件機率
						confidence = float(probAB) / probA

						# 大於minConfidence的項目留下來形成規則
						if (confidence >= minConfidence):
							ruleList.append((setA, setB, confidence))
		#--------------------

		return ruleList
	#------------------------------

	#------------------------------
	def run(self, data):
	
		# 取出交易中的項目
		itemSet, transactionList = self.getItems(data)	
		# 取出大於minSupport的項目
		feqItems = self.getFrequencyItems(transactionList, itemSet, self.minSupport)	
		# 建立規則
		ruleList = self.getRules(feqItems, self.minConfidence)
	
		return ruleList, feqItems
	#------------------------------

	#------------------------------
	def load(self, path):
		data = list()
		f = open(path, 'rU')
		for l in f:
			row = list(map(str.strip, l.split(',')))
			yield row
	#------------------------------

	#------------------------------
	def report(self, ruleList, feqItems):

		print('[Frequent Itemset]')
		frequentItemset = sorted(feqItems.items(), key=lambda x: x[1])
		for item, support in frequentItemset:
			print('[I] {} : {}'.format(tuple(item), round(support, 4)))
	
		print('------------------------------')

		print('[Rules]')
		ruleItemSet = sorted(ruleList, key=lambda x: x[2])
		for A, B, confidence in ruleItemSet:
			print('[R] {} => {} : {}'.format(tuple(A), tuple(B), round(confidence, 4))) 
	#------------------------------
#--------------------------------------------------



#--------------------------------------------------
if __name__ == '__main__':

	#----------
	minSupport = 0.3
	minConfidence = 0.3
	apriori = APRIORI(minSupport, minConfidence)
	#----------

	#----------
	data = apriori.load('commodity.csv')
	#----------
	#----------
	ruleList, feqItems = apriori.run(data)
	#----------
	#----------
	apriori.report(ruleList, feqItems)
	#----------
#--------------------------------------------------