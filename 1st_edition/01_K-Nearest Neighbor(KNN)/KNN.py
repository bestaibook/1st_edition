import numpy as np
from collections import Counter



#----------------------------------------
class KNN:

	#------------------------------
	def __init__(self, k):

		self.k = k
	#------------------------------
	
	#------------------------------
	# 訓練
	def train(self, samples, labels):

		self.samples = samples
		self.labels = labels

		return
	#------------------------------
	
	#------------------------------
	# 篩選鄰居
	def getNeighbors(self, input):	

		#--------------------
		def distance(locA, locB):

			locA = np.array(locA) 
			locB = np.array(locB)

			# 計算歐式距離
			dist = np.linalg.norm(locA - locB)

			return dist
		#--------------------

		#--------------------
		distances = list()

		for index in range(len(self.samples)):
			# 計算相對於input sample的距離
			dist = distance(input, self.samples[index])
			distances.append((self.samples[index], dist, self.labels[index]))

		# 依據相對距離將每一筆資料由近到遠進行排序
		distances.sort(key=lambda x: x[1])

		# 選取K個最相近的sample做為鄰居
		neighbors = distances[:self.k]
		#--------------------

		return neighbors
	#------------------------------

	#------------------------------
	# 投票
	def vote(self, neighbors):

		classCounter = Counter()

		#--------------------
		for neighbor in neighbors:
			# neighbor的第2個元素為Label
			# 對相同label的neighbor投下一票
			classCounter[neighbor[2]] += 1
		#--------------------

		# 票數最多的Label即為相近的sample
		# 輸出它的Label成為預測結果
		selected = classCounter.most_common(1)[0][0]

		return selected
	#------------------------------

	#------------------------------
	# 預測
	def predict(self, inputs):		
		
		predictResults = list()

		#--------------------
		for input in inputs:
			# 統計鄰居
			neighbors = self.getNeighbors(input)
			# 投票
			result= self.vote(neighbors)
			# 紀錄當選者(預測結果)
			predictResults.append(result)
		#--------------------

		return predictResults
	#------------------------------
#----------------------------------------



if __name__ == '__main__':

	samples=[]
	samples.append((85,93,40,25))
	samples.append((90,88,30,33))
	samples.append((15,20,92,89))
	samples.append((17,29,91,93))
	samples.append((35,94,88,34))
	samples.append((28,92,85,15))


	labels = []
	labels.append('Science')
	labels.append('Science')
	labels.append('Art')
	labels.append('Art')
	labels.append('Business')
	labels.append('Business')


	knn = KNN(3)

	knn.train(samples, labels)

	inputs = [(79,77,50,65)]
	predictResults = knn.predict(inputs)

	print(predictResults)