import numpy as np
import pandas as pd
import drawing as dw
from scipy.spatial import distance



#--------------------------------------------------
class KMEANS:

	#------------------------------
	def __init__(self, maxIters=10):
	
		self.maxIters = maxIters
	#------------------------------

	#------------------------------
	# 計算兩點之間的距離
	def getDist(self, pA, pB):

		dist = distance.euclidean(pA, pB)

		return dist
	#------------------------------

	#------------------------------
	def clustering(self, data, K, maxIters=3):
		
		#--------------------
		# 隨機選取中心點
		randomIndices = np.random.choice(np.arange(len(data)), K)
		# 此處為了便於說明先將中心點固定
		self.centroidList = [[5, 3], [9, 8], [8, 9]]		
		#--------------------		

		#--------------------
		self.clusters = [0] * len(data)

		for i in range(maxIters):

			#----------
			# 指派資料所屬的分群中心			
			newClusterList = list()

			for d in data:
				#-----
				# 配置每一筆資料到距離自己最近的中心點
				distList = list()
				for c in self.centroidList:
					distList.append(self.getDist(d, c))
				index_min = np.argmin(distList)
				#-----

				newClusterList.append(index_min)
			#----------						

			#----------
			# 進行分群，配置到相同中心的資料視為一群
			newCentroidList = list()

			for k in range(K):
				#-----
				clusterList = list()
				for d_i in range(len(data)):
					if(newClusterList[d_i]==k):
						clusterList.append(data[d_i])
				#-----

				if(len(clusterList)==0): continue

				# 重新計算分群中心
				newCentroidList.append(list(np.mean(clusterList, axis=0).round(2)))
			#----------

			self.clusters = np.array(newClusterList)
			self.centroidList = np.array(newCentroidList)
			self.clusterCentroids = self.centroidList[self.clusters]
		#--------------------

		return 
	#------------------------------

	#------------------------------
	def load(self, path):

		df = pd.read_csv(path)  

		self.samples = df.values[:,:].tolist()
		self.features = [sample[:-1] for sample in self.samples]
		self.labels = [sample[-1] for sample in self.samples]	

		return
	#------------------------------

	#------------------------------
	def train(self, K):

		self.clustering(self.samples, K)

		return
	#------------------------------

	#------------------------------
	def predict(self):

		return
	#------------------------------

	#------------------------------
	def paint(self):

		n_clusters = len(set(self.clusters))
		k_means_labels = np.array(self.clusters)
		X = np.array(self.samples)
		
		dw.drawCluster(n_clusters, k_means_labels, X, self.centroidList)
		
		return
	#------------------------------
#--------------------------------------------------



#--------------------------------------------------
if __name__ == '__main__':

	kmeans = KMEANS()
	kmeans.load('coordinates.csv')
	kmeans.train(3)
	kmeans.paint()
#--------------------------------------------------