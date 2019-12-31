import sys
import itertools
import pandas as pd
from scipy.spatial import distance
from drawing import drawdendrogram



#--------------------------------------------------
class NODE:
	def __init__(self, data, left=None, right=None, dist=0.0, id=None):
		self.data = data	
		
		self.left = left
		self.right = right

		self.id = id
		self.distance = dist
#--------------------------------------------------

#--------------------------------------------------
class HIERARCHICAL:
	
	#------------------------------
	# 計算兩點之間的距離
	def getDist(self, pA, pB):

		dist = distance.euclidean(pA, pB)

		return dist
	#------------------------------

	#------------------------------
	def clustering(self, data):

		#--------------------
		# 聚合最相鄰的2個節點
		def agglomerative(nodes):

			minPair = [nodes[0],nodes[1]]
			minDist = sys.maxsize

			for i in range(len(nodes)):
				for j in range(i + 1, len(nodes)):
			
					dist = closest(nodes[i].data, nodes[j].data)
			
					if(dist < minDist):
						minPair = [nodes[i],nodes[j]]
						minDist = dist
			
			return minPair, minDist
		#--------------------

		#--------------------
		# 計算2個資料集合間彼此最相鄰的2個座標之距離
		def closest(pointsA, pointsB):

			minDist = sys.maxsize
			pairs = list(itertools.product(pointsA, pointsB))

			for pA, pB in pairs:
				d = self.getDist(pA, pB)
				if(d<minDist): 
					minDist = d

			return minDist
		#--------------------
				
		#--------------------
		id = -1

		tree = [NODE([data[i]], id=i) for i in range(len(data))]
	
		while(len(tree) > 1):
			#----------
			minPair, minDist = agglomerative(tree)
			#----------
			#----------
			newData = []
			newData.extend(minPair[0].data)
			newData.extend(minPair[1].data)
			#----------

			#----------
			newNode = NODE(	newData, 
							left=minPair[0], 
							right=minPair[1],
							id=id,
							dist=minDist	)

			id -= 1
			#----------

			#----------
			tree.remove(minPair[0])
			tree.remove(minPair[1])
			tree.append(newNode)
			#----------
		#--------------------

		return tree[0]
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
	def train(self):

		self.clusters = self.clustering(self.features)

		return
	#------------------------------

	#------------------------------
	def predict(self):

		return
	#------------------------------

	#------------------------------
	def paint(self):
		
		drawdendrogram(self.clusters, self.labels)

		return
	#------------------------------
#--------------------------------------------------



if __name__ == '__main__':


	hierarchical = HIERARCHICAL()
	hierarchical.load('coordinates.csv')
	hierarchical.train()
	hierarchical.paint()

	#labels = ['A','B','C','D','E','F']
	##data = [(11,11),(5,5),(7,7),(17,17),(18,18),(2,2)]
	#data = [(11),(5),(7),(17),(18),(2)]

	##data1 = [(7,7),(18,18)]
	##data2 = [(11,11),(5,5)]
	##closest(data1,data2)

	#clust = hcluster(data)
	#printclust(clust)
	#drawdendrogram(clust, labels)