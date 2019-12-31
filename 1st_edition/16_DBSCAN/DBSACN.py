import sys
import math
import pylab as pl
import pandas as pd 

from collections import defaultdict,Counter



#--------------------------------------------------
class DBSCAN:

	#------------------------------
	def loadSamples(self, samplePath):

		df = pd.read_csv(samplePath)  

		self.samples = df.values[:,:].tolist() 
		self.titles = df.columns.values[:].tolist()
		self.features = [sample[:] for sample in self.samples]

		return
	#------------------------------

	#------------------------------
	# 計算兩點之間的距離
	def getDist(self, p1, p2):
		dist = round(math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
		return dist
	#------------------------------

	#------------------------------
	def clustering(self, eps = 10, minPoints = 3):
		
		#--------------------
		#[POINT 1] 	
		# 建立相鄰點網路(任兩點的距離小於eps即為相鄰點)
		neighborPoints = defaultdict(list)
	
		for selfIndex,selfPoint in enumerate(self.features):
			for neighborIndex,neighborPoint in enumerate(self.features):
				if (selfIndex < neighborIndex):
					if(self.getDist(selfPoint, neighborPoint) <= eps):
						neighborPoints[selfIndex].append(neighborIndex)
						neighborPoints[neighborIndex].append(selfIndex)
		#--------------------
		#--------------------	
		#[POINT 2] 	
		# 收集核心點(相鄰點數目大於minPoints即為核心點)		
		corePointIndices = []

		for pointIndex,surPointIndices in neighborPoints.items():
			if (len(surPointIndices)>=minPoints):
				corePointIndices.append(pointIndex)

		self.corePoints = [self.features[pointIndex] for pointIndex in corePointIndices] 
		#--------------------
		#--------------------
		#[POINT 3]	 
		# 收集邊界點(不是核心點但是鄰居包含了某個核心點即為邊界點)
		borderPointIndices = []

		for pointIndex,surPointIndices in neighborPoints.items():
			if (pointIndex not in corePointIndices):
				for surPointIndex in surPointIndices:
					if (surPointIndex in corePointIndices):
						borderPointIndices.append(pointIndex)
						break

		self.borderPoints = [self.features[pointIndex] for pointIndex in borderPointIndices]
		#--------------------
		#--------------------
		#[POINT 4]
		# 收集雜點(不是核心點並也不是邊界點即為雜點)
		noisePointIndices = []

		for pointIndex in range(len(self.features)):
			isSelfCorePoint = (pointIndex not in corePointIndices)
			isSelfBorderPoint = (pointIndex not in borderPointIndices)

			if (isSelfCorePoint==True and  isSelfBorderPoint==True):
				noisePointIndices.append(pointIndex)

		self.noisePoints = [self.features[pointIndex] for pointIndex in noisePointIndices]
		#--------------------
		#--------------------
		#[POINT 5]
		#----------
		#[POINT 5.1]
		# 合併相鄰的核心點形成群
		# 群的編號為核心點的索引
		self.cluster = [index for index in range(len(self.features))]

		for pointIndex,surPointIndices in neighborPoints.items():
			for surIndex in surPointIndices:			
				isForword = (pointIndex < surIndex)
				isSelfCorePoint = (pointIndex in corePointIndices)
				isNeighborCorePoint = (surIndex in corePointIndices)

				if (isForword==True and isSelfCorePoint==True and isNeighborCorePoint==True):
					for index in range(len(self.cluster)):
						if (self.cluster[index] == self.cluster[surIndex]):
							self.cluster[index] = self.cluster[pointIndex]				
		#----------
		#----------
		#[POINT 5.2]
		# 以核心點為群心吸收領域內的邊界點			
		for pointIndex,surPointIndices in neighborPoints.items():
			for surIndex in surPointIndices:
				isSelfBorderPoint = (pointIndex in borderPointIndices)
				isNeighborCorePoint = (surIndex in corePointIndices)

				if (isSelfBorderPoint==True and isNeighborCorePoint==True):
					self.cluster[pointIndex] = self.cluster[surIndex]
					break
		#----------
		#--------------------

		return
	#------------------------------

	#------------------------------
	def train(self):

		self.clustering()
	
		return 
	#------------------------------

	#------------------------------
	def predict(self, inputs):
		
		predictResults = list()

		#--------------------
		for input in inputs:
			labelIndex = None
			minDist = sys.maxsize

			for index in range(len(self.features)):
				p1 = input
				p2 = self.features[index]
				dist = self.getDist(p1, p2)

				if(dist<minDist):
					minDist = dist
					labelIndex = index
			
			predictResults.append( [input,self.cluster[labelIndex]])
		#--------------------

		return predictResults
	#------------------------------

	#------------------------------
	def paint(self, maxClusterNumber=3):

		finalCluster = Counter(self.cluster).most_common(maxClusterNumber)
		finalCluster = [onecount[0] for onecount in finalCluster]

		#--------------------
		clusterDict = dict()
		clusterNumber = 0

		for clusterNumber in range(maxClusterNumber):
			pointList = list()

			for index in range(len(self.features)):
				if (self.cluster[index]==finalCluster[clusterNumber]):
					pointList.append(self.features[index])

			clusterDict.update({str(clusterNumber):pointList})
		#--------------------
		#--------------------
		colorList = ['or','og','ob','oc','om','oy']

		#----------
		#plot Cluster
		for clusterNumber, pointList in clusterDict.items():

			x = [point[0] for point in pointList]
			y = [point[1] for point in pointList]

			colorIndex = int(float(clusterNumber)%len(colorList))
			color = colorList[colorIndex]

			pl.plot(x, y, color)
		#----------
		#----------
		#plot Noise
		x = [point[0] for point in self.noisePoints]
		y = [point[1] for point in self.noisePoints]

		pl.plot(x, y, 'ok')
		#----------
		#--------------------

		pl.show()

		return
	#------------------------------
#--------------------------------------------------



#--------------------------------------------------
if __name__ == '__main__':

	dbscan = DBSCAN()

	dbscan.loadSamples('coordinates.csv')
	dbscan.train()
	dbscan.paint()
#--------------------------------------------------