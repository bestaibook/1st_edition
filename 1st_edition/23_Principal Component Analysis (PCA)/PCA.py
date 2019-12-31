import numpy as np

from itertools import accumulate
from scipy.spatial import distance



#--------------------------------------------------
class PCA:
	
	#------------------------------
	# 分析組成成分	
	def setPCA(self, dataMat):

		#--------------------
		# 建立關聯性
		meanVals = np.mean(dataMat, axis=0) # 均值	
		meanRemoved = dataMat - meanVals	
		covMat = np.cov(meanRemoved, rowvar=0) # 協方差
		#--------------------

		#--------------------			
		# 計算特徵值、特徵向量 
		eigVals,eigVects=np.linalg.eig(np.mat(covMat))
		# 合併特徵值與特徵向量形成特徵對
		self.eigPairs = [(np.abs(eigVals[i]), eigVects[:,i]) for i in range(len(eigVals))]
		# 根據特徵值，將特徵對由大到小進行排序
		self.eigPairs.sort(key=lambda x: x[0], reverse=True)
		#--------------------

		return
	#------------------------------
	
	#------------------------------
	# 降低原始矩陣的維度
	def reductDimension(self, threshold):

		#--------------------
		#----------
		# 計算累積的比重
		eigVals = np.array([pair[0] for pair in self.eigPairs])
		accRates = list(accumulate((eigVals/np.sum(eigVals))))
		#----------
		#----------
		# 收集累積比重已超過閥值的項目索引
		passingIndexes = [accRates.index(ar)for ar in accRates if ar>threshold]
		#----------
		#----------
		# 最小的項目索引即為臨界索引
		size = np.min(passingIndexes)
		#----------
		#--------------------

		#--------------------
		eigVecs = [eigPair[1] for eigPair in self.eigPairs[:size+1]]
		self.p = np.mat(np.hstack(eigVecs))
		#--------------------

		return self.p
	#------------------------------

	#------------------------------
	# 轉換新的矩陣到以V為基底的向量空間中
	def transform(self,input):

		newInput = np.dot(input, self.p).round(2)

		return newInput
	#------------------------------

	#------------------------------
	# 在資料矩陣中尋找最接近的項目名稱
	def recommand(self, input, dataMat, nameMat):

		minIndex = distance.cdist(input, dataMat).argmin()

		name = nameMat[minIndex]

		return name
	#------------------------------
#--------------------------------------------------



#--------------------------------------------------
if __name__ == "__main__":
	
	dataMat = np.array([[6,3,7],
						[4,5,1]])
	
	pca = PCA()

	#--------------------
	pca.setPCA(dataMat)
	#--------------------

	#--------------------
	threshold = 0.9
	pca.reductDimension(threshold)
	#--------------------

	#--------------------
	newInput = pca.transform(dataMat)
	#--------------------

	print('Original: \n'+str(dataMat))
	print('Reduct Dimension:\n'+str(newInput))
#--------------------------------------------------