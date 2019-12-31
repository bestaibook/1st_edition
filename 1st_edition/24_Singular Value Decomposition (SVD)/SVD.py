import numpy as np
from itertools import accumulate
from scipy.spatial import distance



#--------------------------------------------------
class SVD:
	
	#------------------------------
	# 利用SVD分解矩陣
	def setSVD(self, dataMat):

		#--------------------		
		u,s,vh = np.linalg.svd(dataMat)
		#--------------------
		#--------------------
		self.singularValues = s
		#--------------------
		#--------------------
		self.U = u.round(2)
		#--------------------
		#--------------------
		sigma = np.zeros(dataMat.shape)

		for index in range(s.shape[0]):
			sigma[index][index] = s[index]

		self.S = sigma.round(2)
		#--------------------
		#--------------------
		self.V = np.transpose(vh).round(2)	
		self.VT = vh.round(2)
		#--------------------

		return
	#------------------------------
	
	#------------------------------
	# 降低原始矩陣的維度
	def reductDimension(self, threshold):

		#--------------------
		#----------
		# 計算累積的比重
		accRates = list(accumulate((self.singularValues/np.sum(self.singularValues))))
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
		#----------
		# 降低U矩陣的維度
		self.dU = self.U[:,:size]
		#----------
		#----------
		# 降低sigma矩陣的維度
		self.dS = self.S[:size,:size]
		#----------	
		#----------
		# 降低VT矩陣的維度
		self.dVT = self.VT[:size,:] 
		#----------
		#----------
		# 對VT矩陣進行轉置得到降低維度後的V矩陣
		self.dV = np.transpose(self.dVT)
		#----------
		#--------------------

		return
	#------------------------------

	#------------------------------
	# 轉換新的矩陣到以V為基底的向量空間中
	def transform(self,input):

		newInput = np.dot(input, self.dV).astype(np.int)

		return newInput
	#------------------------------

	#------------------------------
	# 在資料矩陣中尋找最接近的項目名稱
	def recommand(self, input, dataMat, nameMat):

		minIndex = distance.cdist(input, dataMat).argmin()

		name = nameMat[minIndex]

		return name
	#------------------------------

	#------------------------------
	def printMatrix(self, U, S, V, VT):
		print('U:')
		print(str(U))	
		print('S:')
		print(str(S))
		print('V:')
		print(str(V))
		print('VT:')
		print(str(VT))	
	#------------------------------


#--------------------------------------------------


if __name__ == "__main__":

	nameMat = np.array(	[	"Isaac",
							"Ray",
							"Frank",
							"Tom",
							"Debra",
							"Kai",
							"Jack"	]	)

	dataMat = np.array(	[	[5,3,1,2,1],
							[4,4,1,0,1],
							[4,5,0,1,1],
							[1,1,1,0,0],
							[0,2,4,5,3],
							[0,1,5,3,4],
							[1,1,2,5,4]]	)
	
	svd = SVD()

	#--------------------
	svd.setSVD(dataMat)
	#--------------------

	#--------------------
	threshold = 0.9
	svd.reductDimension(threshold)
	#--------------------

	#--------------------
	input = np.array([[3,6,0,1,1]])

	#----------
	name = svd.recommand(input, dataMat, nameMat)
	#----------
	#----------
	newInput = svd.transform(input)
	newDataMat = svd.transform(dataMat)

	newName = svd.recommand(newInput, newDataMat, nameMat)
	#----------
	#--------------------

	#--------------------
	print('Recommand: ' + name)
	print('New Recommand: ' + newName)
	#--------------------