import pandas as pd
import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import *



#--------------------------------------------------
class LINEARREGRESSION:

	#------------------------------
	def ordinaryLeastSquares(self, xmat, ymat):
		
		#--------------------
		# 計算回歸係數
		def getRegressionWeights(xmat, ymat):
			weights = (xmat.T*(xmat)).I*(xmat.T*(ymat.T))

			return weights
		#--------------------

		m,n = shape(xmat)
		ypred = zeros(m)

		# 取得回歸係數
		weights = getRegressionWeights(xmat, ymat)
		
		for i in range(m):
			# 將每一筆資料乘以回歸係數，得到預測值。			
			ypred[i] = xmat[i]* weights

		return ypred
    #------------------------------
	
	#------------------------------
	#載入資料
	def load(self, path):

		self.data = pd.read_csv(path) 

		#--------------------
		#----------
		featuresVectors = mat(self.data.X)

		featuresLength = shape(featuresVectors)[1]
		one = mat(ones(featuresLength))
		self.featureMatrix = hstack((one.T, featuresVectors.T))
		#----------
		#----------
		self.labelMatrix = mat(self.data.Y)
		#----------
		#--------------------

		return
	#------------------------------

	#------------------------------
	def train(self):

		self.ypred = self.ordinaryLeastSquares(self.featureMatrix, self.labelMatrix)

		return
	#------------------------------

	#------------------------------
	def predict(self):

		return
	#------------------------------

	#------------------------------
	# 繪製資料座標，以及回歸線
	def paint(self):

		sortIndex = self.featureMatrix[:,1].argsort(0)
		xsort = self.featureMatrix[sortIndex][:,0]

		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(xsort[:,1], self.ypred[sortIndex], color = 'red', linewidth=5)
		ax.scatter(array(self.data.X), array(self.data.Y), color='black')

		plt.xlabel('Feature')
		plt.ylabel('Label')
		plt.show()

		return
	#------------------------------
#--------------------------------------------------



linearRegression = LINEARREGRESSION()
linearRegression.load('coordinate.csv')
linearRegression.train()
linearRegression.paint()