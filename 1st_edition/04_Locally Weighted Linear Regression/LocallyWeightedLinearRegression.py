
import pandas as pd
import matplotlib.pyplot as plt

from numpy import *
from numpy.linalg import *



#--------------------------------------------------
class LOCALLYWEIGHTEDLINEARREGRESSION:

	#------------------------------
	def lwlr(self, xmat, ymat, k):

		#--------------------
		# point為中心，利用高斯曲線計算其他點的權重
		# 距離自己進的點權重高，遠者權重低。
		def kernel(point, xmat, k):
			m,n = shape(xmat)
			weights = mat(eye((m)))

			for j in range(m):
				diff = point - xmat[j]
				weights[j,j] = exp(diff*diff.T/(-2.0*k**2))

			return weights 
		#--------------------
		#--------------------
		# 取得迴歸係數
		def getRegressionWeights(point, xmat, ymat, k):
			wei = kernel(point,xmat,k)
			# 在公式中參入了權重，造就了親疏有別的效果。
			w = (xmat.T*(wei*xmat)).I*(xmat.T*(wei*ymat.T))

			return w
		#--------------------

		m,n = shape(xmat)
		ypred = zeros(m)

		for i in range(m):
			ypred[i] = xmat[i]*getRegressionWeights(xmat[i], xmat, ymat, k)

		return ypred
    #------------------------------
	
	#------------------------------
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

		self.ypred = self.lwlr(self.featureMatrix, self.labelMatrix, 0.5)

		return
	#------------------------------

	#------------------------------
	def predict(self):

		return
	#------------------------------

	#------------------------------
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



lwlr = LOCALLYWEIGHTEDLINEARREGRESSION()
lwlr.load('coordinate.csv')
lwlr.train()
lwlr.paint()