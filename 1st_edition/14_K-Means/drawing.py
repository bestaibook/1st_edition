import numpy as np
import matplotlib.pyplot as plt



#------------------------------
def getColor(numberOfColor):

	colors = []

	#--------------------
	colorCycle = [	'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
					'#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'	]

	for no in range(numberOfColor):
		index = no % len(colorCycle)
		color = colorCycle[index]
		colors.append(color)
	#--------------------

	return colors
#------------------------------



#------------------------------
def drawCluster(numOfClusters, labels, data, centers):

	colors = getColor(numOfClusters)
	
	plt.figure()

	#--------------------
	for k, col in zip(range(numOfClusters), colors):
		#----------
		members = labels == k		
		plt.plot(data[members, 0], data[members, 1], 'o', markerfacecolor=col, marker='.', markersize=20)
		#----------
		#----------
		clusterCenter = centers[k]		
		plt.scatter(clusterCenter[0], clusterCenter[1], s=80, c=col, marker='*')
		#----------
	#--------------------

	plt.title('KMeans')    
	plt.grid(True)
	plt.show()
#------------------------------



if __name__ == '__main__':

	colors = getColor(15)
	print(colors)



# [REFERENCE]
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
# https://stackoverflow.com/questions/16006572/plotting-different-colors-in-matplotlib