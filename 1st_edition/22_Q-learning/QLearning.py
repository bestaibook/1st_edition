import numpy as np
import matplotlib.pyplot as plt

from matplotlib import collections  as mc



#--------------------------------------------------
class QLEARNING:

	#------------------------------
	def __init__(self):

		self.gamma = 0.9
		self.alpha = 1.0
		self.n_episodes = 1E3
		self.epsilon = 0.05
		self.random_state = np.random.RandomState(1999)

		self.paths = dict()
	#------------------------------

	#------------------------------
	def setEnviroment(self, rewards, coords):
		self.r = rewards
		self.q = np.zeros_like(self.r)
		self.n_states = len(self.r[0])
		self.n_actions = len(self.r[0])
		self.lastState = len(self.r[0])-1

		self.coords = coords
	#------------------------------

	#------------------------------
	def update_q(self, q, r, state, next_state, action, alpha, gamma):

		#--------------------
		rsa = r[state, action]
		qsa = q[state, action]
		#--------------------

		#--------------------
		# 執行Q Learning的價值函數
		new_q = qsa + alpha * (rsa + gamma * max(q[next_state, :]) - qsa)
		# 更新Q值
		q[state, action] = new_q
		#--------------------	

		return q
	#------------------------------

	#------------------------------
	def QL(self, r, q, n_states, n_actions, n_episodes, epsilon, alpha, gamma, random_state):

		times=0

		for e in range(int(n_episodes)):
			
			#--------------------
			# 隨機抽取一個起始狀態
			states = list(range(n_states))
			random_state.shuffle(states)
			current_state = states[0]
			goal = False
			#--------------------
			#--------------------
			if e % int(n_episodes / 10.) == 0 and e > 0:
				pass
			#--------------------
			#--------------------
			# 還沒有到達目的就持續執行
			while not goal:
				times=times+1
				# 從目前狀態的周邊挑選出可行之路
				valid_moves = r[current_state] >= 0

				#----------
				if random_state.rand() < epsilon: #希望隨機作決策
					#-----
					# 蒐集可行之路上所能進行的各種動作			
					actions = np.array(list(range(n_actions)))
					actions = actions[valid_moves == True]
					#-----
					if type(actions) is int: actions = [actions]
					#-----
					# 隨機從所能進行的各種動作中選擇一個作為之後要執行的動作。
					random_state.shuffle(actions)
					action = actions[0]
					#-----

					next_state = action

				else: # 希望根據經驗做決策
					if np.sum(q[current_state]) > 0:
						# 利用經驗值做決策
						action = np.argmax(q[current_state])
					else:
						#-----
						# 蒐集可行之路上所能進行的各種動作	
						actions = np.array(list(range(n_actions)))
						actions = actions[valid_moves == True]
						#-----
						#-----					
						# 隨機從所能進行的各種動作中選擇一個作為之後要執行的動作。
						random_state.shuffle(actions)
						action = actions[0]		
						#-----
						#						
					next_state = action
				#----------	
				#----------
				# 更新目前狀態的Q值
				self.q = self.update_q(self.q, self.r, current_state, next_state, action, alpha=alpha, gamma=gamma)
				#----------
				#----------
				# 如果獎勵為正代表已達終點，可以結束迴圈。
				reward = r[current_state, action]
				if reward > 1:
					goal = True
				#----------
				#----------
				# 還沒有走到終點就以下一個狀態為起點繼續更新Q值
				current_state = next_state
				#----------
			#--------------------

		return 
	#------------------------------

	#------------------------------
	def train(self):

		self.QL(	self.r, 
					self.q, 
					self.n_states, 
					self.n_actions, 
					self.n_episodes, 
					self.epsilon, 
					self.alpha, 
					self.gamma, 
					self.random_state	)

		return
	#------------------------------

	#------------------------------
	def show_traverse(self):

		q = self.q
		lastState = self.lastState
		
		#--------------------
		for i in range(len(q)):
			
			current_state = i
			self.paths.update({i:[current_state]})

			#----------
			n_steps = 0
			while current_state != lastState and n_steps < 20:
				next_state = np.argmax(q[current_state])
				current_state = next_state
				self.paths[i].append(next_state)
				n_steps = n_steps + 1
			#----------
		#--------------------

		return
	#------------------------------

	#------------------------------
	def paint(self):

		#--------------------
		def drawLineSegment(img, lines, width=5, color=(1,0,0,1.0)):

			lc = mc.LineCollection(lines, linewidths=width, color=color)
			
			img.add_collection(lc)
			img.autoscale()
			img.margins(0.1)			
		#--------------------

		#--------------------
		def drawNodeName(coords):

			verticalalignment = 'top'
			horizontalalignment = 'left'

			for i in range(len(coords)):
				x = coords[i][0]
				y = coords[i][1]
				name = str(i)

				plt.text(x, y, name, size=10,
						 horizontalalignment=horizontalalignment,
						 verticalalignment=verticalalignment,
						 bbox=dict(	facecolor='w', 
									edgecolor=plt.cm.nipy_spectral(float(len(coords))),
									alpha=.6))
		#--------------------

		#--------------------
		def toLineSegments(path):

			#----------
			vertexs = list()

			for index in range(0,len(path)-1):
				vertexs.append((path[index], path[index+1]))

			segments = np.array(sorted(list(set(vertexs))))
			#----------
			#----------
			lineSegment = list()

			for segment in segments:
				start = coords[segment[0]]
				end = coords[segment[1]]

				lineSegment.append([start,end])
			#----------

			return lineSegment
		#--------------------

		coords = self.coords

		#--------------------		
		lineSegments = dict()

		for start, path in self.paths.items():
			lineSegment = toLineSegments(path)
			lineSegments.update({start:lineSegment})
		#--------------------
		#--------------------
		img = plt.axes([0., 0., 1., 1.])

		drawLineSegment(img, lineSegments[0], width=1, color=(1, 0, 0, 1.0))
		drawLineSegment(img, lineSegments[1], width=5, color=(0, 1, 0, 1.0))
		drawLineSegment(img, lineSegments[2], width=3, color=(0, 0, 1, 1.0))
		drawNodeName(coords)

		plt.show()
		#--------------------
	#------------------------------

	#------------------------------
	def printFile(self, file, content, writeType='a'):

		if writeType == 'w':
			with open(file, 'w') as f:
				f.write('\n\n') 
				f.write(str(content)) 
				f.close
		else:
			with open(file, 'a+') as f:
				f.write('\n\n') 
				f.write(str(content)) 
				f.close
	#------------------------------
#--------------------------------------------------





rewards = np.array([	[-1,  0, -1, -1, -1, -1],
						[ 0, -1,  0, -1, -1, -1],
						[-1,  0, -1, -1, -1, 10],
						[-1, -1, -1, -1,  0, -1],
						[-1,  0, -1,  0, -1, 10],
						[-1, -1,  0, -1,  0, 10]]).astype("float32")


coords = np.array([	[0, 5],	[1, 5],	[2, 5],
					[0, 1],	[1, 1],	[2, 1]])

qLearning = QLEARNING()
qLearning.setEnviroment(rewards, coords)
qLearning.train()
qLearning.show_traverse()
qLearning.paint()

print(qLearning.q)