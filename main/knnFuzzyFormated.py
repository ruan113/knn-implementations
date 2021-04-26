import numpy as np
import pandas as pd
import operator
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

import pdb

class FuzzyKNN(BaseEstimator, ClassifierMixin):
	def __init__(self, k=3, plot=False):
		self.k = k
		self.plot = plot
		self.name = 'Fuzzy KNN'

	def fit(self, X, y=None):
		self._check_params(X,y)
		self.X = X
		self.y = y

		self.xdim = len(self.X[0])
		self.n = len(y)

		# list(set(y)) -> remove valores iguais deixando somente uma classe de cada
		classes = list(set(y))
		classes.sort()
		self.classes = classes

		# transforma dados em um data frame (matriz)
		self.df = pd.DataFrame(self.X)
		self.df['y'] = self.y # adiciona a classe do dado na coluna y

		self.memberships = self._compute_memberships()

		# adiciona uma coluna com o objeto { 'identificador_da_classe': grau_de_filiação }
		self.df['membership'] = self.memberships

		self.fitted_ = True
		return self


	def predict(self, X):
		if self.fitted_ == None:
				raise Exception('predict() called before fit()')
		else:
			m = 2
			y_pred = []

			# Para cada dado de teste
			for x in X:
				neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
				# print(neighbors)
				votes = {}
				# Para cada classe, calcule o voto
				for c in self.classes:
					den = 0
					num = 0
						
					for j in range(self.k):
							dist = neighbors['distances'].iloc[j]
							if dist != 0:
									den += 1 / (dist**(2/(m-1)))
									num += (neighbors.iloc[j].membership[c] * 1) / (dist**(2/(m-1)))
							else:
									num = neighbors.iloc[j].membership[c]
									den = 1
									break
						
					vote = num/den
					votes[c] = vote

				pred = max(votes.items(), key=operator.itemgetter(1))[0]
				y_pred.append((pred, votes))
				
			return y_pred

	def score(self, X, y):
		if self.fitted_ == None:
			raise Exception('score() called before fit()')
		else:
			try:
				predictions = self.predict(X)
				y_pred = [t[0] for t in predictions]
				confidences = [t[1] for t in predictions]
				return accuracy_score(y_pred=y_pred, y_true=y)
			except Exception as e:
				print('Fail to score!')
				print(e)


	def _find_k_nearest_neighbors(self, df, x):
		X = df.iloc[:, 0:self.xdim].values

		df['distances'] = [np.linalg.norm(X[i] - x) for i in range(self.n)]

		df.sort_values(by='distances', ascending=True, inplace=True)
		neighbors = df.iloc[0:self.k]
		return neighbors


	def _get_counts(self, neighbors):
		groups = neighbors.groupby('y')
		# counts = {group[1]['y'].iloc[0]:group[1].count()[0] for group in groups}
		counts = {}	
		for group in groups:
			counts[group[1]['y'].iloc[0]] = group[1].count()[0]
		return counts

	def _compute_memberships(self):
		memberships = []
		for i in range(self.n):
			x = self.X[i]
			y = self.y[i]

			neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
			counts = self._get_counts(neighbors)
			
			membership = dict()
			for c in self.classes:
				try:
					uci = 0.49 * (counts[c] / self.k)
					if c == y:
						uci += 0.51
					membership[c] = uci
				except:
					membership[c] = 0

			memberships.append(membership)
		return memberships


	def _check_params(self, X, y):
		if not(type(self.k) == int or type(self.k) == np.int32):
			raise Exception('"k" should have type int')
		if self.k >= len(y):
			raise Exception('"k" should be less than no of feature sets')
		# if self.k % 2 == 0:
		# 	raise Exception('"k" should be odd')
		if type(self.plot) != bool:
			raise Exception('"plot" should have type bool')