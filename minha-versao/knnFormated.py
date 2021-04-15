import pdb
import numpy as np
import pandas as pd
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# self.k -> valor de k
# self.X -> dados da instancia
# self.y -> classes da instancia
# self.xdim -> quantidade de atributos dentro de um dado
# self.n -> quantidade de dados
# self.df -> data frame
# 
# 

class CrispyKNN(BaseEstimator, ClassifierMixin):
  def __init__(self, k=3, plot=False):
	  self.k = k
	  self.plot = plot
	  self.name = 'KNN'
   
  def fit(self, X, y=None):
    self._check_params(X,y)
    self.X = X
    self.y = y

    self.xdim = len(self.X[0])
    self.n = len(y)

    classes = list(set(y))
    classes.sort()
    self.classes = classes

    self.df = pd.DataFrame(self.X)
    self.df['y'] = self.y

    self.memberships = self._compute_memberships()
    self.df['membership'] = self.memberships
    self.fitted_ = True
    return self


  def predict(self, X):
    if self.fitted_ == None:
      raise Exception('predict() called before fit()')
    else:
      y_pred = []

      for x in X:
        # Acha os k vizinhos mais próximos
        neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x)
        # Pega a quantidade de classes
        counts = self._get_counts(neighbors)
        
        for c in self.classes:
          try:
            if(type(counts[c]) == int or type(counts[c]) == float):
              pass
          except:
            counts[c] = 0

        pred = max(counts.items(), key=operator.itemgetter(1))[0]
        y_pred.append((pred, counts))

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
      except:
        print('Fail to score!')


  def _find_k_nearest_neighbors(self, df, x):
    X = df.iloc[:,0:self.xdim].values

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
          membership[c] = counts[c]
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
    #   raise Exception('"k" should be odd')
    if type(self.plot) != bool:
      raise Exception('"plot" should have type bool')