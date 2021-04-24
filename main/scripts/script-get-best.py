import time
import datetime
import numpy as np
import pandas as pd
import pdb

from sklearn.model_selection import GridSearchCV

from knnFormated import CrispyKNN
from knnFuzzyFormated import FuzzyKNN

from utils import getData

runTime = 0

def classificate(dataPath, trainPerc=0.75):
  print('here we go again')
  crispyModel = CrispyKNN()
  fuzzyModel = FuzzyKNN()
  
  fileName = dataPath.split("/")[-1]
  dataset = getData(dataPath , trainPerc)
  X = dataset['complete']['data']
  y = dataset['complete']['target']

  param_grid = {'k': np.arange(1, 5)}
  knn_gscv = GridSearchCV(crispyModel, param_grid, cv=5)
  fknn_gscv = GridSearchCV(fuzzyModel, param_grid, cv=5)
  print(np.arange(1, 5))
  print('começando...')
  print('Knn...')
  knn_gscv.fit(X, y)
  print(f'{fileName} - {knn_gscv}')
  print(f'{fileName} - bestParams: {knn_gscv.best_params_} - bestScore: {knn_gscv.best_score_}')
  print('FKNN...')
  fknn_gscv.fit(X, y)
  print(f'{fileName} - {fknn_gscv}')
  print(f'{fileName} - bestParams: {fknn_gscv.best_params_} - bestScore: {fknn_gscv.best_score_}')

def run(): 
  runTime = time.time()
      
  execList = [
    # 'minha-versao/data-sets/adult-data-set/adult_1k.data',
    # 'minha-versao/data-sets/iris-data-set/iris_full.data',
    # 'minha-versao/data-sets/winequality-data-set/winequality-red.data',
    # 'minha-versao/data-sets/winequality-data-set/winequality-white.data',
    # 'minha-versao/data-sets/bankmarketing-data-set/additional/formatted-bank.data',
    # 'minha-versao/data-sets/bankmarketing-data-set/normal/formatted-bank.data',
    # 'minha-versao/data-sets/abalone-data-set/formatted-abalone.data',
    'minha-versao/data-sets/student-performance-data-set/formatted-student-mat.data',
    # 'minha-versao/data-sets/student-performance-data-set/formatted-student-por.data',
  ]
  
  for path in execList:
    classificate(path)

  runTime = str(datetime.timedelta(seconds=(time.time() - runTime)))

  print(f'Total Tests: {len(execList)}')
  print(f'Total run time: {runTime}')
    
run()