# from knnCrispy import CrispyKNN
from knnFormated import CrispyKNN
from knnFuzzyFormated import FuzzyKNN
from utils import getClasses
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import pandas as pd

from utils import getData

# iris = load_iris()
# breast = load_breast_cancer()
# diabetes = load_diabetes()
# dataset = diabetes

# dataset = getData('minha-versao/data-sets/adult-data-set/adult_1k.data' , 0.75)
dataset = getData('minha-versao/data-sets/iris-data-set/iris_full.data' , 0.75)

xTrain = dataset['training']['data']
yTrain = dataset['training']['target']
xTest = dataset['test']['data']
yTest = dataset['test']['target']

# print(yTrain)
# print(yTest)

# Passar valor como parametro caso queira valor de k diferente de 3
skModel = CrispyKNN()
custModel = FuzzyKNN()

# Inicializa instâncias de treino
skModel.fit(xTrain, yTrain)
custModel.fit(xTrain, yTrain)

try:
  # print(cross_val_score(cv=5, estimator=skModel, X=xTest, y=yTest))
  # print(cross_val_score(cv=5, estimator=custModel, X=xTest, y=yTest))
  # print([t[0] for t in skModel.predict(xTest)])
  # print(len(yTest))
  # print(len([t[0] for t in skModel.predict(xTest)]))
  # print(dataset['complete']['target'])
  # print(getClasses(dataset['complete']['target']))

  # print(list(yTest))
  # print([t[0] for t in skModel.predict(xTest)])

  knnReport = classification_report(
    list(yTest), 
    [t[0] for t in skModel.predict(xTest)], 
    labels=getClasses(dataset['complete']['target']),
    zero_division=0
  )
  knnConfusionMatrix = confusion_matrix(
    list(yTest), 
    [t[0] for t in skModel.predict(xTest)], 
    labels=getClasses(dataset['complete']['target'])
  )
  fknnReport = classification_report(
    list(yTest), 
    [t[0] for t in custModel.predict(xTest)], 
    labels=getClasses(dataset['complete']['target']),
    zero_division=0
  )
  fknnConfusionMatrix = confusion_matrix(
    list(yTest), 
    [t[0] for t in custModel.predict(xTest)], 
    labels=getClasses(dataset['complete']['target'])
  )

  print('-------------------------- Knn --------------------------')
  print(knnReport)
  print('matriz de confunsão\n')
  print(knnConfusionMatrix)
  print('-------------------------- FKnn --------------------------')
  print(fknnReport)
  print('matriz de confunsão\n')
  print(fknnConfusionMatrix)
except:
  print('Fail to cross validate!')