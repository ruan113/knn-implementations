# from knnCrispy import CrispyKNN
from knnFormated import CrispyKNN
from knnFuzzyFormated import FuzzyKNN
from pprint import pprint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes

from utils import getData

# iris = load_iris()
# breast = load_breast_cancer()
# diabetes = load_diabetes()
# dataset = diabetes

dataset = getData('minha-versao/data-sets/adult-data-set/adult_1k.data' , 0.75)

xTrain = dataset['training']['data']
yTrain = dataset['training']['target']
xTest = dataset['test']['data']
yTest = dataset['test']['target']

# Passar valor como parametro caso queira valor de k diferente de 3
skModel = CrispyKNN()
custModel = FuzzyKNN()

# Inicializa instâncias de treino
skModel.fit(xTrain, yTrain)
custModel.fit(xTrain, yTrain)

try:
  print(cross_val_score(cv=5, estimator=skModel, X=xTest, y=yTest))
  print(cross_val_score(cv=5, estimator=custModel, X=xTest, y=yTest))
except:
  print('Fail to cross validate!')


# porcentagem = 0.85 # porcentagem de dados que será usada como instancia de treinamento
# k = 3 # num de vizinhos que serão considerados

# knn('minha-versao/data-sets/iris-data-set/iris_full.data', porcentagem, 5, k)
# knn('minha-versao/data-sets/adult-data-set/adult_1k.data', porcentagem, 14, k)
# knn('minha-versao/data-sets/adult-data-set/adult_5k.data', porcentagem, 14, k)
# knn('minha-versao/data-sets/adult-data-set/adult_10k.data', porcentagem, 14, k)