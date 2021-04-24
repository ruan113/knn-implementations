from utils import getData, getNumClasses
import pandas as pd

datasetList = [
  'main/data-sets/iris-data-set/iris_full.data',
  'main/data-sets/adult-data-set/adult_full.data',
  'main/data-sets/winequality-data-set/winequality-red.data',
  'main/data-sets/winequality-data-set/winequality-white.data',
  'main/data-sets/bankmarketing-data-set/normal/formatted-bank.data',
  'main/data-sets/bankmarketing-data-set/additional/formatted-bank.data',
  'main/data-sets/abalone-data-set/formatted-abalone.data',
  'main/data-sets/student-performance-data-set/formatted-student-mat.data',
  'main/data-sets/student-performance-data-set/formatted-student-por.data',
]

df = pd.DataFrame(list()) 

matrix = [[], [], [], []]
for path in datasetList:
  data = getData(path)
  fileName = path.split('/')[-1].replace('.data', '')
  nAtributes = len(data['complete']['data'][0])
  nClasses = getNumClasses(data['complete']['target'])
  matrix[0].append(fileName)
  matrix[1].append(nAtributes)
  matrix[2].append(nClasses)
  matrix[3].append(len(data['complete']['data']))
  
df['name'] = matrix[0]
df['nAtributes'] = matrix[1]
df['nClasses'] = matrix[2]
df['totalData'] = matrix[3]
print(df)
