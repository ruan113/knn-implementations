from utils import getData, getClasses, calculateSummary, formatConfusionMatrix
from knnFormated import CrispyKNN
from knnFuzzyFormated import FuzzyKNN
from sklearn.metrics import confusion_matrix, classification_report

import time
import pdb
import datetime

class TesterController():
  def __init__(self, dataPath, k = 3, testPerc = 0.75):
    self.k = k
    self.initializeData(dataPath, testPerc)
    
  def initializeData(self, dataPath, testPerc):
    self.dataset = getData(dataPath , testPerc)
    self.xTrain = self.dataset['training']['data']
    self.yTrain = self.dataset['training']['target']
    self.xTest = self.dataset['test']['data']
    self.yTest = self.dataset['test']['target']
    self.classes = getClasses(self.dataset['complete']['target'])
    
  def execModel(self, model):
    print(f'Começando classificação do index {self.index} - {model.name}...')

    try: 
      runTime = time.time()
      model.fit(self.xTrain, self.yTrain)
      predictions = [t[0] for t in model.predict(self.xTest)]
      runTime = time.time() - runTime
      
      metrics = classification_report(
        list(self.yTest), 
        predictions, 
        labels = self.classes,
        zero_division=0,
        output_dict=True
      )
      
      metricsString = classification_report(
        list(self.yTest), 
        predictions, 
        labels = self.classes,
        zero_division=0
      )

      confusionMatrix = confusion_matrix(
        list(self.yTest), 
        predictions, 
        labels=self.classes
      )
        
      print(f'classificação do index {self.index} - {model.name} finalizada!')
      return {
        'name': model.name,
        'runTime': str(datetime.timedelta(seconds=runTime)),
        'metrics': metrics,
        'metricsString': metricsString,
        'confusionMatrix': confusionMatrix,
        'summary': calculateSummary(confusionMatrix, True),
        'summaryString': calculateSummary(confusionMatrix, False),
        'error': None
      }
    except:
      return {
        'name': model.name,
        'error': f'Error while execModel using {self.index} - {model.name}'
      }
      
    
  def generateReport(self, reportData):
    report = ''
    print(f'Gerando reports no index {self.index}...')
    for key in reportData:
      name = reportData[key]["name"]
      runTime = reportData[key]['runTime']
      confusionMatrix = reportData[key]['confusionMatrix']
      metrics = reportData[key]['metrics']
      error = reportData[key]['error']

      report += f'-------------------------- {name} --------------------------\n'

      if(error is not None):
        report += f"{error}\n"
      else:
        report += f'Execution Time: {runTime} seconds\n'
        report += calculateSummary(confusionMatrix)
        report += '\n'
        report += f'{metrics}\n'
        report += '\n'
        report += 'matriz de confunsão\n'
        confusionMatrix = formatConfusionMatrix(confusionMatrix,self.classes)
        report += f'{confusionMatrix}\n'
      
    return report
    
  def run(self, executionIndex):
    self.index = executionIndex
    crispyModel = CrispyKNN(self.k)
    fuzzyModel = FuzzyKNN(self.k)
        
    try:
      report = {
        "crispyReport": self.execModel(crispyModel),
        "fuzzyReport": self.execModel(fuzzyModel)
      }
      
      return self.generateReport(report)
    except:
      print(f'Houve um erro durante a execução do index "{executionIndex}"')