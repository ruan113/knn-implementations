from utils import getData, getClasses, calculateSummary, formatConfusionMatrix, formatMetrics
from knnFormated import CrispyKNN
from knnFuzzyFormated import FuzzyKNN
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import pandas as pd
import time
import pdb
import datetime

class TesterController():
  def __init__(self, dataPath, kValues = [3], testPerc = 0.75):
    self.kValues = kValues
    self.fileName = dataPath.split("/")[-1]
    self.dataPath = dataPath
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
        zero_division=0,
        labels = self.classes,
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
        'confusionMatrix': confusionMatrix,
        # Brute Data
        'metrics': metrics,
        'summary': calculateSummary(confusionMatrix, True),
        # String report
        'metricsString': metricsString,
        'summaryString': calculateSummary(confusionMatrix, False),
        # Errors
        'error': None
      }
    except:
      return {
        'name': model.name,
        'error': f'Error while execModel using {self.index} - {model.name}'
      }
      
    
  def generateReport(self, reportData):
    report = ''
    k = reportData["k"]
    print(f'Gerando reports no index {self.index} com k = {k}...')
    for key in reportData:
      if(key != "k"):
        name = reportData[key]["name"]
        summary = reportData[key]['summaryString']
        runTime = reportData[key]['runTime']
        confusionMatrix = reportData[key]['confusionMatrix']
        metrics = reportData[key]['metricsString']
        error = reportData[key]['error']

        report += f'-------------------------- {name} (k = {k}) --------------------------\n'

        if(error is not None):
          report += f"{error}\n"
        else:
          report += f'Execution Time: {runTime} seconds\n'
          report += summary
          report += '\n'
          report += f'{metrics}\n'
          report += '\n'
          report += 'matriz de confunsão\n'
          confusionMatrix = formatConfusionMatrix(confusionMatrix,self.classes)
          report += f'{confusionMatrix}\n'
      
    return report
    
  def run(self, executionIndex):
    self.index = executionIndex
    reports = ""
    
    try:
      for k in self.kValues:
        crispyModel = CrispyKNN(k)
        fuzzyModel = FuzzyKNN(k)
          
        report = {
          "k": k,
          "crispyReport": self.execModel(crispyModel),
          "fuzzyReport": self.execModel(fuzzyModel)
        }
        
        reports += self.generateReport(report)
      
      return reports
    except:
      print(f'Houve um erro durante a execução do index "{executionIndex}"')