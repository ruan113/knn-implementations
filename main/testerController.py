from utils import getData, getClasses, calculateSummary, formatConfusionMatrix, generateCSV, getCSVInfo
from knnFormated import CrispyKNN
from knnFuzzyFormated import FuzzyKNN

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import time
import pdb
import datetime

class TesterController():
  def __init__(self, dataPath, kValues = [3], trainPerc = 0.75):
    self.kValues = kValues
    self.fileName = dataPath.split("/")[-1]
    self.dataPath = dataPath
    self.initializeData(dataPath, trainPerc)
    
  def initializeData(self, dataPath, trainPerc):
    self.dataset = getData(dataPath , trainPerc)
    self.X = self.dataset['complete']['data']
    self.y = self.dataset['complete']['target']
    
    self.xTrain = self.dataset['training']['data']
    self.yTrain = self.dataset['training']['target']
    self.xTest = self.dataset['test']['data']
    self.yTest = self.dataset['test']['target']
    
    self.classes = getClasses(self.dataset['complete']['target'])
    
  def execModel(self, model):
    print(f'Começando classificação do dataset {self.fileName} - {model.name}...')
    
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
      print(f'classificação do fileName {self.fileName} - {model.name} finalizada!')
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
        'error': f'Error while execModel using {self.fileName} - {model.name}'
      }
      
    
  def generateReport(self, reportData):
    report = ''
    k = reportData["k"]
    print(f'Gerando reports no fileName {self.fileName} com k = {k}...')
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
  
  def getAllInfo(self):
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
      print(f'Houve um erro durante a execução do fileName "{self.fileName}"')
    
  def getScoreKValue(self):
    reports = ",KNN,,,FKNN,,\n" + "k,Avg Score,Best Score,Time,Avg Score,Best Score,Time\n"
    csvName = f'scoreKvalues-{self.fileName}'
    reportInfo = getCSVInfo(csvName)
    
    if (len(reportInfo['kValues']) == 0):
      generateCSV(csvName, reports, 'a')
        
    try:
      for k in self.kValues:
        if(k in reportInfo['kValues']):
          pass
        else:   
          print(f'classificando {self.fileName} utilizando k = {k}')
          crispyModel = CrispyKNN(k)
          fuzzyModel = FuzzyKNN(k)

          print('Começando cross_validate KNN...');
          cvScores = cross_validate(crispyModel, self.X, self.y, cv=5)
          print('Começando cross_validate FKNN...');
          fcvScores = cross_validate(fuzzyModel, self.X, self.y, cv=5)

          scores = self.getBestValues(cvScores)
          fscores = self.getBestValues(fcvScores)

          row = f'{k},{scores["avgScore"]},{scores["bestScore"]},{scores["time"]},{fscores["avgScore"]},{fscores["bestScore"]},{fscores["time"]}\n'
          generateCSV(csvName, row, 'a')
          reports += row
      
      print(f'{csvName}-sucesso...');
      return reports
    except:
      print(f'Houve um erro durante a execução do fileName "{self.fileName}"')
    
  def getBestValues(self, result):
    bestScore = 0
    bestTime = 0
    
    i = 0
    for it in result['test_score']:
      if (bestScore < it):
        bestScore = it
        bestTime = result['fit_time'][i] + result['score_time'][i]
        i += 1
          
    return {
      'avgScore': np.mean(result['test_score']),
      'bestScore': bestScore,
      'time': bestTime,
    }
    
  def handleExecution(self, executionType):
    if (executionType == 'all'):
      return self.getAllInfo()
    else: 
      if (executionType == 'score-kvalue'):
        return self.getScoreKValue()
    
    return None
    
  def run(self, executionType = 'all'):
    report = self.handleExecution(executionType)
    
    if (report != None): 
      return report
    else:
      return ''
