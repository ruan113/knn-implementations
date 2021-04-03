import csv
import numpy as np
import random
import pandas as pd
import pdb

def loadDataset(filename):
    try:
        with open(filename, 'rt', encoding="utf8") as csvfile:
            lines = csv.reader(csvfile)
            data = list(lines)
            return data
    except:
        print(f"Error: The directory \"{filename}\" does not exists, or it couldn't be read!")
        raise
    
def formatData(data):
    try:
        target = []
        formatedData = []
        for it in data:
            formatedData.append(it[:-1])
            target.append(it[-1])
            
        formatedData = np.array(formatedData).astype(np.float)
        target = np.array(target).astype(np.float)
        return {
            "data": formatedData,
            "target": target,
        }
    except:
        print(f"Error: The system failed to format this data:\n {data}")
        raise
        
def splitData(dataset, percentage):
    try:
        trainingSet = []
        testSet = []
        maxTrainData = int(len(dataset) * percentage)
        dataset = list(dataset)
        random.shuffle(dataset)
        
        for n in range(0, len(dataset)):
            if n < maxTrainData:
                trainingSet.append(dataset[n])
            else:
                testSet.append(dataset[n])
                
        return {
            "complete": formatData(dataset),
            "training": formatData(trainingSet),
            "test": formatData(testSet),
        }
    except:
        print(f"Error: The system failed to split this data:\n {dataset}")
        raise

# Get file data and set it into a dataset, formated and splited between train and test sets
# - splitPerc: percentage of training data, it's between 0.75, 0.8, or 0.9
def getData(fileName, splitPerc):
    dataset = loadDataset(fileName)
    return splitData(dataset, splitPerc)

def getClasses(y):
    classes = list(set(y))
    classes.sort()
    return classes

def formatConfusionMatrix(dataframe, classes): 
    try:
        df = pd.DataFrame(dataframe, columns=classes, index=classes)
        return df
    except:
        print('Erro ao formatar matrix de confusão')
        raise
        
def calculateSummary(confusionMatrix):
    try:
        columnsLength = len(confusionMatrix[0])
        correctlyClassified = 0
        incorrectlyClassified = 0
        
        for i in range(columnsLength):
            for j in range(columnsLength):
                if (i == j):
                    correctlyClassified += confusionMatrix[i][j]
                else:
                    incorrectlyClassified += confusionMatrix[i][j]
                    
        correctlyClassifiedRate = correctlyClassified/(correctlyClassified + incorrectlyClassified)
        incorrectlyClassifiedRate = incorrectlyClassified/(incorrectlyClassified + correctlyClassified)
    
        result = ''
        result += f'Correctly Classified: {correctlyClassified}\t{formatPercentage(correctlyClassifiedRate)}%\n'
        result += f'Incorrectly Classified: {incorrectlyClassified}\t{formatPercentage(incorrectlyClassifiedRate)}%\n'
        result += f'Total Number Of Instances: {correctlyClassified + incorrectlyClassified}\n'
        return result
    except:
        print('Erro ao calcular o summary')
        raise

    
def formatPercentage(value):
    value *= 100
    value = '{0:.3g}'.format(value)
    return value