import csv
import numpy as np
import random

def loadDataset(filename):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        return data
    
def formatData(data):
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
        
def splitData(dataset, percentage):
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

# Get file data and set it into a dataset, formated and splited between train and test sets
# - splitPerc: percentage of training data, it's between 0.75, 0.8, or 0.9
def getData(fileName, splitPerc):
    dataset = loadDataset(fileName)
    return splitData(dataset, splitPerc)

def getClasses(y):
    classes = list(set(y))
    classes.sort()
    return classes

# def formatConfusionMatrix(dataframe, classes): 
    