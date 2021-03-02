import csv
import numpy as np

def loadDataset(filename):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        return data
    
def formatData(data):
    target = []
        
    for it in data:
        target.append(it[-1])
        del it[-1]
        
    data = np.array(data).astype(np.float)
    target = np.array(target).astype(np.float)
    return {
        "data": data,
        "target": target,
    }
        
def splitData(dataset, percentage):
    trainingSet = []
    testSet = []
    maxTrainData = int(len(dataset) * percentage)
    
    for n in range(0, len(dataset)):
        if n < maxTrainData:
            trainingSet.append(dataset[n])
        else:
            testSet.append(dataset[n])
            
    return {
        "training": formatData(trainingSet),
        "test": formatData(testSet),
    }

# Get file data and set it into a dataset, formated and splited between train and test sets
# - splitPerc: percentage of training data, it's between 0.75, 0.8, or 0.9
def getData(fileName, splitPerc):
    dataset = loadDataset(fileName)
    return splitData(dataset, splitPerc)
    