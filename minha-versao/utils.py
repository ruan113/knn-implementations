import csv
import numpy as np

# load datasets into matrix form where x is row, and y is column
# params:
# - filename: file name that contains the datasets
# - split: percentage of training data, it's between 0.75, 0.8, or 0.9
# output:
# - trainingSet: training data
# - testSet: test data  
def loadDataset(filename):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
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