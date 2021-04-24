import csv

# load datasets into matrix form where x is row, and y is column
# params:
# - filename: file name that contains the datasets
# - SPLIT: percentage of training data, it's between 0.75, 0.8, or 0.9
# output:
# - trainingSet: training data
# - testSet: test data  
def loadDataset(filename, split, trainingSet=[] , testSet=[], numColunas=1):
    with open(filename, 'rt', encoding="utf8") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        maxTrainData = int(split * len(dataset))
        for x in range(0, len(dataset)):
            for y in range(numColunas):
                dataset[x][y] = float(dataset[x][y])
            if x < maxTrainData:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])