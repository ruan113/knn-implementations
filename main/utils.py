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
def getData(fileName, splitPerc = 0.75):
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
        
def calculateSummary(confusionMatrix, output_dict=False):
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
        if(output_dict == False):
            result += f'Correctly Classified: {correctlyClassified}\t{formatPercentage(correctlyClassifiedRate)}%\n'
            result += f'Incorrectly Classified: {incorrectlyClassified}\t{formatPercentage(incorrectlyClassifiedRate)}%\n'
            result += f'Total Number Of Instances: {correctlyClassified + incorrectlyClassified}\n'
        else: 
            result = {
                'correctlyClassified': correctlyClassified,
                'incorrectlyClassified': incorrectlyClassified
            }
        
        return result
        
    except:
        print('Erro ao calcular o summary')
        raise

def formatPercentage(value):
    value *= 100
    value = '{0:.3g}'.format(value)
    return value

def getNumClasses(y):
    classes = list(set(y))
    classes.sort()
    return len(classes)

def formatMetrics(metrics, hideFields=[
    'micro avg',
    'macro avg',
    'weighted  avg',
]):
    result = list()
    
    for key in metrics:
        aux = list()
        if(key in hideFields):
            for field in metrics[key]:
                aux.append(field)
            result.append(aux)
            
    return result

    
def generateCSV(fileName, data):
    fileName = fileName.replace('.data', '')
    try:
        print('Iniando geração de csv!')
        f = open(f"main/results/{fileName}.csv", "w")
        f.write(data)
        f.close()
    except:
        print(f'Falha ao tentar gerar arquivo csv - {fileName}')

def generateAllInfoCsv(fileName, bruteData):
    print('gerando csv...')
    csvString = "Dataset, K, Method, Acertos, Erros, precision, recall, f1-score, support\n"

    i = 2
    for report in bruteData:
        for key in report:
            if(key != "k"):
                k = report["k"]
                name = report[key]["name"]
                summary = report[key]['summary']
                acertos = summary['correctlyClassified']
                erros = summary['incorrectlyClassified'] 
                metrics = report[key]['metrics']
                precision = metrics['precision']
                recall = metrics['recall']
                f1Score = metrics['f1-score']
                support = metrics['support']
                if(i == 2):
                    csvString += f"{fileName}, {k}, {name}, {acertos}, {erros}, {precision}, {recall}, {f1Score}, {support}\n"
                else:
                    if(i % 2 == 0):
                        csvString += f", {k}, {name}, {acertos}, {erros}, {precision}, {recall}, {f1Score}, {support}\n"
                    else: 
                        csvString += f",, {name}, {acertos}, {erros}, {precision}, {recall}, {f1Score}, {support}\n"
        i += 1    

    return csvString  