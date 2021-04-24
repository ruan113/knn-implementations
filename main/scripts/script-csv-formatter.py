import pdb
from pprint import pprint

from main.utils import loadDataset

# Descrição: Este script transformará todos os valores não numéricos de uma matriz em valores numéricos#
#
# Explicação: Este script fará uma varredura pela matriz e fará um cadastro de todos os campos não 
# numericos em um array (columnReference), transformando a referencia desse valor naquela coluna em um numero.
# O script varrerá cada um dos valores na matriz, e a cada valor não numerico o script varrerá o array de referencias
# atrás do index da coluna e dentro deste index, procurará se o valor string já não possui um valor numérico atrelado ao mesmo
# caso exista, o valor será retornado, caso não exista o mesmo será criado e retornado, resultando em uma transformação
# de todos os valores não numericos em numéricos

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
      return False

def exists(s):
  try:
    float(s)
    return True
  except ValueError:
      return False

def getNumericValue(references, columnIndex, columnValue):
  if(references is None):
    print('References not initilized')
    return
  
  if(columnIndex in references):
    lastNumericRef = 1
    for columnRef in references[columnIndex]:
      lastNumericRef = int(columnRef['numericRef'])
      if(columnRef['stringRef'] == columnValue):
        return columnRef['numericRef']
      
    newRef = {
      'stringRef': columnValue,
      'numericRef': str(int(lastNumericRef) + 1)
    }
    references[columnIndex].append(newRef)
    return newRef['numericRef']
  else:
    references[columnIndex] = list()
    references[columnIndex].append({
      'numericRef': '1',
      'stringRef': columnValue,
    })
    return '1'
    
def createFile(path, data):
  f = open(path, "a")
  f.write(arrayToString(data))
  f.close()

def arrayToString(data):
  result = ''
  for row in range(0, len(data)):
    result += ','.join(data[row]) + '\n'
  
  return result
    

def run():
  # Mudar path para o arquivo desejado 
  path = 'minha-versao/data-sets/student-performance-data-set/student-por.data'
  aux = path.split('/')
  aux.pop(-1)
  relativePath = '/'.join(aux) + '/'

  data = loadDataset(path)
  nAtributes = len(data[0])

  # Interface: 
  #
  # 'columnIndex': [
  #   {
  #     'stringRef': 'string',
  #     'numericRef': 'number'
  #   }
  # ]
  columnReference = {}

  for row in range(0, len(data)):
    for column in range(0, nAtributes):
      columnValue = data[row][column]
      if(not is_number(columnValue)):
        data[row][column] = getNumericValue(columnReference, column, columnValue)
       
  try:
    newFileName = relativePath + ('formatted-' + path.split('/')[-1])
    createFile(newFileName, data)
    print(f'Arquivo {newFileName} foi criado/atualizado com sucesso')
  except:
    print('Um erro ocorreu durante a criação/atualização do arquivo!');
      
run()