import time
import datetime

from testerController import TesterController

# Values of K that will be used when classificating
# kValues= list(np.arange(1, 25))
kValues= range(1, 2)

runTime = 0
try:
  runTime = time.time()
  
  execList = [
    'main/data-sets/abalone-data-set/formatted-abalone.data',
    'main/data-sets/adult-data-set/adult_full.data',
    'main/data-sets/bankmarketing-data-set/additional/formatted-bank_full.data',
    'main/data-sets/bankmarketing-data-set/normal/formatted-bank_full.data',
    'main/data-sets/cs-go-data-set/formatted-csgo_round.csv'
    'main/data-sets/iris-data-set/iris_full.data',
    'main/data-sets/student-performance-data-set/formatted-student-mat.data',
    'main/data-sets/student-performance-data-set/formatted-student-por.data',
    'main/data-sets/winequality-data-set/winequality-red.data',
    'main/data-sets/winequality-data-set/winequality-white.data',
  ]
  
  index = 0
  report = ''
  for path in execList:
    print(f'Começando execução do index {index}...')
    report += f'{index} - {path.split("/")[-1]}:\n'
    report += TesterController(path, kValues=kValues).run('score-kvalue')
    report += '--------------------------------------------\n'
    print(f'Execução do index {index} finalizada!')
    index += 1

  runTime = str(datetime.timedelta(seconds=(time.time() - runTime)))

  print(f'Total Tests: {len(execList)}')
  print(f'Total run time: {runTime}')
  print(f'{report}')
except Exception:
  print('something went wrong!')
