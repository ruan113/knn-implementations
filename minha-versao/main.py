import time
import datetime

from testerController import TesterController

# Values of K that will be used when classificating
# kValues= list(np.arange(1, 25))
kValues= range(1, 11)

runTime = 0
try:
  runTime = time.time()
  
  execList = [
    # 'minha-versao/data-sets/adult-data-set/adult_1k.data',
    # 'minha-versao/data-sets/iris-data-set/iris_full.data',
    # 'minha-versao/data-sets/winequality-data-set/winequality-red.data',
    # 'minha-versao/data-sets/winequality-data-set/winequality-white.data',
    # 'minha-versao/data-sets/bankmarketing-data-set/additional/formatted-bank.data',
    # 'minha-versao/data-sets/bankmarketing-data-set/normal/formatted-bank.data',
    # 'minha-versao/data-sets/abalone-data-set/formatted-abalone.data',
    'minha-versao/data-sets/student-performance-data-set/formatted-student-mat.data',
    # 'minha-versao/data-sets/student-performance-data-set/formatted-student-por.data',
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
