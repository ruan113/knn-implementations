from knnCrispy import run as knn

porcentagem = 0.75 # porcentagem de dados que será usada como instancia de treinamento
k = 3 # num de vizinhos que serão considerados

knn('minha-versao/datasets/iris-data-set/iris_full.data', porcentagem, 5, k)
knn('minha-versao/datasets/adult-data-set/adult_1k.data', porcentagem, 14, k)
knn('minha-versao/datasets/adult-data-set/adult_5k.data', porcentagem, 14, k)
knn('minha-versao/datasets/adult-data-set/adult_10k.data', porcentagem, 14, k)