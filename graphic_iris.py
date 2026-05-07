import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data  # Entrada rotulo Analogo as entradas de uma rede neural.
y = iris.target # Saida

plt.figure()

for i,nome in enumerate(iris.target_names):
    plt.scatter(
        X[y == i, 2],
        X[y == i, 3],
        
        
        #X[:, 0] vs X[:, 1] # Comparando com essas colunas
        #X[:, 1] vs X[:, 2]
        label = nome
    )

plt.axvline(x=2.5, linestyle='--')# Traçar Reta...
    
plt.xlabel('Largura das Sepalas')
plt.ylabel('Altura das Sepalas')
plt.title('Separação das Flores (IRIS)')
plt.legend()
plt.show()