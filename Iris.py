import pandas as pd
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
#Carregando a Iris

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Nossa Saida
#Tranformando em DataFrames

np.random.seed(99)
#definindo seed para reprodução

df = df.sample(frac=1).reset_index(drop=True)
#Embaralhando os dados

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

#aplicando a minha normalização nas 4 primeiras colunas

df_norm = df.copy()
df_norm.iloc[:, 0:4] = df_norm.iloc[:, 0:4].apply(normalize)

#Separação Treino / Teste...

iris_train = df_norm.iloc[: 120, 0:4]
iris_train_target = df.iloc[:120, 4]

iris_test = df_norm.iloc[120:150, 0:4]

print(df_norm.head())

