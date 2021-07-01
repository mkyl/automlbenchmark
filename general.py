import pandas as pd

X = pd.from_csv("auc-matrix.csv", header=[0], index_col=("model", "dataset"))
X = X.pivot("model", "dataset", "auc")
print(X)
