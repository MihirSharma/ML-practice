from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd

cancerdataset = load_breast_cancer()

cancer_dataset = pd.DataFrame(cancerdataset['data'], columns = cancerdataset['feature_names'])

plt.matshow(cancer_dataset.corr())
tick_names = list(cancer_dataset.columns.values[i] for i in range(0,30))
plt.xticks(range(0,30), tick_names, rotation = 90)
plt.yticks(range(0,30), tick_names)
plt.figtext(s = list(cancer_dataset.columns.values[i] for i in range(0, 30)), x = 0, y = 1)
plt.show()
