import pandas as pd
import scipy.stats
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def roc_calc(labels, score):
    fpr, tpr, thresholds = roc_curve(labels, score)
    roc = auc(fpr, tpr)
    return roc


def geom_p_calc(total, outlier, tumor, common):
    return scipy.stats.hypergeom.sf(common - 1, total, outlier, tumor)


cancer = pd.DataFrame(pd.read_csv('../demo/input.txt', sep='\t'))
cancer = cancer.set_index('Symbol').transpose()
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(cancer)
cancer.loc[:, :] = scaled_values

cancer_type = 1
mask = (cancer.TYPE == 0) | (cancer.TYPE == cancer_type)
cancer = cancer[mask]
mask = (cancer.TYPE != 0)
cancer.loc[mask, 'TYPE'] = 1
cancer_type = 1

mask = (cancer.TYPE != 0)
cancer.loc[mask, 'TYPE'] = 2

mask = (cancer.TYPE == 0)
cancer.loc[mask, 'TYPE'] = 1

mask = (cancer.TYPE == 2)
cancer.loc[mask, 'TYPE'] = 0

cancer_type = 1
X_train, X_test, y_train, y_test = train_test_split(cancer,
                                                    cancer.TYPE,
                                                    test_size=0.3)

X_train, X_train_test, y_train, y_train_test = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.3)

normal = X_train[X_train.TYPE == 0]

result = pd.DataFrame()
result = result.append(normal.quantile(.25))
result = result.append(normal.quantile(.75))
iqr = result.iloc[1] - result.iloc[0]
iqr.name = 'iqr'
result = result.append(iqr)
fence_low = result.iloc[0] - 1.5 * result.iloc[2]
fence_low.name = 'fence_low'
result = result.append(fence_low)
fence_high = result.iloc[1] + 1.5 * result.iloc[2]
fence_high.name = 'fence_high'
result = result.append(fence_high)

#Reference Range
total = len(normal.columns)
tumor = X_train[X_train.TYPE == cancer_type]
ct = (tumor > result.iloc[4]).sum(axis=0)
outliers = ct[ct > max(ct) * 0.2].index.to_list()[:-1]

#Prediction
train_tumor = (X_train_test > result.iloc[4]).sum(axis=1)
train_test = X_train_test[outliers]
commons = (train_test > result.iloc[4]).sum(axis=1)
score = []
for exp in commons.index.to_list():
    common = commons[exp]
    total = 10709
    outlier = len(outliers)
    n = train_tumor[exp]
    print(n)
    pval = geom_p_calc(total, outlier, n, common)
    if pval == 0:
        pval = 1e-100
    score.append(-np.log10(pval))
roc_val = roc_calc(y_train_test, score)
#Output
print("roc:", roc_val)
with open('../demo/output.txt', 'w') as f:
    f.write('Pre\tReal\n')
    for pre, real in zip(y_train_test, score):
        f.write(str(pre) + '\t' + str(real) + '\n')
