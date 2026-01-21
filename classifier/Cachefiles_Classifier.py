#!/usr/bin/python3

import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = './samples/filemgr-print_inode_stat.csv'
df = pd.read_csv(file_path, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print("特征 X 的形状:", X.shape)
print("标签 y 的形状:", y.shape)
print("\n前两行特征 X:\n", X[:2])
print("\n前两行标签 y:\n", y[:2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("标准化后的前两行 X:\n", X_scaled[:2])
X = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
print("训练集 X_train 的形状:", X_train.shape)
print("测试集 X_test 的形状:", X_test.shape)
print("训练集 y_train 的形状:", y_train.shape)
print("测试集 y_test 的形状:", y_test.shape)

clf = MLPClassifier(
    hidden_layer_sizes=(512, 200, 2),
    activation='relu',
    solver='sgd',
    alpha=0.0001,
    batch_size=min(200, X_train.shape[0]),
    learning_rate='adaptive',
    learning_rate_init=0.1,
    random_state=numpy.random.RandomState(seed=1),
    momentum=0.5,
    nesterovs_momentum=True,
    max_iter=1500).fit(X_train, y_train)

ret = clf.predict_proba(X_test[:1])
print("预测概率:", ret)

ret = clf.predict(X_test[:5, :])
print("测试集前5行特征 X_test[:5, :]的预测结果:\n", ret)

ret = clf.score(X_test, y_test)
print("模型在测试集上的准确率:", ret)