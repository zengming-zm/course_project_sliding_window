#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import svm
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix

train = genfromtxt('./output/huyun_activity_primitives_global_train.csv',delimiter=',')
test = genfromtxt('./output/huyun_activity_primitives_global_test.csv',delimiter=',')
train_label = train[:,0]
train_data = train[:,1:]
test_label = test[:,0]
test_data = test[:,1:]
#lst = range(1,30,2)
lst = [0.5]
for C in lst:
    print "C = {0}".format(C)
    #svc = svm.LinearSVC(C=C)
    #svc.fit(train_data, train_label)
    #pred = svc.predict(test_data)
    #print svc.score(test_data, test_label)
    #con = confusion_matrix(test_label, pred)
    logreg = linear_model.LogisticRegression(C=C)
    logreg.fit(train_data,train_label)
    pred = logreg.predict(test_data)
    con = confusion_matrix(test_label, pred)
    #print logreg.score(test_data,test_label)
for i in range(0,con.shape[0]):
    s = sum(con[i,:])
    print con[i,i]/float(s)
