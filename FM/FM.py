from  math import exp
import numpy as np
from random import normalvariate
from sklearn import  preprocessing
import pandas as pd

class FM(object):
    def __init__(self):
        self.data=None
        self.feature_potential=None
        self.alpha=None
        self.iter=None
        self._w=None
        self.v=None
        self.with_col=None
        self.first_col=None

    def min_max(self,data):
        self.data=data
        min_max_scaler=preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(self.data)

    def loadDataSet(self,data,with_col=True,first_col=2):
        data=pd.read_csv(data,header=0,sep=",")
        return data[:,:-1],data[:,-1:]


    def sigmoid(self,x):
        return 1.0/(1+exp(-x))

    def fit(self,data,feature_potential=8,alpha=0.01,iter=100):
        self.alpha=alpha
        self.feature_potential=feature_potential
        self.iter=iter
        data,label=self.loadDataSet(data)
        print("data.shape",data.shape)
        print('label.shape',label.shape)
        k=self.feature_potential
        m,n=np.shape(data)
        w=np.zeros((n,1))
        w_0=0
        v=normalvariate(0,0.2)*np.ones((n,k))
        for i in range(m):
            inter_1=data[i]*v
            inter_2=np.multiply(data[i],data[i])*np.multiply(v,v)
            interaction=np.sum(np.multiply(inter_1,inter_2)-inter_2)/2
            p=w_0+data[i]*w+interaction
            print("预测的输出",p)

            loss=self.sigmoid(label[i]*p[0,0])-1

            w_0-=self.alpha*loss*label[i]

            for x in range(n):
                if data[i,x]!=0:
                    w[x,0]-=self.alpha*loss*label[i]*data[i,x]

                    for j in range(k):
                        v[x,j]-=self.alpha*loss*label[k]*(data[i,x]*inter_1[0,j]-v[x,j]*data[i,x]*data[i,x])
        self._w_0,self._w,self._v=w_0,w,v
    def predict(self,X):
        if (self._w_0==None) or (self._w==None) or (self.v==None):
            raise Exception("fit fail")

        w_0=self._w_0
        w=self._w
        v=self._v
        m,n=np.shape(X)
        result=[]
        for x in range(m):
            inter_1 = np.mat(X[x]) * v
            inter_2 = np.mat(np.multiply(X[x], X[x])) * np.multiply(v, v)
            interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + X[x] * w + interaction
            pre = self.sigmoid(p[0, 0])

            result.append(pre)

        return result

    def getAccuracy(self, data):
        dataMatrix, classLabels = self.loadDataSet(data)
        w_0 = self._w_0

        w = self._w
        v = self._v
        m, n = np.shape(dataMatrix)
        allItem = 0
        error = 0
        result = []
        for x in range(m):
            allItem += 1
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) *np. multiply(v, v)  # multiply对应元素相乘
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
            pre = self.sigmoid(p[0, 0])
            result.append(pre)
            if pre < 0.5 and classLabels[x] == 1.0:
                error += 1
            elif pre >= 0.5 and classLabels[x] == -1.0:

                error += 1
            else:
                continue
        value = 1 - float(error) / allItem
        return value