import lightgbm as lgb

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

print("load Data")

train_data=pd.read_csv('data/train.csv')

test_data=pd.read_csv("data/test.csv")

NUMERIC_COLS=[]

y_train=train_data[""]

y_test=test_data[""]

X_train=train_data[NUMERIC_COLS]

X_test=test_data[NUMERIC_COLS]

lgb_train=lgb.Dataset(X_train,y_train)

lgb_test=lgb.Dataset(X_test,y_test,reference=lgb_train)

params={
    'task':'train',
    'boosting_type':"gbdt",
    "object":"binary",
    'metric':{"binary_logloss"},
    'num_leaves':64,
    'num_trees':100,
    'learning_rate':0.01,
    'feature_fraction':0.9,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'verbose':0
}

num_leaf=64


print("start training")


gbm=lgb.train(
    params,lgb_train,num_boost_round=100,valid_sets=lgb_train
)

print("save model")

gbm.save_model("model.txt")

print("start predicting")

y_pred=gbm.predict(X_train,pred_leaf=True)

print(np.array(y_pred).shape)

print("transforming training data")

transform_train_matrix=np.zeros([len(y_pred),len(y_pred[0])*num_leaf],dtype=np.int64)

for i in range(0,len(y_pred)):
    temp=np.arange(len(y_pred[0])*num_leaf+np.array(y_pred[i]))
    transform_train_matrix[i][temp]+=1


y_pred=gbm.predict(X_test,pred_leaf=True)
transform_test_matrix=np.zeros([len(y_pred),len(y_pred[0])*num_leaf],dtype=np.int64)

for i in range(0,len(y_pred)):
    temp=np.arange(len(y_pred[0])*num_leaf+np.array(y_pred[i]))
    transform_test_matrix[i][temp]+=1
lm=LogisticRegression(penalty='l2',C=0.5)

lm.fit(transform_train_matrix,y_train)

y_pred_test=lm.predict_proba(transform_test_matrix)

print(y_pred_test)

NE=(-1)/len(y_pred_test)*sum((1+y_test)/2*np.log(y_pred_test[:,1])+(1-y_test)/2*np.log(1-y_pred_test[:,1]))

print("loss"+str(NE))