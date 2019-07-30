import scipy.io as sio
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from dimension_reduction import elasticNet
import utils.tools as utils


from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json

start=time.time()
path1='gcforest4.json'
config = load_json(path1)
gc = GCForest(config)


mask_data= sio.loadmat('yeast_elastic_mask_scale_0.03_0.1.mat')
mask=mask_data.get('yeast_elastic_mask')
extraction= sio.loadmat('yeast_feature_end.mat')
proteinA=extraction.get('feature_A')
protein_A=np.array(proteinA)
proteinB=extraction.get('feature_B')
protein_B=np.array(proteinB)
X_=np.concatenate((protein_A,protein_B),axis=1)
X_=np.array(X_)
[row,column]=np.shape(X_)
label_P=np.ones(int(row/2))
label_N=np.zeros(int(row/2))
label_=np.hstack((label_P,label_N))
y_raw=np.mat(label_)
y_raw=np.transpose(y_raw)
y_=np.array(y_raw)
def get_shuffle(dataset,label,random_state):    
    #shuffle data
    np.random.seed(random_state)
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label  

X_shuffle,y=get_shuffle(X_,y_,random_state=1)
scaler = preprocessing.StandardScaler().fit(X_shuffle)
X_shuffle=scaler.transform(X_shuffle) 
train_dim=X_shuffle[:,mask]
train_shu=np.reshape(train_dim,(train_dim.shape[0],train_dim.shape[2]))
train_label=y


data_test= sio.loadmat('Wnt_feature_end.mat')
test_proteinA=data_test.get('feature_A')
test_protein_A=np.array(test_proteinA)
test_proteinB=data_test.get('feature_B')
test_protein_B=np.array(test_proteinB)
test_protein=np.concatenate((test_protein_A,test_protein_B),axis=1)
test_protein=np.array(test_protein)
test_protein=scaler.transform(test_protein) 
test_dim=test_protein[:,mask]
test_shu=np.reshape(test_dim,(test_dim.shape[0],test_dim.shape[2]))
[row1,column1]=np.shape(test_shu)
test_y_raw=np.ones(int(row1))

test_y_=np.mat(test_y_raw)
test_y=np.transpose(test_y_)
test_label=np.array(test_y)




with open("model_gc4.pkl", "rb") as f:
    gc = pickle.load(f)
y_score=gc.predict_proba(test_shu)
y_test=utils.to_categorical(test_label)    
y_class= utils.categorical_probas_to_classes(y_score)
y_test_tmp=test_label
accu= accuracy_score(y_test_tmp, y_class)
print(accu)
acc, precision,npv, sensitivity, specificity, mcc,f1= utils.calculate_performace(len(y_class), y_class, y_test_tmp)
sio.savemat('yeast_Wnt_class.mat', {'yeast_Wnt_class':y_class})






