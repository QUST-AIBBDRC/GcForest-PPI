
import scipy.io as sio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc
from dimension_reduction_com import elasticNet
import utils.tools as utils



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
X_=scale(X_)
X_shuffle,y=get_shuffle(X_,y_,random_state=1)
mask_data= sio.loadmat('yeast_elastic_mask_scale_0.03_0.1.mat')
mask=mask_data.get('yeast_elastic_mask')
X_dim=X_shuffle[:,mask]
X_initial=np.reshape(X_dim,(X_dim.shape[0],X_dim.shape[2]))
X=X_initial
#y_raw=np.mat(label_)
#y=np.transpose(y_raw)
#X_train_origin, X_test_origin, y_train, y_test = train_test_split(X, y,test_size=0.2)
cv_clf = SVC(probability=True)
skf= StratifiedKFold(n_splits=5)
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
for train, test in skf.split(X,y): 
    X_train_enc=cv_clf.fit(X[train], y[train])
    y_score=cv_clf.predict_proba(X[test])
    yscore=np.vstack((yscore,y_score))
    y_test=utils.to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('SVM:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))

result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
#column=data.shape[1]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yeast_SVM_yscore.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('yeast_SVM_ytest.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('yeast_SVM_result.csv')
interval = (time.time() - start)
print("Time used:",interval)


