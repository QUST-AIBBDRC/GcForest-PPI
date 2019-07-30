import numpy as np
import scipy.io as sio
import pickle as p



def average(matrixSum, seqLen):
    matrix_array = np.array(matrixSum)
    matrix_array = np.divide(matrix_array, seqLen)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [(np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1], )))]
    return matrix_average

def preHandleColumns(PSSM,STEP,ID):
    PSSM=PSSM.astype(float)
    matrix_final = [ [0.0] * 20 ] * 20
    matrix_final=np.array(matrix_final)
    seq_cn=np.shape(PSSM)[0]

    if ID==0:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j]+=(PSSM[k][i]*PSSM[k+STEP][j])

    elif ID==1:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j] += ((PSSM[k][i]-PSSM[k+STEP][j]) * (PSSM[k][i]-PSSM[k+STEP][j])/4.0)
    return matrix_final
def aac_pssm(input_matrix):
    seq_cn=float(np.shape(input_matrix)[0])
    aac_pssm_matrix=input_matrix.sum(axis=0)
    aac_pssm_vector=aac_pssm_matrix/seq_cn
    return aac_pssm_vector
def dpc_pssm(input_matrix):
    STEP = 1
    ID = 0
    matrix_final = preHandleColumns(input_matrix, STEP, ID)
    seq_cn = float(np.shape(input_matrix)[0])
    dpc_pssm_vector = average(matrix_final, seq_cn-STEP)
    return dpc_pssm_vector
f1=open('Matine_pssm_NB.data','rb')
pssm1=p.load(f1)
aac=[]
dpc=[]
for i in range(len(pssm1)):
    aac_pssm_obtain=aac_pssm(pssm1[i])
    aac.append(aac_pssm_obtain)
for i in range(len(pssm1)):
    dpc_pssm_obtain=dpc_pssm(pssm1[i])
    dpc.append(dpc_pssm_obtain)
aac=np.array(aac)
dpc=np.array(dpc)
sio.savemat('Matine_aac_pssm_NB.mat',{'aac_pssm_NB':aac})
sio.savemat('Matine_dpc_pssm_NB.mat',{'dpc_pssm_NB':dpc})