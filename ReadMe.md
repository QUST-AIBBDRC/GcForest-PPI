##GcForest-PPI

Prediction of protein-protein interactions based on elastic net and deep forest.

###GcForest-PPI uses the following dependencies:
* python 3.6 
* numpy
* scipy
* scikit-learn
* gcForst
gcForest is developed by Zhou et al.[1,2], which can be download from  http://lamda.nju.edu.cn/code_gcForest.ashx.

[1] Z.H. Zhou, J. Feng, Deep forest: towards an alternative to deep neural networks, 
     In: Proceedings of the 26th International Joint Conference on Artificial Intelligence, 2017, pp. 3553-3559.
[2] Z.H. Zhou, J. Feng, Deep forest, National Science Review 6 (2019) 74-86.

###Guiding principles:

**The dataset file contains the S. cerevisiae, H. pylori, the independent dataset and network dataset.

**Feature extraction
1) Evolutionary information: 
   obtain_pssm.py is the implementation of AAC-PSSM and DPC-PSSM.
2) Physicochemical_information: 
   PAAC_Yeast.m is the implementation of PAAC.
   Auto_yeast.m is the implementation of Auto.
3) Sequence_information:
   yeast_CTDC.py, yeast_CTDT.py, yeast_CTDD.py are the implementation of CTD
   ExMI.m is the implementation of MMI.
  
** Dimensional reduction:
   yeast_elastic_end.py represents the elastic.
   yeast_KPCA.py represents KPCA.
   yeast_LLE.py represents LLE.
   yeast_PCA.py represents PCA.
   yeast_SE.py represents SE.
   yeast_SSDR.py represents SSDR.
   yeast_TSVD.py represents SVD.

** Classifier:
   yeast_gcforest.py is the implementation of GcForest.
   yeast_Ad.py is the implementation of AdaBoost.
   yeast_KNN.py is the implementation of KNN.
   yeast_LR.py is the implementation of LR.
   yeast_NB.py is the implementation of NB.
   yeast_RF.py is the implementation of RF.
   yeast_SVM.py is the implementation of SVM.

** independent_test:
   The independent_test file contains the code of the test of independent dataset and network dataset. 
