from sklearn.metrics import roc_curve, auc
import glob	
import ntpath
import numpy as np
import scipy.io as io
b = io.loadmat('res.mat')
labels=b['lbl'][0]
testfiles = b['A']
print(np.shape(labels))
print(np.shape(testfiles))
fpr, tpr, _ = roc_curve(labels, testfiles, 0)
roc_auc = auc(fpr, tpr)
print(roc_auc)
f = open("svddgan.txt", "a")
f.write(str(roc_auc))
f.close()
