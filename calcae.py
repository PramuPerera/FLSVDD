import train
import testsvdd
from options.test_options import TestOptions
from options.train_options import TrainOptions
from sklearn.metrics import roc_curve, auc
import glob	
import ntpath
import numpy as np
import scipy.io as io
import os

f = open("svddgan.txt", "w")
f.close()
f = open("MSE.txt", "w")
f.close()
for it in [1]:#range(1):
	opt = TrainOptions().parse()
	opt.dataroot = 'nol'
	opt.batch_size = 256
	opt.fineSize = 32
	opt.input_nc = 1
	opt.output_nc = 1
	opt.ngf = 64 
	opt.name = 'cifar_AutoE'
	opt.lambda_A = 1
	opt.model = 'ae'
	#train.run(opt)
	for c in range(10):
		opt = TestOptions().parse()
		opt.model = 'ae'
		opt.dataroot = 'nol'
		opt.batch_size = 256
		opt.fineSize = 32
		opt.input_nc = 1
		opt.output_nc = 1
		opt.ngf =  64 
		opt.name = 'cifar_AutoE'+str(c)
		opt.cname = c
		MSE = testsvdd.run(opt)
		os.system('matlab -nodisplay -nosplash -nodesktop -r \"run ocgan.m; quit;\"')
		b = io.loadmat('res.mat')
		labels=b['lbl'][0]
		testfiles = b['A']
		fpr, tpr, _ = roc_curve(labels, testfiles, 0)
		roc_auc = auc(fpr, tpr)
		f = open("svddgan.txt", "a")
		f.write(str(c)+"     "+str(roc_auc)+"\n")
		f.close()
		f = open("MSE.txt", "a")
		f.write(str(c)+"     "+str(MSE)+"\n")
		f.close()
