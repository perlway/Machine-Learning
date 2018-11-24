# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 06:57:06 2018

@author: vk186043
"""

#!/usr/bin/python

import os
import sys
import numpy as np
import sys, getopt
from PIL import Image
import matplotlib.pyplot as plt
from pylab import imshow, show
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
#'''
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#'''



def load_CIFAR_batch(filename):
	with open(filename, 'r') as f:
		datadict = unpickle(filename)
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
		Y = np.array(Y)
		return X, Y

def load_CIFAR10(ROOT):
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
		X, Y = load_CIFAR_batch(f)
		xs.append(X)
		ys.append(Y)    
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X, Y
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
	return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=10000, show_sample=True):

	cifar10_dir = '/home/cloudera/Machine-Learning/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

	mask = range(num_training, num_training + num_val)
	X_val = X_train[mask]
	y_val = y_train[mask]
	mask = range(num_training)
	X_train = X_train[mask]
	y_train = y_train[mask]
	mask = range(num_test)
	X_test = X_test[mask]
	y_test = y_test[mask]

	return X_train, y_train, X_val, y_val, X_test, y_test

def preprocessing_CIFAR10_data(X_train,X_test):
	X_train = StandardScaler().fit_transform(X_train)
	X_test  = StandardScaler().fit_transform(X_test)
	return X_train,X_test 

def perform_reduction(reduction,num,X_train,X_test,y_train_raw):
	if reduction == "pca":
		pca = PCA(n_components=num)
		X_r = pca.fit(X_train).transform(X_train)
		X_test_r = pca.fit(X_test).transform(X_test)
	elif reduction == "lda":
		lda = LinearDiscriminantAnalysis(n_components=num)
		X_r = lda.fit_transform(X_train, y_train_raw)
		X_test_r = lda.transform(X_test)
	return X_r, X_test_r

def perform_classification(classifier,X_train,X_test,y_train_raw,y_test_raw,args):
	if classifier == 'logistic':
		solver='sag'
		c=1
		tol=0.0001
		multiclass='multinomial'
		max_iter=200
		args_list = args.split(',')
		for i in range(len(args_list)):
			#print("")
			parameter = args_list[i].split('=')
			print("parameter",parameter[0],parameter[1])
			if parameter[0] == 'solver':
				solver = parameter[1]
				print("argument :",solver)
			elif parameter[0] == 'c':
				c = float(parameter[1])
				print("argument :",c)
			elif parameter[0] == 'tol':
				tol = float(parameter[1])
				print("argument :",tol)
			elif parameter[0] == 'multiclass':
				multiclass = parameter[1]
				print("argument :",multiclass)
			elif parameter[0] == 'max_iter':
				max_iter = int(parameter[1])
				print("argument :",max_iter)
		print("arguments :",solver,c,tol,multiclass,max_iter)
		clf = LogisticRegression(solver=solver,multi_class=multiclass,max_iter=max_iter,C=c,tol=tol).fit(X_train, y_train_raw)
		y_pred = clf.predict(X_test)
		score = accuracy_score(y_test_raw,y_pred)
		print("accuracy : ", score)
	elif classifier == 'svm':
		kernel='rbf'
		c=0.0001
		cache_size=200
		args_list = args.split(',')
		for i in range(len(args_list)):
			parameter = args_list[i].split('=')
			if parameter[0] == 'kernel':
				kernel = parameter[1]
			elif parameter[0] == 'c':
				c = float(parameter[1])
				print("argument :",c)
			elif parameter[0] == 'cache_size':
				cache_size = float(parameter[1])
				print("argument :",cache_size)
		lsvm = SVC(C=c,kernel=kernel,cache_size=cache_size)
		lsvm.fit(X_train,y_train_raw)
		y_pred = lsvm.predict(X_test)
		score = accuracy_score(y_test_raw,y_pred)
		print("accuracy : ", score)
	elif classifier == 'mlp':
		solver='adam'
		hidden_layer_sizes=[]
		batch_size=200
		learning_rate ='constant'
		activation='logistic'
		max_iter=200
		args_list = args.split(',')
		for i in range(len(args_list)):
			parameter = args_list[i].split('=')
			if parameter[0] == 'solver':
				solver = parameter[1]
			elif parameter[0] == 'hidden_layer_sizes':
				print("map:",parameter[1])
				hidden_layer_sizes = list(map(int,parameter[1].split(':')))
				print("argument :",hidden_layer_sizes)
			elif parameter[0] == 'batch_size':
				batch_size = int(parameter[1])
				print("argument :",batch_size)
			elif parameter[0] == 'learning_rate':
				learning_rate = parameter[1]
			elif parameter[0] == 'activation':
				activation = parameter[1]
			elif parameter[0] == 'max_iter':
				max_iter = int(parameter[1])
		clf = MLPClassifier(solver=solver, alpha=1e-5,hidden_layer_sizes=hidden_layer_sizes, random_state=1,batch_size=batch_size,learning_rate =learning_rate,activation=activation,max_iter=max_iter)
		clf.fit(X_train,y_train_raw)
		train_score = clf.score(X_train,y_train_raw)
		test_score = clf.score(X_test,y_test_raw)
		print("test accuracy : ", test_score)
		print("train accuracy : ", train_score)
	elif classifier == 'dtree':
		criterion='gini'
		presort=False
		max_depth=None
		args_list = args.split(',')
		for i in range(len(args_list)):
			parameter = args_list[i].split('=')
			if parameter[0] == 'criterion':
				criterion = parameter[1]
			elif parameter[0] == 'presort':
				presort = bool(parameter[1])
			elif parameter[0] == 'max_depth':
				max_depth = int(parameter[1])
		clf = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion,presort=presort)
		clf.fit(X_train,y_train_raw)
		test_score = clf.score(X_test,y_test_raw)
		print("test accuracy : ", test_score)








def main(argv):
	reduction = ''
	classifier = ''
	args = ''
	
	try:
		opts, args = getopt.getopt(argv,"hr:c:a:n:",["reduction=","classifier=","arguments=","no_of_components="])
	except getopt.GetoptError:
		print ("test.py -r <pca/lda/none> -c <logistic/dtree/svm/ksvm/mlp> -a <arguments to the classifier> -n <number of components>")
		sys.exit(2)
		
	for opt, arg in opts:
		if opt == '-h':
			print ("test.py -r <pca/lda/none> -c <logistic/dtree/svm/ksvm/mlp> -a <arguments to the classifier> -n <number of components>")
			sys.exit()
		elif opt in ("-r", "--reduction"):
			reduction = arg
		elif opt in ("-c", "--classifier"):
			classifier = arg
		elif opt in ("-a","--arguments"):
			args = arg
		elif opt in ("-n","--no_of_components"):
			num = int(arg)
	X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
	X_gray = np.empty(shape=(49000,1024))
	X_gray_test = np.empty(shape=(10000,1024))
	
	for iter in range(49000):
                image_3d = X_train_raw[iter,:].reshape(3,32,32).transpose(1,2,0)
                temp = rgb2gray(image_3d).flatten()
                X_gray[iter,:] = temp[:]
                if iter < 10000:
                        image_3d = X_test_raw[iter,:].reshape(3,32,32).transpose(1,2,0)
                        temp = rgb2gray(image_3d).flatten()
                        X_gray_test[iter,:] = temp[:]
	X_train, X_test = preprocessing_CIFAR10_data(X_gray, X_gray_test)
	X_train,X_test = perform_reduction(reduction,num,X_train,X_test,y_train_raw)
	perform_classification(classifier,X_train,X_test,y_train_raw,y_test_raw,args)

if __name__ == "__main__":
	main(sys.argv[1:])
