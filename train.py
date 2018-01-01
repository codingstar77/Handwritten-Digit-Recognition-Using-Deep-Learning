from sklearn.externals import joblib
from sklearn import datasets
import numpy as np
from sklearn.neural_network.multilayer_perceptron import  MLPClassifier


dataset = datasets.fetch_mldata("MNIST Original")
X = numpy.array(dataset.data) 
y = numpy.array(dataset.target)
X =  X.astype('float32')


X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


X_train = X_train /255
X_test = X_test /255
