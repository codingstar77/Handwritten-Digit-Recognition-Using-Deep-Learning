import numpy as np
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import datasets


#getting MNIST of size 70k images
dataset = datasets.fetch_mldata("MNIST Original")
X = np.array(dataset.data)  #Our Features
y = np.array(dataset.target) #Our labels

X =  X.astype('float32') 

#getting Our Test Data
X_test,y_test = X[60000:], y[60000:]
 


#Normalizing Our Features in range 0 and 1

X_test = X_test /255

#loading out saved model
model = joblib.load('model.pkl')

#predicting Now
y_pred = model.predict(X_test)

print(classification_report(y_pred,y_test))








