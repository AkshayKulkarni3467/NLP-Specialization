import numpy as np
import pandas as pd
from svm import SVM_Classifier
from dataset import dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



features, targets = dataset('data/diabetes.csv') #dataset function is hardcoded to the dataset used for training.

model = SVM_Classifier(lr = 1e-3,epochs=1e4,lamb=1e-2)

X_train,X_test,Y_train,Y_test = train_test_split(features,targets,test_size = 0.2,random_state=2)

model.fit(X_train,Y_train)

X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_pred)

Y_pred = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test,Y_pred)

print("Accuracy score of training data : {:.2f} , Accuracy score of testing data : {:.2f}".format(training_data_accuracy, testing_data_accuracy))

