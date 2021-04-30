import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def	categorical_to_numerical_conversion(x, rows_names):
	#Converting categorical data to numerical data
	#Fitting the data into label
	#Now we transform that data
	# -> transforming it from categorical to numerical values
	for name in rows_names:
		label = LabelEncoder()
		x[name] = label.fit_transform(x[name])
	#Avoiding the Dummy variable trap
	#	Creating 2 Geography columns instead of one
	#	Instead of aving values ranging from 0 to 2
	#	and aving the model considering that a value is
	#	greater than another, we make 2 columns with only 0 and 1
	#	values, the model will look at both geography columns instead
	#	of looking at only one
	x = pd.get_dummies(x, drop_first=True, columns=['Geography'])
	return x

def	data_preparation():
	data = pd.read_csv("Churn_Modelling.csv")
	#	Selecting the Independent and Dependent variables
	#	Dropping Independent variables on the columns axes
	#	axis = 0 means the ids, or the rows (y) and axis = 1 means
	#	the columns (x)
	x = data.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
	#Getting the labels from the dataset, aka dependent variables
	y = data['Exited']
	return data, x, y

def	data_scaling(x_train, x_test):
	#Scaling data so we don't have any scale difference
	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.fit_transform(x_test)
	return x_train, x_test

def	model_building(x_train, y_train):
	model = tf.keras.models.Sequential()
	#Input layer
	model.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim=11))
	#Hidden layer
	model.add(tf.keras.layers.Dense(units=6, activation='relu'))
	#Output layer, sigmoid activation because we want a binary output
	model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
	#The loss function act as a guide for the optimizer so the optimizer
	#	will move to the right direction and find the global minimum
	#We use accuracy as metrics because you are dealing with one output
	#	if we were to deal with multi outputs we would use sparse_categorical_accuracy
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
	print("\nModel compiled, informations about its structure below..\n")
	model.summary()
	return model

if __name__ == "__main__":
	#Data Preprocessing
	data, x, y = data_preparation()

	#Converting catagorical data to numerical data
	x = categorical_to_numerical_conversion(x, ['Geography', 'Gender'])

	#Splitting the dataset, setting random_state to 0 means we always get the same
	#	splitting, with random_state = 1 the splitting would be randomized
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

	#Feature scaling
	x_train, x_test = data_scaling(x_train, x_test)

	#Model building
	model = model_building(x_train, y_train)

	#Model training
	model.fit(x_train, y_train.to_numpy(), batch_size=10, epochs=20, verbose=2)

	#Model Evaluation and Prediction
	print("\n\n")
	test_loss, test_accuracy = model.evaluate(x_test, y_test.to_numpy())
	print("Test Accuracy : {:.2f}%".format(test_accuracy * 100))
	yhat = np.argmax(model.predict(x_test), axis=-1)

	#Confusion matrix
	y_test = y_test.to_numpy()
	cm = confusion_matrix(y_test, yhat)
	acc_cm = accuracy_score(y_test, yhat)
	print("Accuracy over the confusion matrix : {:.2f}".format(acc_cm * 100))))

