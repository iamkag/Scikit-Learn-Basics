import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import time

#Import csv and convert csv to an array
forest=pd.read_csv("/Users/user/Desktop/DDPMS/texnnikes _mixanikis_mathisis/ergasia/covtype.data", sep= ',')

# Specify the data
data = forest.iloc[:, :-1].values

# Specify the target labels and flatten the array
output = forest.iloc[:, -1].values
#output = output.reshape(len(output),1)

# Split the data up in train and test sets
from sklearn.model_selection import train_test_split
data_train, data_test, output_train, output_test = train_test_split(data, output, test_size=565893, shuffle=False)
print("LEN train", len(data_train))
unscaled_data_test= data_test

#Feature scaling
from sklearn.preprocessing import MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)


#Logistic Regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
time_start = time.time()
classifier = LinearDiscriminantAnalysis()
classifier.fit(data_train, output_train)

#Predict result
y_pred_test = classifier.predict(data_test)
y_pred_train = classifier.predict(data_train)
final_time = time.time() - time_start
print("EXECUTION TIME {} sec".format(final_time))

acur_test = accuracy_score(output_test, y_pred_test)
print("Acur test", acur_test)
acur_train = accuracy_score(output_train, y_pred_train)
print("Acur train", acur_train)

data_test_one_col=[]
for i in range(len(unscaled_data_test)):
    data_test_one_col.append(unscaled_data_test[i][0])
#
print(max(data_test_one_col))
print(min(data_test_one_col))
#
plt.scatter(data_test_one_col, output_test,  color='black',linewidths=10)
plt.scatter(data_test_one_col, y_pred_test, color='blue',linewidths=0.05)
#
plt.xticks((np.arange(1500, 4000, step=200)))
plt.yticks((np.arange(0, 8, step=1)))
plt.title('LDA')
plt.xlabel('Elevation')
plt.ylabel('Forest type')
plt.legend(("Actual","Predicted"))
plt.show()