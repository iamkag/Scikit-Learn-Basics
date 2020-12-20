import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def Least_Squares():
    #Import csv and convert csv to an array

    concrete_infos=pd.read_csv("/Users/user/Desktop/DDPMS/texnnikes _mixanikis_mathisis/ergasia/Concrete_Data.csv", sep= ',')
    #print('concrete_infos',concrete_infos)

    # Specify the data
    data = concrete_infos.iloc[:, :-1].values
    #print('data',data)

    # Specify the target labels and flatten the array
    output = concrete_infos.iloc[:, -1].values
    output = output.reshape(len(output),1)
    #print('Output',output)


    # Split the data up in train and test sets
    from sklearn.model_selection import train_test_split

    data_train, data_test, output_train, output_test = train_test_split(data, output, test_size=0.30, shuffle=False)

    unscaled_data_test=data_test
    #Scale the data
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train= scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Building Least Square Model
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(data_train, output_train)
    y_pred_test =lin_reg.predict(data_test)
    y_pred_train =lin_reg.predict(data_train)

    data_test_one_col=[]
    for i in range(len(unscaled_data_test)):
        data_test_one_col.append(unscaled_data_test[i][0])

    print(max(data_test_one_col))
    print(min(data_test_one_col))

    #print(data_test_one_col,len(data_test_one_col))
    # Plot outputs
    plt.scatter(data_test_one_col, output_test,  color='black')
    plt.scatter(data_test_one_col, y_pred_test, color='blue')

    plt.xticks((np.arange(100, 650, step=50)))
    plt.yticks((np.arange(0, 120, step=20)))
    plt.title('Least Square Model')
    plt.xlabel('Concrete category')
    plt.ylabel('Consrete strenght')
    plt.legend(("Actual", "Predict"))
    plt.show()

Least_Squares()