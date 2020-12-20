import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape_cal(real,pred):
    sum = 0
    N = len(real)
    for i in range(len(real)):
        sum = sum + (abs(real[i]-pred[i])/real[i])
    sum = sum/N
    return sum

def Ridge():
    #Import csv and convert csv to an array
    concrete_infos=pd.read_csv("/Users/user/Desktop/DDPMS/texnnikes _mixanikis_mathisis/ergasia/Concrete_Data.csv", sep= ',')
    #print('concrete_infos',concrete_infos)

    # Specify the data
    data = concrete_infos.iloc[:, :-1].values
    #print('data',data)

    # Specify the target labels and flatten the array
    output = concrete_infos.iloc[:, -1].values
    #output = output.reshape(len(output),1)
    #print('Output',output)

    # Split the data up in train and test sets
    from sklearn.model_selection import train_test_split
    data_train, data_test, output_train, output_test = train_test_split(data, output, test_size=0.30, shuffle=True)

    #Scale the data
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train= scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Building Least Square Model
    from sklearn.linear_model import Ridge
    regressor = Ridge(alpha=0.0001)
    regressor.fit(data_train, output_train)
    y_pred_test =regressor.predict(data_test)
    y_pred_train = regressor.predict(data_train)

    # The mean squared error
    mse_test = mean_squared_error(output_test, y_pred_test)
    # The mean absolute error
    mae_test = mean_absolute_error(output_test, y_pred_test)
    mape_test = mape_cal(output_test, y_pred_test)

    # The mean squared error
    mse_train = mean_squared_error(output_train, y_pred_train)
    mae_train = mean_absolute_error(output_train, y_pred_train)
    # The mean absolute error
    mape_train = mape_cal(output_train, y_pred_train)


    return mse_train, mae_train, mape_train, mse_test, mae_test, mape_test


def main():
    mse_train = 0
    mae_train = 0
    mape_train = 0
    mse_test = 0
    mae_test = 0
    mape_test = 0
    n = 10
    for i in range(n):
        mse_train += Ridge()[0]
        mae_train += Ridge()[1]
        mape_train += Ridge()[2]
        mse_test += Ridge()[3]
        mae_test += Ridge()[4]
        mape_test += Ridge()[5]

    mse_train = mse_train / n
    mae_train = mae_train / n
    mape_train = mape_train / n
    mse_test = mse_test / n
    mae_test = mae_test / n
    mape_test = mape_test / n

    print("ΤΡΑΙΝ:",mse_train, mae_train, mape_train)
    print("TEST:", mse_test, mae_test, mape_test)

if __name__ == "__main__":
    main()