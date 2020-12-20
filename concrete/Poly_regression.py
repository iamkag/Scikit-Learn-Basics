import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statistics import mean , stdev

def fit_poly_regr(degree,data_to_fit):
    degree += 1
    data_pol = []
    for element in data_to_fit:
        #print("Element", element)
        #print(len(element))
        new_data = []
        for i in range(len(element)):
            for j in range(1, degree):
                x = (element[i] ** j)
                new_data.append(x)
        data_pol.append(new_data)

    return data_pol

def mape_cal(real,pred):
    sum = 0
    N = len(real)
    for i in range(len(real)):
        sum = sum + (abs(real[i]-pred[i])/real[i])
    sum = sum/N
    return sum

def MachineLearnig(data,output,degree):
    # Split the data up in train and test sets
    from sklearn.model_selection import train_test_split
    data_train, data_test, output_train, output_test = train_test_split(data, output, test_size=0.30, shuffle=True)

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Polyonimal regression at features
    #print("Data train", data_train)
    data_train_poly = fit_poly_regr(degree,data_train)

    #print("Data test", data_test)
    data_test_poly = fit_poly_regr(degree, data_test)
    #print("Data poly test", data_test_poly)

    # Building Least Square Model
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    regressor = LinearRegression()
    regressor.fit(data_train_poly, output_train)
    y_pred_test =regressor.predict(data_test_poly)
    y_pred_train = regressor.predict(data_train_poly)

    # The mean squared error
    mse_train = mean_squared_error(output_train, y_pred_train)
    # The mean absolute error
    mae_train = mean_absolute_error(output_train, y_pred_train)
    # The mean absolute percentance error
    mape_train = mape_cal(output_train, y_pred_train)

    # The mean squared error
    mse_test = mean_squared_error(output_test, y_pred_test)
    # The mean absolute error
    mae_test = mean_absolute_error(output_test, y_pred_test)
    # The mean absolute percentance error
    mape_test = mape_cal(output_test,y_pred_test)

    #print(mse_test,mae_test,mape_test)
    #print(mse_train,mae_train,mape_train)
    return mse_train,mae_train,mape_train,mse_test,mae_test,mape_test

def main():

    # Import csv and convert csv to an array
    concrete_infos = pd.read_csv("/Users/user/Desktop/DDPMS/texnnikes _mixanikis_mathisis/ergasia/Concrete_Data.csv", sep=',')
    #print('concrete_infos', concrete_infos)

    # Specify the data
    data = concrete_infos.iloc[:, :-1].values
    #print('data', data)

    # Specify the target labels and flatten the array
    output = concrete_infos.iloc[:, -1].values
    output = output.reshape(len(output), 1)
    #print('Output', output)

    mse_list_train = []
    mae_list_train = []
    mape_list_train = []
    mse_list_test = []
    mae_list_test = []
    mape_list_test = []
    n=10

    for degree in range(1, 11):
        print("Degree", degree)
        mse_train = 0
        mae_train = 0
        mape_train = 0
        mse_test = 0
        mae_test = 0
        mape_test = 0
        count = 0

        for i in range(n):
            mse_train += MachineLearnig(data,output,degree)[0]
            mae_train += MachineLearnig(data,output,degree)[1]
            mape_train += MachineLearnig(data,output,degree)[2]
            mse_test += MachineLearnig(data,output,degree)[3]
            mae_test += MachineLearnig(data,output,degree)[4]
            mape_test += MachineLearnig(data,output,degree)[5]
            count += 1

        print("count",count)
        mse_list_train.append(mse_train/n)
        mae_list_train.append(mae_train/n)
        mape_list_train.append(mape_train/n)

        mse_list_test.append(mse_test/n)
        mae_list_test.append(mae_test/n)
        mape_list_test.append(mape_test/n)


    print('Train', mse_list_train)
    print(mae_list_train)
    print(mape_list_train)

    print('Test', mse_list_test)
    print(mae_list_test)
    print(mape_list_test)

main()