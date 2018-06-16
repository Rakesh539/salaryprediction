# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:26:27 2018

@author: vprayagala

Goal - Predcit the salary based on new job posted

Methods Used - Linear Regression, CART, RF and AB
Metric - RMSE (Root Mean Square Eror)

Assumptions : 1) Discard nulls (oher option is impute them)  if any , but no nulls found. 
              2) Check exreme observations presence and do not eliminate these
              3) Need to build regression in quick time with less emphasis 
                 on pre-processing/data analysis
              4) Limited Parameter Tuning
              5) Traditional ML rather neural nets
              6) Prediction over exlpainability/inference
"""
#%%
#Import the required Packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
#%%
#Read Data train_salaries_2013-03-07
train_raw_feat=pd.read_csv(".\\indeed\\data\\train_features_2013-03-07.csv")
train_raw_sal=pd.read_csv(".\\indeed\\data\\train_salaries_2013-03-07.csv")
test_raw_feat=pd.read_csv(".\\indeed\\data\\test_features_2013-03-07.csv")

#Check the train/test data shape and data types
print("Training Feature has {} rows and {} features".format(train_raw_feat.shape[0],train_raw_feat.shape[1]))
print("Training Salary has {} rows and {} features".format(train_raw_sal.shape[0],train_raw_sal.shape[1]))
print("Test Feature has {} rows and {} features".format(test_raw_feat.shape[0],test_raw_feat.shape[1]))

print("Training Feature Types:\n{}".format(train_raw_feat.dtypes))
print("Training Salary Types:\n{}".format(train_raw_sal.dtypes))
print("Test Feature Types:\n{}".format(test_raw_feat.dtypes))
#%%
#Combine Training Features and Sal
train_data=pd.merge(train_raw_feat,train_raw_sal,how='inner',on='jobId')
#%%
#Preprocess Data
#1. jobId is a unique identiier and it can be dropped
#2. Check nulls and basic cenral properties for attributes
#3. Standardize (used MinMaxScaler) numerical features - yearsExperience,milesFromMetropolis
#4. Label encode the features comapnyId,jobTypedegree, major,industry
#5. Basic Plots - Check how are these are going with target atribute salary
#6. Check the Correlation
#7. Split the data into train/test. We could use train/test/validation set, 
##  the additional validation set for parameter tuning.  
#8 Done Grid Search for tuning and used cross validation for parameter tuning 
# Grid search is not extensive just limited to number of trees and number of features


#This block is to define all the functions used here

#1. Pre-processing function - to be used for train data and test data
def preProcess(in_data):
    data=in_data.copy()
    data.drop(['jobId'],axis=1,inplace=True)
    data.isnull().sum()
    data.describe()
    
    #Scale the numeric attributes
    minmax_scaler = MinMaxScaler()
    for i,typ in enumerate(data.dtypes):
        if typ == 'int64':
            col_name=data.dtypes.index[i]
            if col_name == 'salary':
                print("Not Scaling target")
            else:
                print("Sclaing Numeric Attribute:{}".format(col_name))
                minmax_scaler.fit(data[col_name].values.reshape(-1,1))
                data[col_name]=minmax_scaler.transform(data[col_name].values.reshape(-1,1))
    #Label Encode attributes. We could create le object for each attribute incase if we need 
    # to inverse transform these later. I am going with one object
    le=LabelEncoder()
    for i,typ in enumerate(data.dtypes):
        if typ == 'O':
            col_name=data.dtypes.index[i]
            print("Encoding Categorical Attribute:{}".format(col_name))
            data[col_name] = le.fit_transform(data[col_name])
    
    #check data type
    print("Transformed Data:\n{}".format(data.dtypes))

    return data  

#2. Create different models on the same train/test split
def build_model(TrainX,TrainY,TestX,TestY,seed):
    
    models=[]
    models.append(('LR',LinearRegression()))
    models.append(('CART',DecisionTreeRegressor(random_state=seed)))
    #models.append(('SVM',SVR())) - Computationally Epensive and not converging
    models.append(('RF', RandomForestRegressor(random_state=seed))) 
    models.append(('AB', AdaBoostRegressor(random_state=seed)))  
    
    results={}
    for name,model in models:
        mdl=model.fit(TrainX,TrainY)
        temp=[]
        #calculate the score - use Root Mean Square Error
        y_pred=mdl.predict(TrainX)
        train_score=np.sqrt(mean_squared_error(TrainY,y_pred))
        temp.append(train_score)
        
        y_pred=mdl.predict(TestX)
        test_score=np.sqrt(mean_squared_error(TestY,y_pred))
        temp.append(test_score)
        
        train_r2=mdl.score(TrainX,TrainY)
        temp.append(train_r2)
        
        test_r2=mdl.score(TestX,TestY)
        temp.append(test_r2)
        msg = "%s:(%f) (%f) (%f) (%f)" % (name,train_score,test_score,train_r2,test_r2)
        print(msg)
        results[name]=temp
    return results

#3. Function to calculate error metric Root Mean Sqaure Error
def RMSE(est,xtest,ytest):
    rmse = np.sqrt(mean_squared_error(ytest, est.predict(xtest)))
    print("RMSE: {}".format(rmse))
    return rmse

#4. Function to fine tune RF 
def tune_rf_model(TrainX,TrainY,TestX,TestY,seed):
    #Grid Search for Random Forest
    n_est=[30,20,10]
    max_feat=['auto','sqrt','log2']

    param_grid = dict(n_estimators=n_est,\
                      max_features=max_feat
                      )
    model = RandomForestRegressor(random_state=seed)
    kfold = KFold(n_splits=2, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=RMSE,cv=kfold)
    grid_result = grid.fit(TrainX, TrainY)
    print("Best: %f using %s" % (grid_result.best_score_, 
                                 grid_result.best_params_))
    train_score=grid_result.best_score_
    #Build the final model - going with RandomForest, because of the less number of samples for
    #negative cases
    RF_Model=RandomForestRegressor(max_features=grid_result.best_params_['max_features'],\
                                    n_estimators=grid_result.best_params_['n_estimators'],\

                                    )
    RF_Model.fit(TrainX,TrainY)
    test_score=np.sqrt(mean_squared_error(RF_Model.predict(TestX),TestY))
    
    return RF_Model,train_score,test_score

#End of function block  
#%%
# Checked the exreme values using boxplot. No action has been taken on these values
# We could drop observations with zero values or fill them with mean value

#Check the mean salary for each company , average around 115-116, but there are outliers for few companies where salry is zero
#the mean is little low due to these extreme values
train_data.boxplot(column='salary',by='companyId',figsize=(60,20))
plt.figure(figsize=(60,20))
train_data[['companyId','salary']].groupby('companyId').agg({'salary':'mean'}).plot(kind='bar',figsize=(60,20))
#%%
# jobType vs salary
train_data.boxplot(column='salary',by='jobType',figsize=(15,10))
train_data[['jobType','salary']].groupby('jobType').agg({'salary':'mean'}).plot(kind='bar',figsize=(15,10))
#%%
# degree vs salary
train_data.boxplot(column='salary',by='degree',figsize=(15,10))
train_data[['degree','salary']].groupby('degree').agg({'salary':'mean'}).plot(kind='bar',figsize=(15,10))
#%%
# major vs salary
train_data.boxplot(column='salary',by='major',figsize=(10,10))
train_data[['major','salary']].groupby('major').agg({'salary':'mean'}).plot(kind='bar',figsize=(10,10))
#%%
# industry vs salary
train_data.boxplot(column='salary',by='industry',figsize=(15,10))
train_data[['industry','salary']].groupby('industry').agg({'salary':'mean'}).plot(kind='bar',figsize=(15,10))
#%%
#
#There is an increasing trend with mean salary as experience is more
plt.figure(figsize=(15,10))
plt.scatter(train_data['yearsExperience'],train_data['salary'])
train_data[['yearsExperience','salary']].groupby('yearsExperience').agg({'salary':'mean'}).plot(kind='bar')
#%%
#bin the miles and check the avergage salary, farther from metro have lessser pay
plt.figure(figsize=(15,10))
plt.scatter(train_data['milesFromMetropolis'],train_data['salary'])

custom_bucket_array = np.linspace(0, np.max(train_data['milesFromMetropolis']), 20)
buk_data=pd.cut(train_data['milesFromMetropolis'], custom_bucket_array)
train_data[['milesFromMetropolis','salary']].groupby(buk_data).agg({'salary':'mean'}).plot(kind='bar')
#%%
#Check Correlation matrix, negative correlation with miles
train_data.corr()
#%%
#Preprocess Data and split into train/test
train_data=preProcess(train_data)
X=train_data.loc[:,train_data.columns != 'salary']
y=train_data.loc[:,train_data.columns == 'salary']

seed=7
X_train, X_test, y_train, y_test = train_test_split(\
                X.values, y.values, test_size=0.33, random_state=seed)
#%%
#Start Working with model building
result=build_model(X_train,y_train.ravel(),X_test, y_test.ravel(),seed)
#%%
#Visualize the different model pefomances on same train/test set
res_df=pd.DataFrame.from_dict(result,orient='index')
res_df.columns=['Train_RMSE','Test_RMSE','Train_R2','Test_R2']
res_df.sort_index(inplace=True)
#ax=res_df.plot()
plt.figure(figsize=(7,7))
plt.plot(res_df['Train_RMSE'],color='red')
plt.plot(res_df['Test_RMSE'],color='blue')

red_patch = mpatches.Patch(color='red', label='Train RMSE')
blue_patch = mpatches.Patch(color='blue', label='Test RMSE')

plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("Model Name")
plt.ylabel("Root Mean Square Error")
plt.title("Model Performance Comparision")
#%%
plt.figure(figsize=(7,7))
plt.plot(res_df['Train_R2'],color='red')
plt.plot(res_df['Test_R2'],color='blue')

red_patch = mpatches.Patch(color='red', label='Train R Square')
blue_patch = mpatches.Patch(color='blue', label='Test R Square')

plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("Model Name")
plt.ylabel("R Square")
plt.title("Model Performance Comparision")
#%%
#From above plot we an see hat RF is performing better on test data. CARoverfitting
#in which train error is very less and test error is more
#We can tune RF parameters further
#Fine tune model - tune number of trees and number of estimators
#Please not this takes time as it has to try different combination of parm grid

RF_Model,train_scoe,test_score=tune_rf_model(X_train,y_train.ravel(),X_test, y_test.ravel(),seed)
feature_importances = pd.DataFrame(RF_Model.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).\
                                   sort_values('importance',ascending=False)
print("Atribute Importance:\n{}".format(feature_importances))
#%%
#Test Data - pre-processing using function, then pass test data to model
#for predictions. Create a file for predictions and save it to disk
#Retain test job Id
test_result=pd.DataFrame(columns=['jobId','salary'])
test_result.loc[:,'jobId']=test_raw_feat.loc[:,'jobId']
test_data=preProcess(test_raw_feat)

test_pred=RF_Model.predict(test_data.values)
test_result.loc[:,'salary']=pd.Series(test_pred,index=test_data.index)
test_result.to_csv("./indeed/AssignmentResults/test_salaries.csv",index=False)

