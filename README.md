<h1> Python Import Libraries | Predict Missing Windspeed Values | RMSLE Error Function </h1>


```python
import pandas as pd
import numpy as np
import datetime 
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor 
import math
os.chdir('C:/Users/Shashi Suman/Downloads/')

######## Finding Missing Windspeed Values ##########
def missing_values(cf, val):
    from sklearn.ensemble import RandomForestRegressor as Forest
    from sklearn.model_selection import GridSearchCV

    temp = cf.copy()
    temp = temp.reset_index()
    temp.drop('index', axis=1, inplace=True)
    temp.fillna(-1, inplace=True)
    X_wind = temp[temp['windspeed'] != -1]
    T_wind = temp[temp['windspeed'] == -1]
    Y_wind = X_wind['windspeed']
    X_wind.drop(['datetime','holiday','workingday','windspeed','count'], axis=1, inplace=True)
    T_wind.drop(['datetime','holiday','workingday','windspeed','count'], axis=1, inplace=True)

    param_dist = {"max_depth": [1,3,5,10,15],
                  "min_samples_split" : [2,4,6,8],
                  "n_estimators": [10,20,30,40,50,60]}

    rf = Forest(n_jobs=-1)
    if val:
        clf = GridSearchCV(rf, param_dist, cv = 10, scoring='neg_mean_squared_log_error', n_jobs=-1, verbose=0)
        clf.fit(X_wind, Y_wind)
        print(clf.best_params_)
        rf = Forest(n_estimators = clf.best_params_['n_estimators'], max_depth = clf.best_params_['max_depth'], min_samples_split = clf.best_params_['min_samples_split'])
    else:
        rf = Forest(n_estimators = 40, max_depth = 5, min_samples_split = 2, n_jobs=-1)
    #rf = Forest(n_estimators = clf.best_params_['n_estimators'], max_depth = clf.best_params_['max_depth'], min_samples_split = clf.best_params_['min_samples_split'])
    rf.fit(X_wind, Y_wind)
    blank_windspeed = []
    val_ = rf.predict(T_wind)
    for index, row in temp.iterrows():
        if row['windspeed'] == -1:
            blank_windspeed.append(index)
    for i in range(len(blank_windspeed)):
        try:
            temp.iloc[blank_windspeed[i], 8] = val_[i]
        except IndexError:
            print(i, blank_windspeed[i])
            break
    return temp

def load(file):
    df = pd.read_csv(file)
    df['hour'], df['year'], df['month'], df['mday'], df['wday'] = 0,0,0,0,0
    l = len(df.columns)-5
    for i, r in df.iterrows():
        df.iloc[i, l] = datetime.datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').hour
        df.iloc[i, l+1] = datetime.datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').year
        df.iloc[i, l+2] = datetime.datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').month
        df.iloc[i, l+3] = datetime.datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').day
        df.iloc[i, l+4] = datetime.datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').weekday()
    df['weather'] = df['weather'].replace(4, 3)
    return df

def rmsle(y, y_pred):
    y_pred = np.where(y_pred < 0, 0, y_pred)
    u = y.copy()
    assert len(u) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(u[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(u))) ** 0.5
```

<h1> Load Training Set and Testing Set </h1>


```python
train = missing_values(load('train.csv'), False)
cf = train.copy()
#sns.set_style("darkgrid")

x, y, z = np.log1p(cf['count']), np.log1p(cf['casual']), np.log1p(cf['registered'])
cf.drop(['count','datetime'], axis=1, inplace=True)
cf['count'], cf['casual'], cf['registered'] = x, y, z

Train, Test = cf.copy(), load('test.csv')
iter_test = Test.copy()
```

<h1><center>FEATURE ENGINEERING</center></h1>

<h2> Find starting and ending index of each month </h2>


```python
start_end, year, month, start_end_, col = np.zeros((2, 24)), pd.unique(cf['year']), pd.unique(cf['month']), np.zeros((2, 24)), 0
for i in year:
    for j in month:
        start_end[0, col] = min(np.where((cf['month'].values == j) & (cf['year'] == i))[0])
        start_end[1, col] = start_end[0, col] + Train[(Train['year'].astype('int')==i) & (Train['month'].astype('int')==j)].shape[0]
        start_end_[0, col] = min(np.where((iter_test['month'].values == j) & (iter_test['year'] == i))[0])
        start_end_[1, col] = start_end_[0, col] + iter_test[(iter_test['year'].astype('int')==i) & (iter_test['month'].astype('int')==j)].shape[0]
        col += 1        
```

<h2> Find Peak Hours in both training and testing sets </h2>


```python
for i in [Train, Test]:
    i.loc[(i['hour'] > 9) & (i['hour'] < 20), 'busy_casual'], i.loc[(i['workingday'] == 1) & ((i['hour'] == 8) | ((i['hour'] > 16) & (i['hour'] < 19))), 'busy_reg'] = 1, 11
    i.loc[(i['workingday'] == 0) & ((i['hour'] > 9) & (i['hour'] < 20)), 'busy_reg'] = 1
    i.fillna(0, inplace=True)
```

<h2>Find weekend parameter</h2>


```python
Train['weekend'] = Train['wday'].apply(lambda x: 1 if (x==5) | (x==6) else 0)
Test['weekend'] = Train['wday'].apply(lambda x: 1 if (x==5) | (x==6) else 0)
```

<h2>Additional Holidays and WorkingDays </h2>


```python
Train.loc[(Train['mday']==15) & (Train['month']==4) & (Train['year']==2011), "workingday"] = 1
Train.loc[(Train['mday']==16) & (Train['month']==4) & (Train['year']==2012), "workingday"] = 1
Train.loc[(Train['mday']==25) & (Train['month']==11) & (Train['year']==2011), "workingday"] = 0
Train.loc[(Train['mday']==23) & (Train['month']==11) & (Train['year']==2012), "workingday"] = 0
Train.loc[(Train['mday']==15) & (Train['month']==4) & (Train['year']==2011), "holiday"] = 0
Train.loc[(Train['mday']==16) & (Train['month']==4) & (Train['year']==2012), "holiday"] = 0
Train.loc[(Train['mday']==25) & (Train['month']==11) & (Train['year']==2011), "holiday"] = 1
Train.loc[(Train['mday']==23) & (Train['month']==11) & (Train['year']==2012), "holiday"] = 1
Train.loc[(Train['mday']==21) & (Train['month']==5) & (Train['year']==2012), "holiday"] = 1
Train.loc[(Train['mday']==1) & (Train['month']==6) & (Train['year']==2012), "holiday"] = 1
```

<h2>Training Data Sample </h2>


```python
Train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>hour</th>
      <th>year</th>
      <th>month</th>
      <th>mday</th>
      <th>wday</th>
      <th>count</th>
      <th>busy_casual</th>
      <th>busy_reg</th>
      <th>weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>10.912309</td>
      <td>1.386294</td>
      <td>2.639057</td>
      <td>0</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2.833213</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>10.912309</td>
      <td>2.197225</td>
      <td>3.496508</td>
      <td>1</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3.713572</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>10.912309</td>
      <td>1.791759</td>
      <td>3.332205</td>
      <td>2</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3.496508</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>11.155078</td>
      <td>1.386294</td>
      <td>2.397895</td>
      <td>3</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2.639057</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>11.155078</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>4</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<h2>Testing Data Sample</h2>


```python
Test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>hour</th>
      <th>year</th>
      <th>month</th>
      <th>mday</th>
      <th>wday</th>
      <th>busy_casual</th>
      <th>busy_reg</th>
      <th>weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>0</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>1</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>3</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>4</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<h2> Predictive Modelling </h2>


```python
ftr_cas = ['season', 'holiday', 'workingday', 'weather','temp', 'atemp', 'humidity',
       'windspeed', 'hour', 'year', 'wday']#,'busy_casual']
ftr_reg = ['season', 'holiday', 'workingday', 'weather','temp', 'atemp', 'humidity',
       'windspeed', 'hour', 'year', 'wday']#,'busy_reg']

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor 
from sklearn.ensemble import GradientBoostingRegressor as GBM
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
np.set_printoptions(suppress=True)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
XGB = GBM(random_state=42)
krr = KernelRidge(kernel='polynomial', gamma=0.0001, alpha=3)
predict_val = []
predict_GB = []
predict_rf = []
predict_krr = []
RMSLE_gb = []
RMSLE_rf = []
RMSLE_krr = []
year = pd.unique(iter_test['year'])
month = pd.unique(iter_test['month'])
cas_err = []
reg_err = []
net_err = []
for i in year:
    for j in month:
        if i == 2011:
            end = int(start_end[1][j-1])
            Data = Train.iloc[0:end,:]
            start, end = int(start_end_[0][j-1]), int(start_end_[1][j-1])
        else:
            end = int(start_end[1][11+j])
            Data = Train.iloc[0:end,:]
            start, end = int(start_end_[0][11+j]), int(start_end_[1][11+j])
        
        X_T = Test.iloc[start:end, :]
        ###### Gradient Boosting ######
        XGB.set_params(n_estimators= 280, min_samples_leaf=6, learning_rate=0.15, max_depth=3)
        XGB.fit(Data[ftr_cas], Data['casual'])
        cas, val_cas_XGB = np.array(np.expm1(XGB.predict(X_T[ftr_cas]))), np.array(np.expm1(XGB.predict(Data[ftr_cas])))
        XGB.set_params(n_estimators= 280, min_samples_leaf=6, learning_rate=0.15, max_depth=3)
        XGB.fit(Data[ftr_reg], Data['registered'])
        reg, val_reg_XGB = np.array(np.exp(XGB.predict(X_T[ftr_reg]))-1), np.array(np.expm1(XGB.predict(Data[ftr_reg])))
        cas, reg = np.where(cas < 0, 0, cas), np.where(reg < 0, 0, reg)
        count_XGB, val_XGB = cas.reshape(-1,1) + reg.reshape(-1,1), val_cas_XGB.reshape(-1,1)+val_reg_XGB.reshape(-1,1)
        print('RMSLE Gradient Boosting',rmsle(Data['count'].values, val_XGB)) 
        RMSLE_gb.append(rmsle(Data['count'].values, val_XGB))
        
        ###### Random Forest ######
        rf.set_params(n_estimators = 700, min_samples_leaf = 2, n_jobs=-1)
        rf.fit(Data[ftr_cas], Data['casual'])
        cas, val_cas_rf = np.array(np.expm1(rf.predict(X_T[ftr_cas]))), np.array(np.expm1(rf.predict(Data[ftr_cas])))
        rf.set_params(n_estimators = 500, min_samples_leaf = 2, n_jobs=-1)
        rf.fit(Data[ftr_reg], Data['registered'])
        reg, val_reg_rf = np.array(np.expm1(rf.predict(X_T[ftr_reg]))), np.array(np.expm1(rf.predict(Data[ftr_reg])))
        cas, reg = np.where(cas < 0, 0, cas), np.where(reg < 0, 0, reg)
        count_rf, val_rf = cas.reshape(-1,1) + reg.reshape(-1,1), val_cas_rf.reshape(-1,1) + val_reg_rf.reshape(-1,1)
        print('RMSLE Random Forest',rmsle(Data['count'].values, val_rf)) 
        RMSLE_rf.append(rmsle(Data['count'].values, val_rf))
        
        ###### Kernel Ridge #######
        krr.fit(Data[ftr_cas], Data['casual'])
        cas, val_cas_krr = np.array(np.expm1(krr.predict(X_T[ftr_cas]))), np.array(np.expm1(krr.predict(Data[ftr_cas])))
        krr.fit(Data[ftr_reg], Data['registered'])
        reg, val_reg_krr = np.array(np.expm1(krr.predict(X_T[ftr_reg]))), np.array(np.expm1(krr.predict(Data[ftr_reg])))
        cas, reg = np.where(cas < 0, 0, cas), np.where(reg < 0, 0, reg)
        count_krr, val_krr = cas.reshape(-1,1) + reg.reshape(-1,1), val_cas_krr.reshape(-1,1) + val_reg_krr.reshape(-1,1)
        print('RMSLE Kernel Ridget',rmsle(Data['count'].values, val_krr)) 
        RMSLE_krr.append(rmsle(Data['count'].values, val_krr))
        
        count = 0.8*count_XGB + 0.2*count_rf
        cas = np.sqrt(np.mean((0.8*val_cas_XGB + 0.2*val_cas_rf -Data['casual'])))
        reg = np.sqrt(np.mean((0.8*val_reg_XGB + 0.2*val_reg_rf-Data['registered'])))
        net = np.sqrt(np.mean((0.8*val_XGB + 0.2*val_rf - Data['count'].values.reshape(-1,1))**2))
        cas_err.append(cas)
        reg_err.append(reg)
        net_err.append(net)
        predict_val.append(pd.DataFrame(count))
        predict_GB.append(pd.DataFrame(count_XGB))
        predict_rf.append(pd.DataFrame(count_rf))
        predict_krr.append(pd.DataFrame(count_krr))
        #print('Year: %d Month: %2d ' %(i, j, cas, reg, net))
pred = pd.concat([pd.DataFrame(i) for i in predict_val], axis=0)
pred.reset_index(inplace=True)
pred.drop('index',axis=1,inplace=True)
pred_gb = pd.concat([pd.DataFrame(i) for i in predict_GB], axis=0)
pred_gb.reset_index(inplace=True)
pred_gb.drop('index',axis=1,inplace=True)
pred_rf = pd.concat([pd.DataFrame(i) for i in predict_rf], axis=0)
pred_rf.reset_index(inplace=True)
pred_rf.drop('index',axis=1,inplace=True)
pred_krr = pd.concat([pd.DataFrame(i) for i in predict_krr], axis=0)
pred_krr.reset_index(inplace=True)
pred_krr.drop('index',axis=1,inplace=True)
```

    RMSLE Gradient Boosting 2.2076632468269564
    RMSLE Random Forest 2.1948443075776627
    RMSLE Kernel Ridget 2.1207427490477584
    RMSLE Gradient Boosting 2.316265844953561
    RMSLE Random Forest 2.315087634332903
    RMSLE Kernel Ridget 2.239600672588815
    RMSLE Gradient Boosting 2.382909656888275
    RMSLE Random Forest 2.3833781551718385
    RMSLE Kernel Ridget 2.304974440583867
    RMSLE Gradient Boosting 2.4636999048711603
    RMSLE Random Forest 2.4661915011790962
    RMSLE Kernel Ridget 2.389149568990624
    RMSLE Gradient Boosting 2.5979277215876495
    RMSLE Random Forest 2.6029805815554274
    RMSLE Kernel Ridget 2.528283454794111
    RMSLE Gradient Boosting 2.707619227207463
    RMSLE Random Forest 2.713073623164114
    RMSLE Kernel Ridget 2.639841398186593
    RMSLE Gradient Boosting 2.7887666943327645
    RMSLE Random Forest 2.793771234934828
    RMSLE Kernel Ridget 2.721496409506703
    RMSLE Gradient Boosting 2.833143001463715
    RMSLE Random Forest 2.839093968407748
    RMSLE Kernel Ridget 2.7667451295529824
    RMSLE Gradient Boosting 2.8587302718678513
    RMSLE Random Forest 2.8658190755978645
    RMSLE Kernel Ridget 2.7933499755677893
    RMSLE Gradient Boosting 2.88016160459766
    RMSLE Random Forest 2.887025288169728
    RMSLE Kernel Ridget 2.8160951489005064
    RMSLE Gradient Boosting 2.8908593702710017
    RMSLE Random Forest 2.895889734668419
    RMSLE Kernel Ridget 2.8260191753974713
    RMSLE Gradient Boosting 2.8888578551147126
    RMSLE Random Forest 2.8938043502971884
    RMSLE Kernel Ridget 2.823551325892644
    RMSLE Gradient Boosting 2.8795756063201305
    RMSLE Random Forest 2.8852172276758634
    RMSLE Kernel Ridget 2.814920596526644
    RMSLE Gradient Boosting 2.881882099825373
    RMSLE Random Forest 2.88736384639373
    RMSLE Kernel Ridget 2.8157495862134474
    RMSLE Gradient Boosting 2.902977333624984
    RMSLE Random Forest 2.907565831903719
    RMSLE Kernel Ridget 2.835646024252792
    RMSLE Gradient Boosting 2.932784726352616
    RMSLE Random Forest 2.938248507056204
    RMSLE Kernel Ridget 2.864904489858951
    RMSLE Gradient Boosting 2.961144759424067
    RMSLE Random Forest 2.9677312270205887
    RMSLE Kernel Ridget 2.894201906092869
    RMSLE Gradient Boosting 2.9924887780008964
    RMSLE Random Forest 2.9984021739361784
    RMSLE Kernel Ridget 2.925249771791143
    RMSLE Gradient Boosting 3.0159573722260937
    RMSLE Random Forest 3.0231941594576663
    RMSLE Kernel Ridget 2.9496438915726846
    RMSLE Gradient Boosting 3.041583760322448
    RMSLE Random Forest 3.048258961662618
    RMSLE Kernel Ridget 2.974815757882411
    RMSLE Gradient Boosting 3.0638730019574436
    RMSLE Random Forest 3.070252437543014
    RMSLE Kernel Ridget 2.9971826442874483
    RMSLE Gradient Boosting 3.0822586068248845
    RMSLE Random Forest 3.0880655936165704
    RMSLE Kernel Ridget 3.0153123779241007
    RMSLE Gradient Boosting 3.0914851326019535
    RMSLE Random Forest 3.0979549575589127
    RMSLE Kernel Ridget 3.025448706624857
    RMSLE Gradient Boosting 3.0975968810116172
    RMSLE Random Forest 3.1043439818065566
    RMSLE Kernel Ridget 3.031916893238681
    


```python
sns.set()
fig, ax = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(20, 20)
sns.distplot(pred, color='red', fit=norm, ax=ax[0][0])
ax[0][0].set(xlabel='Ensemble of Gradient Boosting and Random Forest')
sns.distplot(pred_gb, color='red', fit=norm, ax=ax[0][1])
ax[0][1].set(xlabel='Gradient Boosting')
sns.distplot(pred_rf, color='red', fit=norm, ax=ax[1][0])
ax[1][0].set(xlabel='Random Forest')
sns.distplot(pred_krr, color='red', fit=norm, ax=ax[1][1])
ax[1][1].set(xlabel='Kernel Ridge Regression')

file = pd.concat([Test['datetime'], 1.06*pred], axis=1)
file.columns = ['datetime','count']
file.to_csv('Ensemble.csv', index=False)
file = pd.concat([Test['datetime'], 1.06*pred_gb], axis=1)
file.columns = ['datetime','count']
file.to_csv('GB.csv', index=False)
file = pd.concat([Test['datetime'], 1.06*pred_rf], axis=1)
file.columns = ['datetime','count']
file.to_csv('RF.csv', index=False)
file = pd.concat([Test['datetime'], 1.06*pred_krr], axis=1)
file.columns = ['datetime','count']
file.to_csv('KRR.csv', index=False)
```


![png](output_19_0.png)



```python
x = np.linspace(1,24,24)
fig, ax = plt.subplots()
fig.set_size_inches(10,7)
sns.pointplot(x,RMSLE_gb, color='red')
sns.pointplot(x,RMSLE_rf, color='green')
sns.pointplot(x,RMSLE_krr, color='blue')
ax.set(ylabel='rMSLE Error', xlabel='Gradient Boosting VS Random Forest vs Kernel Ridge')
```




    [Text(0, 0.5, 'rMSLE Error'),
     Text(0.5, 0, 'Gradient Boosting VS Random Forest vs Kernel Ridge')]




![png](output_20_1.png)



```python

```
