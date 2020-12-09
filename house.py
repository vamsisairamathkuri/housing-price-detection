import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
boston=load_boston()
house_data=pd.DataFrame(data=boston.data,columns=boston.feature_names)
house_data['price']=boston.target
print(house_data.isnull().sum())
print(house_data.describe())
corr=house_data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,cbar=True,annot=True,cmap='Greens')
plt.title('data correlation')
plt.show()
X=house_data.drop(['price'],axis=1)
Y=house_data['price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)
#LINEAR REGRESSION
print('LINEAR REGRESSION')
lm=LinearRegression()
lm.fit(X_train,Y_train)
coefficients=pd.DataFrame([X_train.columns,lm.coef_]).T
coefficients=coefficients.rename(columns={0:'Attribute',1:'Coeeficients'})
print(coefficients)
trainpred=lm.predict(X_train)
plt.scatter(Y_train,trainpred)
plt.show()
plt.scatter(Y_train,(Y_train-trainpred))
plt.show()
testpredict=lm.predict(X_test)
plt.scatter(Y_test,testpredict)
plt.show()
#model evaluation
print('MODEL EVALUATION')
acc_linreg=metrics.r2_score(Y_test,testpredict)
print('r2 score:',acc_linreg)
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,testpredict))
print('Mean Squared Error:',metrics.mean_squared_error(Y_test,testpredict))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_test,testpredict)))

#RANDOM FOREST REGRESSOR
print('RANDOM FOREST REGRESSOR')
ranreg=RandomForestRegressor()
ranreg.fit(X_train,Y_train)
y_pred=ranreg.predict(X_train)
plt.scatter(Y_train,y_pred)
plt.show()
testpredict=ranreg.predict(X_test)
sns.distplot(Y_test-testpredict)
plt.show()
print('Model Evaluation')
acc_forest=metrics.r2_score(Y_test,testpredict)
print('r2 score:',acc_forest)
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,testpredict))
print('Mean Squared Error:',metrics.mean_squared_error(Y_test,testpredict))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_test,testpredict)))
