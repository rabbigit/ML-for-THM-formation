#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
Data=pd.read_csv("Train_dataset_all.csv")
X,y=Data.iloc[:,0:5],Data.iloc[:,5]
RFr=RandomForestRegressor()
score= cross_val_score(RFr,X,y,cv=10,scoring='neg_root_mean_squared_error').mean()
CV_prediction=cross_val_predict(RFr,X,y,cv=10)
RFr.fit(X,y)
plt.scatter(y,CV_prediction)
expvspred_5cv = {'Exp': y, 'Pred':CV_prediction}
pd.DataFrame(expvspred_5cv).to_csv('RFr_10fcv_predictions.csv')
explainer=shap.TreeExplainer(RFr,X)
shap_values=explainer.shap_values(X)
shap.summary_plot(shap_values,X)
shap_value=explainer(X)
shap.plots.bar(shap_value)


# In[ ]:




