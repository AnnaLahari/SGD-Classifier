#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
iris = load_iris()


# In[2]:


df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']= iris.target


# In[3]:


print(df.head())


# In[4]:


X=df.drop('target',axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)


# In[5]:


accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm= confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print('NAME: A.LAHARI')
print('REG.No : 212223230111')

