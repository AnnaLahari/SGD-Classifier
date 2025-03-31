# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import Necessary Libraries and Load Data 
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy 
5.Generate Confusion Matrix


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: A.LAHARI
RegisterNumber:  212223230111
*/
```

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())

```

```
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confufion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)


```
## Output:

![Screenshot 2025-03-29 190959](https://github.com/user-attachments/assets/b863dec5-3cf0-4e21-aa00-68a15d5f8556)

![Screenshot 2025-03-29 191006](https://github.com/user-attachments/assets/1097d428-7f41-4fc4-8a5c-d5961e3e11da)

![Screenshot 2025-03-29 191012](https://github.com/user-attachments/assets/fef2e7ae-fac2-43bc-a4bd-c2cd52bdb918)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
