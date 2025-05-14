# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages. 
2. Analyse the data
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R.PRIYANGA
RegisterNumber: 212223230161 
*/
```



```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape
```

```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
DATA:
![image](https://github.com/user-attachments/assets/ae31ef4e-76fc-4058-9474-22749302b668)

DATA.SHAPE()
![image](https://github.com/user-attachments/assets/da54a790-9f1b-42e7-81ad-b11c2fda48d9)

X.SHAPE()
![image](https://github.com/user-attachments/assets/6a1aa6bb-b63e-4e88-8d22-382f9bb710fb)

Y.SHAPE()
![image](https://github.com/user-attachments/assets/108365d3-c4a4-4eeb-b22c-c9716cbc66e6)

X_TRAIN
![image](https://github.com/user-attachments/assets/f8230363-6716-472d-9eb4-8cb9a5be9807)

x_train.shape()
![image](https://github.com/user-attachments/assets/2209acb0-4d7c-44a4-997d-9a4cf1342adc)

y_pred
![image](https://github.com/user-attachments/assets/600609a8-0486-474c-a0f3-15504198ab6d)

acc (accuracy)

![image](https://github.com/user-attachments/assets/027d3913-e112-4ea9-a816-a2f6e6f0f867)

con (confusion matrix)
![image](https://github.com/user-attachments/assets/2139616c-df76-4fac-ac1b-393e3ffd6d1e)

cl (classification report)
![image](https://github.com/user-attachments/assets/47d4fc31-335d-46eb-9950-891bc8bc4299)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
