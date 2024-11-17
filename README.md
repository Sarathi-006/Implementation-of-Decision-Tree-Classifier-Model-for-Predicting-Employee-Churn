# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 Load the Data: Read the employee dataset from a CSV file and check its structure and missing values.

2 Preprocess the Data: Encode the categorical salary column using LabelEncoder().

3 Feature Selection: Select the relevant features (x) and the target variable (y, which is left).

4 Split the Data: Split the data into training and test sets (80%-20%) using train_test_split().

5 Train the Model: Train a Decision Tree Classifier (DecisionTreeClassifier with entropy criterion) on the training data.

6 Evaluate the Model: Predict on the test set, calculate the accuracy, and make a prediction for a new data point.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.PARTHASARATHI
RegisterNumber:  212223040144
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![369268104-9d88a883-db85-4036-b3f8-3f27ce8dab33](https://github.com/user-attachments/assets/44f8f3c0-df15-4450-8124-b220fb8c0853)
ACCURACY
![369268140-cf464349-aa04-443d-a466-26e214d95951](https://github.com/user-attachments/assets/b794c176-1c04-4986-9aa1-0335f5639af1)
NEW PREDICTED
![369268175-ce36cfee-0ddf-4d82-be97-ad7028eccfb5](https://github.com/user-attachments/assets/9436a6e1-9f5d-4bee-9fbb-24858d0f02d2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
