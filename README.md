# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GEETHU R
RegisterNumber:  212224040089
*/
```
~~~
import pandas as pd
data = pd.read_csv(r"C:\Users\admin\Downloads\Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~
## Output:
### Top 5 ELEMENTS
<img width="592" height="106" alt="1" src="https://github.com/user-attachments/assets/c7b59108-2ebb-475d-b0be-b07d12410690" />

<img width="605" height="123" alt="2" src="https://github.com/user-attachments/assets/117730a0-0db8-49a9-a974-f6ca2bcff87a" />


### DATA DUPLICATE
<img width="168" height="25" alt="4" src="https://github.com/user-attachments/assets/cfd7a550-62d2-4ec8-a1ea-2fb4c53b3205" />

### PRINT DATA

<img width="977" height="440" alt="5 I" src="https://github.com/user-attachments/assets/6e19ca4b-0622-46ad-ba63-36998a1fae08" />

<img width="1010" height="437" alt="5 II" src="https://github.com/user-attachments/assets/88e12795-4c64-4bec-b7cb-8a130583f81f" />

### DATA STATUS
<img width="192" height="237" alt="3" src="https://github.com/user-attachments/assets/74da882c-755a-4b9d-a989-b528b2a2cc78" />

### Y_PREDICTION ARRAY:
<img width="857" height="62" alt="Y pred" src="https://github.com/user-attachments/assets/017fcdbd-5dc4-4f24-b780-e6936d50ebc2" />

### ACCURACY VALUE
<img width="242" height="37" alt="8" src="https://github.com/user-attachments/assets/ef7902dc-a4b1-441b-a7be-552dafdc9af8" />

### CONFUSION MATRIX

<img width="172" height="52" alt="CONFUSION " src="https://github.com/user-attachments/assets/729f5093-5463-4791-95e3-d273c302b992" />


### CLASSIFICATION REPORT

<img width="517" height="172" alt="10" src="https://github.com/user-attachments/assets/b58de8b6-11ed-4a28-864c-76dc65ebbc31" />

### PREDICTION OF LR
<img width="1027" height="43" alt="L" src="https://github.com/user-attachments/assets/621e30ce-ed86-42bc-8532-797a44762b8c" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
