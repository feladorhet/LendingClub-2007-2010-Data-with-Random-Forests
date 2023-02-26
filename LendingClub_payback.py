import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

loans = pd.read_csv("loan_data.csv")
#Index(['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc','dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util','inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'])

##EXPLORATORY DATA ANALYSIS 
##STARTS HERE

##two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize=(10,6))
loans[loans["credit.policy"]==0]["fico"].hist(alpha=0.5, color="red", label="credit.policy=0")
loans[loans["credit.policy"]==1]["fico"].hist(alpha=0.5, color="blue", label="credit.policy=1")
plt.xlabel("FICO")
plt.legend(loc=0)
#plt.show()

##counts of loans by purpose with the color hue defined by not.fully.paid.
byPurpose = sns.countplot(data=loans, x="purpose", hue="not.fully.paid", palette="rocket")
#plt.show()

##the trend between FICO score and interest rate
jointFicoInterest = sns.jointplot(data=loans, x="fico", y="int.rate", color="purple")
#plt.show()

plt.figure(figsize=(12,8))
sns.lmplot(data=loans, x="fico", y="int.rate", hue="credit.policy", col="not.fully.paid", palette="Set1")
#plt.show()

##EXPLORATORY DATA ANALYSIS 
##ENDS HERE

#"purpose" is a categorıcal column so match them to be directly digestible data
final_data = pd.get_dummies(data=loans, columns=["purpose"], drop_first=True)

#final_data.info() İS AS FOLLOWS:
##   Column                      Non-Null Count  Dtype  
#--  ------                      --------------  -----  
#0   credit.policy               9578 non-null   int64  
#1   int.rate                    9578 non-null   float64
#2   installment                 9578 non-null   float64
#3   log.annual.inc              9578 non-null   float64
#4   dti                         9578 non-null   float64
#5   fico                        9578 non-null   int64  
#6   days.with.cr.line           9578 non-null   float64
#7   revol.bal                   9578 non-null   int64  
#8   revol.util                  9578 non-null   float64
#9   inq.last.6mths              9578 non-null   int64  
#10  delinq.2yrs                 9578 non-null   int64  
#11  pub.rec                     9578 non-null   int64  
#12  not.fully.paid              9578 non-null   int64  
#13  purpose_credit_card         9578 non-null   uint8  
#14  purpose_debt_consolidation  9578 non-null   uint8  
#15  purpose_educational         9578 non-null   uint8  
#16  purpose_home_improvement    9578 non-null   uint8  
#17  purpose_major_purchase      9578 non-null   uint8  
#18  purpose_small_business      9578 non-null   uint8  


##TRAIN A SINGLE DECISION TREE
X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=30)
decTree = DecisionTreeClassifier()
decTree.fit(X=X_train, y=y_train)

decTreePredicts = decTree.predict(X_test)
decTree_confMat = confusion_matrix(y_true = y_test, y_pred = decTreePredicts)
decTree_classReps = classification_report(y_true = y_test, y_pred = decTreePredicts)

print(decTree_confMat)
print(decTree_classReps) #0.74 f1-score accuracy

##TRAIN A RANDOM FOREST FOR BETTER RESULTS
decRF = RandomForestClassifier(n_estimators = 1000)
decRF.fit(X=X_train, y=y_train)

decRF_preds = decRF.predict(X_test)
decRF_confMat = confusion_matrix(y_true = y_test, y_pred = decRF_preds)
decRF_classReps = classification_report(y_true = y_test, y_pred = decRF_preds)

print(decRF_confMat)
print(decRF_classReps) #0.84 f1-score accuracy 

##NEITHER MODELS PERFORMED TOO WELL, NEEDS FURTHER FEATURE ENGINEERING

##TO BE CONTINUED