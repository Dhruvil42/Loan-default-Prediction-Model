#%% Importing Libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None) #to display all columns
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

#%% Importing data
ds= pd.read_csv(r'C:\Users\dhruvil rathod\Downloads\Python Project/XYZCorp_LendingData.txt',delimiter="\t")
ds.shape
ds.head(5)

#%% Creating a copy of data file
ds1 = ds.copy()

#%% Preprocessing
ds1.shape
ds1.columns
ds1.isnull().sum()

#%% Dropping unwanted Columns
ds1=ds1.drop(['id','member_id','sub_grade','emp_title','title','zip_code','addr_state','funded_amnt','funded_amnt_inv',
                          'last_pymnt_d','next_pymnt_d','last_credit_pull_d','pymnt_plan','policy_code',
                          'installment','total_pymnt','out_prncp_inv','total_rec_late_fee','recoveries',
                          'collection_recovery_fee','last_pymnt_amnt','int_rate','initial_list_status'],axis=1)

#%% Handling Missing Values
ds1.isnull().sum()
ds1 = ds1.loc[:,ds1.isnull().sum()/len(ds1) <.50 ]
ds1.columns

#%% Filling Missing Values 
for x in ds1.columns[:]:
    if ds1[x].dtype=='object':
        ds1[x].fillna(ds1[x].mode()[0],inplace=True)
    elif ds1[x].dtype=='int64' or ds1[x].dtype=='float64':
        ds1[x].fillna(ds1[x].mean(),inplace=True)
ds1.isnull().sum()
ds1.issue_d=pd.to_datetime(ds1.issue_d) #
col_name='issue_d'
print(ds1[col_name].dtype)
ds1.issue_d

ds1.dtypes
colname=[]
for x in ds1.columns:
    if ds1[x].dtype=="object":
        colname.append(x)
colname

#%% Label Encoding
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
    ds1[x]=le.fit_transform(ds1[x])
    
#%% Splitting Data based on Date's(issue_d)
ds1.issue_d=pd.to_datetime(ds1.issue_d,infer_datetime_format=True)
col_name='issue_d'
print(ds1[col_name].dtype)
split_data="2015-06-01"
Training=ds1[ds1['issue_d']<split_data]
Training.shape
Testing=ds1.loc[ds1['issue_d']>='2015-06-01',:]
Testing.shape

Training=Training.drop(['issue_d'],axis=1)
Testing=Testing.drop(['issue_d'],axis=1)

X = Training.values[:,:-1]
Y = Training.values[:,-1]

#%% Creating a Cross Validation Data
from sklearn.model_selection import train_test_split
X_train ,  cross_x_test ,Y_train , cross_y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

#%% Splitting data into training and test
X_train = Training.values[:,:-1]
Y_train = Training.values[:,-1]
X_test = Testing.values[:,:-1]
Y_test = Testing.values[:,-1]

#%% Standard Scalar
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
cross_x_test=scaler.transform(cross_x_test)
print(X_train)
print(cross_x_test)

#%% Converting into int
Y_train=Y_train.astype(int)
cross_y_test=cross_y_test.astype(int)

#%%
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(cross_x_test)
#print(list(zip(cross_y_test,Y_pred)))
y_pred_class=[]
for value in y_pred_prob[:,1]:
if value > 0.46:
y_pred_class.append(1)
else:
y_pred_class.append(0)

#%%
print(classifier.coef_)
print(classifier.intercept_)
#%% Model Evaluation 
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm = confusion_matrix(cross_y_test, Y_pred)
print(cfm)
print("Classification Report :")
print(classification_report(cross_y_test, Y_pred))
acc= accuracy_score(cross_y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%% 
y_pred_prob= classifier.predict_proba(cross_x_test)

#%% Adjusting The Threshold
 y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.41:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

#%% Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm = confusion_matrix(y_pred_class, Y_pred)
print(cfm)
print("Classification Report :")
print(classification_report(y_pred_class, Y_pred))
acc= accuracy_score(y_pred_class, Y_pred)
print("Accuracy of the model: ",acc)

#%% 
y_pred_prob= classifier.predict_proba(X_test)
print(y_pred_prob)

#%% Metrics
from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

#%% ROC Curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()
#%%
Y_pred=classifier.predict(X_test)

#%% DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
model_DT = DecisionTreeClassifier(random_state=10,min_samples_leaf=5,max_depth=5,criterion='gini')
#default criterion is gini
model_DT.fit(X_train,Y_train)
#%%

Y_pred = model_DT.predict(cross_x_test)
#print(Y_pred)
#print(list(zip(cross_y_test,Y_pred)))

#%% Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm = confusion_matrix(cross_y_test, Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(cross_y_test, Y_pred))

acc= accuracy_score(cross_y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%TUning the Model
y_pred_prob= classifier.predict_proba(cross_x_test)

#%% Adjusting the Threshold
 y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.41:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

#%% Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm = confusion_matrix(cross_y_test, Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(cross_y_test, Y_pred))

acc= accuracy_score(cross_y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%
from sklearn import tree

with open("model_DecisionTree.txt","w") as f:
    f = tree.export_graphviz(model_DT,feature_names=colname[:-1],out_file=f)
#%% 
from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)
#%%
#ROC curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()

#%% ENSEMBLE MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
#%%

estimators = []
model1 = LogisticRegression()
estimators.append(('log', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC(kernel='rbf',gamma=0.1,C=70.0)
estimators.append(('svm', model3))
print(estimators)
#%%
"ensemble = VotingClassifier(estimators)
    "ensemble.fit(X_train,Y_train)
"Y_pred=ensemble.predict(X_test)"
#%%