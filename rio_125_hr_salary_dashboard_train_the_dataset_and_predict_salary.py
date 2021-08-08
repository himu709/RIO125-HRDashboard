#Author :Himesh
# RIO-125: HR Salary Dashboard - Train the Dataset and Predict Salary

# Author : Himesh#

#Importing required libraires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the data
data = pd.read_csv('HR_comma_sep.csv' )

#Analyzing the data

data.columns

data.shape

data.head()

data.dtypes

data.describe()

data.info()

"""# Data Preprocessing"""

#Checking for null values

data.isna().sum()

#We can see that there are no null values in the dataset.

#Finding outliers

#Finding outliers for ('satisfaction_level','last_evaluation', 'number_project')

selected_col= ['satisfaction_level','last_evaluation', 'number_project']

data.boxplot(selected_col)
plt.title('Boxplot of satisfaction_level, last_evaluation, number_project')

# We can see that there are no outliers for these three columns

selected_col= ['average_montly_hours','time_spend_company', 'Work_accident']

data.boxplot(selected_col)
plt.title('Boxplot of average_montly_hours, time_spend_company, Work_accident')

#We can see that there are outliers for time_spend_company and work_accident.
#We will have to take care of these.

selected_col= ['left','promotion_last_5years']

data.boxplot(selected_col)
plt.title('Boxplot of left, promotion_last_5years')

#We can see that ther are no outliers for these two columns.

#Removing outliers

data.shape

selected_col= ['time_spend_company', 'Work_accident']

for y in selected_col:
    #Calculating Q1, Q2, Q3 values
    Q1=np.percentile(data[y],25,interpolation='midpoint')
    Q2=np.percentile(data[y],50,interpolation='midpoint')
    Q3=np.percentile(data[y],75,interpolation='midpoint')
    print(Q1)
    print(Q2)
    print(Q3)
    limit=1.5
    print(limit)
    #Calculating IQR, lower, upper values
    IQR=Q3-Q1
    low_lim=(Q1-(limit*IQR))
    Up_lim=(Q3+(limit*IQR))
    print("Outlier Lower Limit : ",low_lim)
    print("Outlier Upper Limit: ",Up_lim)
    #Determining the outliers
    outlier=[]
    for x in data[y]:
        if((x>Up_lim) or (x<low_lim)):
            outlier.append(x)
    print(len(outlier))
    print("Outlier : ",outlier)
    #Finding Index of outliers
    index1=data[y]>Up_lim
    ind=[]
    ind=data.loc[index1].index
    print(ind)
    #Removing outliers
    for i in ind:
        data.drop(i,inplace=True)

data.shape

#So now the data is clean with no null values and no null values.

#Checking if the data is balanced,by checking the dependent variable which is salary

print(data["salary"].value_counts())

#Feature Engineering

#To segregate employees according to their performance

avg_evaluation = data['last_evaluation'].mean()
std_evaluation = data['last_evaluation'].std()
data['std_performance'] = (data['last_evaluation'] - avg_evaluation) / std_evaluation
data['performance_differential'] = data['last_evaluation']-(avg_evaluation+std_evaluation)

def performance_classification(row):
    if row['performance_differential']>=0:
        performance_class = 'Top Performer'
    else:
        performance_class = 'Lower Performer'
    return (performance_class)
data['classification'] = data.apply(performance_classification, axis=1)

data['classification']

#Calculating average daily hours.Considering average working days is 22.

data['daily_hours'] = data['average_montly_hours']/22

# Salary is an important indicator but since it is not numeric we will have to map the three classes of salaries
# to arbitrary numbers as shown below

salary_num_dict = {'low':30000, 'medium':60000, 'high':90000}
data['salary_num'] = (data['salary'].map(salary_num_dict))



left_dict = {1: 'left', 0: 'stayed'}

#data['left(as_string)'] = (data['left'].map(left_dict))
new_data = data

#new_data['performance_group'] = new_data['left(as_string)'] + ':' + new_data['classification']




#Model Creation

#Since our problem is a classification,we have will have to develop a classification model.

new_data.head()

"""#As we can see that there are catogorical features,we will either have to label encoding or one hot encoding."""

data_final = new_data
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in data.columns:
    if column != 'salary':
        data_final[column] = labelencoder.fit_transform(data_final[column])

data_final.head()

#Feature Selection

plt.rcParams['figure.figsize'] = (14,12)
sns.heatmap(data_final.corr(), vmin=-1, vmax=1, center=0,
            square=True, cmap = sns.diverging_palette(240, 10, n=9))
plt.show()

cor = data.corr()

#Correlation with output variable
cor_target = abs(cor["salary_num"])

#Selecting highly correlated features,setting a cutoff value for corelation
relevant_features = cor_target[cor_target>0.02]
relevant_features

#We need to split data to X and Y.

#Since salary and salry_num are highly corelated,we can drop salary_num.
data_final= data_final.drop(['salary_num'], axis = 1)

y = new_data['salary'].values
X = data_final.drop(['salary'], axis=1).values

#Splitting data to train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape

#To display accuracy
def dispAcc(y_pred):
  print("Classification Report")
  print(classification_report(y_test,y_pred))
  print("Accuracy is ",accuracy_score(y_test,y_pred))
  print("Precision is ", precision_score(y_test,y_pred,pos_label='positive',
                                           average='micro'))
  print("Recall is ",recall_score(y_test,y_pred,y_pred,pos_label='positive',
                                           average='micro'))
  print("f1 score is ",f1_score(y_test,y_pred,y_pred,pos_label='positive',
                                           average='micro'))
  confusion_matrix(y_test,y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report,roc_curve
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
print('Random Forest: ')
rf.fit(X_train,y_train)
y_pred= rf.predict(X_test)
dispAcc(y_pred)
pickle.dump(rf,open('model.pkl','wb'))
