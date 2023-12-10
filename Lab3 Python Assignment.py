#!/usr/bin/env python
# coding: utf-8

# ## Learning Outcomes
# - Exploratory data analysis & preparing the data for model building. 
# - Machine Learning - Supervised Learning Classification
#   - Logistic Regression
#   - Naive bayes Classifier
#   - KNN Classifier
#   - Decision Tree Classifier
#   - Random Forest Classifier
#   - Ensemble methods
# - Training and making predictions using different classification models.
# - Model evaluation

# ## Objective: 
# - The Classification goal is to predict “heart disease” in a person with regards to different factors given. 
# 
# ## Context:
# - Heart disease is one of the leading causes of death for people of most races in the US. At least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking. 
# - Detecting and preventing the factors that have the greatest impact on heart disease is very important in healthcare. Machine learning methods may detect "patterns" from the data and can predict whether a patient is suffering from any heart disease or not..
# 
# ## Dataset Information
# 
# #### Source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease?datasetId=1936563&sortBy=voteCount
# Originally, the dataset come from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents. 
# 
# This dataset consists of eighteen columns
# - HeartDisease: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
# - BMI: Body Mass Index (BMI)
# - Smoking: smoked at least 100 cigarettes in your entire life
# - AlcoholDrinking: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
# - Stroke:Ever had a stroke?
# - PhysicalHealth: physical health, which includes physical illness and injury
# - MentalHealth: for how many days during the past 30 days was your mental health not good?
# - DiffWalking: Do you have serious difficulty walking or climbing stairs?
# - Sex: male or female?
# - AgeCategory: Fourteen-level age category
# - Race: Imputed race/ethnicity value
# - Diabetic: diabetes?
# - PhysicalActivity: Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
# - GenHealth: Would you say that in general your health is good, fine or excellent?
# - SleepTime: On average, how many hours of sleep do you get in a 24-hour period?
# - Asthma: you had asthma?
# - KidneyDisease: Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
# - SkinCancer: Ever had skin cancer?

# ### 1. Importing Libraries

# In[16]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# ### 2. Load the dataset and display a sample of five rows of the data frame.

# In[17]:


df = pd.read_csv('heart_2020_cleaned.csv')
df.head()


# ### 3. Check the shape of the data (number of rows and columns). Check the general information about the dataframe using the .info() method.

# In[18]:


df.shape


# In[19]:


df.info()


# ### 4. Check the statistical summary of the dataset and write your inferences.

# In[20]:


df.describe().T


# In[21]:


df.describe(include='O')


# The minimum value of the BMI is around 12 and maximum is 94.85
# The mental health indicates that for how many days during the past 30 days was your mental health not good, so that minimum value 0 means the person's mental health was good throughout the month whereas on an average it is 7 days that mental health was not good.
# HeartDisease, Smoking, Alcohol Drinking, Stroke, DiffWalking, Sex, PhysicalActivity, Asthma, KidneyDisease, and SkinCancer columns contain the binary categories 'Yes' or 'NO'.
# There are 6 different race category.

# ### 5. Check the percentage of missing values in each column of the data frame. Drop the missing values if there are any.

# In[22]:


df.isnull().sum()/len(df)*100


# There are no missing values

# ### 6. Check if there are any duplicate rows. If any drop them and check the shape of the dataframe after dropping duplicates.

# In[23]:


len(df[df.duplicated()])


# In[24]:


df.drop_duplicates(inplace=True)


# In[25]:


df.shape


# ### 7. Check the distribution of the target variable (i.e. 'HeartDisease') and write your observations.

# In[26]:


df['HeartDisease'].value_counts().plot(kind='pie',autopct='%1.0f%%')
plt.show()


# We can observe that the target class distribution is highly imbalanced.

# ### 8. Visualize the distribution of the target column 'Heart disease' with respect to various categorical features and write your observations.

# In[29]:


### Categorical features in the dataset
categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns


# Let's look at the distribution of the number of people with heart disease from various factors

# In[30]:


i = 1
plt.figure(figsize = (30,25))
for feature in categorical_features:
    plt.subplot(6,3,i)
    sns.countplot(x = feature,hue = 'HeartDisease' , data = df)
    i +=1


# In[ ]:





# ### 9. Check the unique categories in the column 'Diabetic'. Replace 'Yes (during pregnancy)' as 'Yes' and 'No, borderline diabetes' as 'No'.

# In[31]:


df['Diabetic'].unique()


# In[32]:


df['Diabetic'] = df['Diabetic'].replace({'Yes (during pregnancy)':'Yes','No, borderline diabetes':'No'})


# In[33]:


df['Diabetic'].value_counts()


# ### 10. For the target column 'HeartDiease', Replace 'No' as 0 and 'Yes' as 1. 

# In[34]:


df['HeartDisease'] = df['HeartDisease'].replace({'Yes':1,'No':0})


# In[35]:


df['HeartDisease'].value_counts()


# ### 11. Label Encode the columns "AgeCategory", "Race", and "GenHealth". Encode the rest of the columns using dummy encoding approach.

# In[36]:


object_type_variables = [i for i in df[['AgeCategory','Race','GenHealth']] if df.dtypes[i] == object]
object_type_variables 


le = LabelEncoder()

def encoder(df):
    for i in object_type_variables:
        q = le.fit_transform(df[i].astype(str))  
        df[i] = q                               
        df[i] = df[i].astype(int)
encoder(df)


# In[37]:


df = pd.get_dummies(df,drop_first=True)


# In[38]:


## let check few samples after encoding.
df.head(2)


# ### 12. Store the target column (i.e.'HeartDisease') in the y variable and the rest of the columns in the X variable.

# In[39]:


X = df.drop('HeartDisease',axis=1)
y = df['HeartDisease']


# ### 13. Split the dataset into two parts (i.e. 70% train and 30% test) and print the shape of the train and test data

# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)


# ### 14. Standardize the numerical columns using Standard Scalar approach for both train and test data.

# In[41]:


ss = StandardScaler()

X_train.iloc[:,:7] = ss.fit_transform(X_train.iloc[:,:7])
X_test.iloc[:,:7] = ss.transform(X_test.iloc[:,:7])


# In[42]:


## Lets check few scaled features
X_train.head(2)


# In[43]:


X_test.head(2)


# ### 15. Write a function.
# - i) Which can take the model and data as inputs.
# - ii) Fits the model with the train data.
# - iii) Makes predictions on the test set.
# - iv) Returns the Accuracy Score.

# In[44]:


def fit_n_print(model, X_train, X_test, y_train, y_test):  # take the model, and data as inputs

    model.fit(X_train, y_train)   # fits the model with the train data

    pred = model.predict(X_test)  # makes predictions on the test set

    accuracy = accuracy_score(y_test, pred)
                   
    return accuracy  # return all the metrics


# ### 16. Use the function and train a Logistic regression, KNN, Naive Bayes, Decision tree, Random Forest, Adaboost, GradientBoost, and Stacked Classifier models and make predictions on test data and evaluate the models, compare and write your conclusions and steps to be taken in future in order to improve the accuracy of the model.

# In[45]:


lr = LogisticRegression()
nb = GaussianNB()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
adb = AdaBoostClassifier()
gb = GradientBoostingClassifier()

estimators = [('rf', rf),('knn', knn), ('gb', gb), ('adb', adb)]
sc = StackingClassifier(estimators=estimators, final_estimator=rf)


# In[46]:


result = pd.DataFrame(columns = ['Accuracy'])

for model, model_name in zip([lr, nb, knn, dt, rf, adb, gb, sc], 
                             ['Logistic Regression','Naive Bayes','KNN','Decision tree', 
                              'Random Forest', 'Ada Boost', 'Gradient Boost', 'Stacking']):
    
    result.loc[model_name] = fit_n_print(model, X_train, X_test, y_train, y_test)


# In[47]:


result


# ### Conclusion

# In[ ]:





# ----
# ## Happy Learning:)
# ----
