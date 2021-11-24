#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# #  Data Science and AI
# ## Lab 7.1.2: Random Forests
# 
# INSTRUCTIONS:
# 
# - Read the guides and hints then create the necessary analysis and code to find and answer and conclusion for the scenario below.
# - The baseline results (minimum) are:
#     - **Accuracy** = 0.7419
#     - **ROC AUC**  = 0.6150
# - Try to achieve better results!

# # Foreword
# It is common that companies and professionals start with the data immediately available. Although this approach works, ideally the first stp is to idenfy the problem or question and only then identify and obtain the set of data that can help to solve or answer the problem.
# 
# Also, given the current abundance of data, processing power and some particular machine learning methods, there could be a temptation to use ALL the data available. **Quality** is _**better**_ then **Quantity**!
# 
# Part of calling this discipline **Data Science** is that it is supposed to follow a process and not reach conclusions without support from evidence.
# 
# Moreover, it is a creative, exploratory, labour and iteractive processes. It is part of the process to repeat, review and change when finding a dead-end.

# # Step 1: Define the problem or question
# Identify the subject matter and the given or obvious questions that would be relevant in the field.
# 
# ## Potential Questions
# List the given or obvious questions.
# 
# ## Actual Question
# Choose the **one** question that should be answered.

# # Step 2: Find the Data
# ### Blood Transfusion Service Center DataSet
# - **Abstract**: Data taken from the **Blood Transfusion Service Center** in Hsin-Chu City in Taiwan.
# - Date Donated: 2008-10-03
# - Source:
#         Original Owner and Donor: Prof. I-Cheng Yeh 
#         Department of Information Management 
#         Chung-Hua University, 
#         Hsin Chu, Taiwan 30067, R.O.C. 
# 
# - Citation Request:
#     **NOTE**: Reuse of this database is unlimited with retention of copyright notice for Prof. I-Cheng Yeh and the following published paper: 
# 
#         Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence, "Expert Systems with Applications, 2008
#         
# ### UCI - Machine Learning Repository
# - Center for Machine Learning and Intelligent Systems
# 
# The [**UCI Machine Learning Repository**](http://archive.ics.uci.edu/ml/about.html) is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.

# In[6]:


# Find the dataset described above 

# Hint: search for it through the UCI Machine Learning Repository
## Import Libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import seaborn as sns
sns.set(style = 'ticks')


# # Step 3: Read the Data
# - Read the data
# - Perform some basic structural cleaning to facilitate the work

# In[4]:


# Import libraries
## reading and inspect the data

df = pd.read_csv('transfusion.data')
print(df.dtypes)
df.head()


# In[5]:


# Read data in pandas
# change the names of the columns and inspect again
df.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donated_Mar_2007']
print(df.dtypes)
df.head()
# Check data has loaded correctly


# # Step 4: Explore and Clean the Data
# - Perform some initial simple **EDA** (Exploratory Data Analysis)
# - Check for
#     - **Number of features**
#     - **Data types**
#     - **Domains, Intervals**
#     - **Outliers** (are they valid or expurious data [read or measure errors])
#     - **Null** (values not present or coded [as zero of empty strings])
#     - **Missing Values** (coded [as zero of empty strings] or values not present)
#     - **Coded content** (classes identified by numbers or codes to represent absence of data)

# In[7]:


# Perform EDA by investigating each of the points above 
# Number of features

print('- Number of features: %d' % df.shape[1])
for c in df.columns:
    print('  - %s' % c)

# Data types
print('\n- Data types')
print(df.dtypes)

# Domains, Intervals
print('\n- Domains, Intervals')
for c in df.columns:
    x = df[c].unique()
    x.sort()
    print('  - %-16s: min: %d, max: %d' % (c, df[c].min(), df[c].max()))
    print('    values: %s' % x)

print('\n- Nulls')
for c in df.columns:
    print('  - %-16s: Nulls: %d' % (c, df[c].isna().sum()))


# In[8]:


# create X and y to match Scikit-Learn parlance

features = ['Recency', 'Frequency', 'Monetary', 'Time']
outcome = 'Donated_Mar_2007'

# X include all the features
X = df[features].copy()
# y is the target variable
# Note: As it is a classification problem, 0 and 1 are converted to '0' and '1' (int to str)
y = df[outcome].astype(str).copy()


# In[9]:


df.head()


# In[10]:


df['Donated_Mar_2007'].value_counts(normalize=True)


# In[11]:


## Check the data

# About X
print('X is a %s' % type(X))
print('X has %d rows and %d columns' % X.shape)
print('Basic Statistics about X%s' % ('_'*50))
print(df.describe())
print('')
print('Sample of X%s' % ('_'*50))
print(X.head())


# In[12]:


# About y
print('y is a %s' % type(y))
print('y has %d rows' % y.shape)
print('')
print('Sample of y%s' % ('_'*50))
print(y[:5])


# In[13]:


# Visualise the data points

# visualise features in pairs
sns.pairplot(df)
plt.show()


# # Step 5: Prepare the Data
# - Deal with the data as required by the modelling technique
#     - **Outliers** (remove or adjust if possible or necessary)
#     - **Null** (remove or interpolate if possible or necessary)
#     - **Missing Values** (remove or interpolate if possible or necessary)
#     - **Coded content** (transform if possible or necessary [str to number or vice-versa])
#     - **Normalisation** (if possible or necessary)
#     - **Feature Engeneer** (if useful or necessary)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Filter/subset/clean the data according to your EDA findings


# # Step 6: Modelling
# Refer to the Problem and Main Question.
# - What are the input variables (features)?
# - Is there an output variable (label)?
# - If there is an output variable:
#     - What is it?
#     - What is its type?
# - What type of Modelling is it?
#     - [ ] Supervised
#     - [ ] Unsupervised 
# - What type of Modelling is it?
#     - [ ] Regression
#     - [ ] Classification (binary) 
#     - [ ] Classification (multi-class)
#     - [ ] Clustering

# In[14]:


print('- What are the input variables (features)?')
print('  - %s' % ', '.join(features))
print('- Is there an output variable (label)?')
print('  - %s' % ('Yes' if outcome else 'No'))
print('- If there is an output variable:')
print('    - Which one is it?')
print('      - %s' % outcome)
print('    - What is its type?')
print('      - %s' % y.dtypes)
print('  - What type of Modelling is it?')
print('    - [%s] Supervised' % ('x' if outcome else ' '))
print('    - [%s] Unsupervised' % (' ' if outcome else 'x'))
print('  - What type of Modelling is it?')
print('    - [%s] Regression' % ('x' if y.dtypes != 'object' else ' '))
print('    - [%s] Classification (binary)' % ('x' if (y.dtypes == 'object') and (len(y.unique()) == 2) else ' '))
print('    - [%s] Classification (multi-class)' % ('x' if (y.dtypes == 'object') and (len(y.unique()) != 2) else ' '))
print('    - [%s] Clustering' % (' ' if outcome else 'x'))


# # Step 7: Split the Data
# 
# Need to check for **Supervised** modelling:
# - Number of known cases or observations
# - Define the split in Training/Test or Training/Validation/Test and their proportions
# - Check for unbalanced classes and how to keep or avoid it when spliting

# In[15]:


# Split your data
X.shape


# In[16]:


## Create training and testing subsets
test_size = X.shape[0] - 500

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = test_size,
                                                    random_state = 100666001,
                                                    stratify = y)


# In[17]:


y_train.value_counts(normalize=True)


# In[18]:


y_test.value_counts(normalize=True)


# # Step 8: Define a Model
# 
# Define the model and its hyper-parameters.
# 
# Consider the parameters and hyper-parameters of each model at each (re)run and after checking the efficiency of a model against the training and test datasets.

# In[19]:


# Choose a model or models

model = RandomForestClassifier()
print(model)


# # Step 9: Fit the Model

# In[20]:


# Fit model
model.fit(X_train, y_train)


# In[21]:


model.score(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[22]:


# Feature Selection
model.feature_importances_


# In[23]:


model.score(X_test, y_test)


# In[24]:


important = pd.DataFrame(list(zip(X_train.columns, model.feature_importances_)))
important


# # Step 10: Verify and Evaluate the Training Model
# - Use the **training** data to make predictions
# - Check for overfitting
# - What metrics are appropriate for the modelling approach used
# - For **Supervised** models:
#     - Check the **Training Results** with the **Training Predictions** during development
# - Analyse, modify the parameters and hyper-parameters and repeat (within reason) until the model does not improve

# In[ ]:





# In[25]:


# Evaluate model against training set
def show_summary_report(actual, prediction, probabilities):

    if isinstance(actual, pd.Series):
        actual = actual.values.astype(int)
    prediction = prediction.astype(int)

    print('Accuracy : %.4f [TP / N] Proportion of predicted labels that match the true labels. Best: 1, Worst: 0' % accuracy_score(actual, prediction))
    print('Precision: %.4f [TP / (TP + FP)] Not to label a negative sample as positive.        Best: 1, Worst: 0' % precision_score(actual, prediction))
    print('Recall   : %.4f [TP / (TP + FN)] Find all the positive samples.                     Best: 1, Worst: 0' % recall_score(actual, prediction))
    print('ROC AUC  : %.4f                                                                     Best: 1, Worst: < 0.5' % roc_auc_score(actual, probabilities[:, 1]))
    print('-' * 107)
    print('TP: True Positives, FP: False Positives, TN: True Negatives, FN: False Negatives, N: Number of samples')

    # Confusion Matrix
    mat = confusion_matrix(actual, prediction)

    # Precision/Recall
    precision, recall, _ = precision_recall_curve(actual, prediction)
    average_precision = average_precision_score(actual, prediction)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(actual, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)


    # plot
    fig, ax = plt.subplots(1, 3, figsize = (18, 6))
    fig.subplots_adjust(left = 0.02, right = 0.98, wspace = 0.2)

    # Confusion Matrix
    sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False, cmap = 'Blues', ax = ax[0])

    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('True label')
    ax[0].set_ylabel('Predicted label')
    
    # Precision/Recall
    step_kwargs = {'step': 'post'}
    ax[1].step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    ax[1].fill_between(recall, precision, alpha = 0.2, color = 'b', **step_kwargs)
    ax[1].set_ylim([0.0, 1.0])
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('2-class Precision-Recall curve')

    # ROC
    ax[2].plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (AUC = %0.2f)' % roc_auc)
    ax[2].plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    ax[2].set_xlim([0.0, 1.0])
    ax[2].set_ylim([0.0, 1.0])
    ax[2].set_xlabel('False Positive Rate')
    ax[2].set_ylabel('True Positive Rate')
    ax[2].set_title('Receiver Operating Characteristic')
    ax[2].legend(loc = 'lower right')

    plt.show()


# In[26]:


y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)


# In[27]:


show_summary_report(y_train, y_train_pred, y_train_prob)


# # Step 11: Make Predictions and Evaluate the Test Model
# **NOTE**: **Do this only after not making any more improvements in the model**.
# 
# - Use the **test** data to make predictions
# - For **Supervised** models:
#     - Check the **Test Results** with the **Test Predictions**

# In[28]:


# Evaluate model against test set
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)


# In[29]:


show_summary_report(y_test, y_test_pred, y_test_prob)


# # Step 12: Solve the Problem or Answer the Question
# The results of an analysis or modelling can be used:
# - As part of a product or process, so the model can make predictions when new input data is available
# - As part of a report including text and charts to help understand the problem
# - As input for further questions

# >

# >

# >

# In[ ]:





# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# > > > > > > > > > © 2021 Institute of Data
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# 
