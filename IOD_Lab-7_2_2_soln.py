#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Lab 7.2.2: Boosting
# 
# INSTRUCTIONS:
# 
# - Read the guides and hints then create the necessary analysis and code to find an answer and conclusion for the scenario below.
# - The baseline results (minimum) are:
#     - **Accuracy** = 0.9429
#     - **ROC AUC**  = 0.9333
# - Try to achieve better results!

# # Foreword
# It is common that companies and professionals start with the data immediately available. Although this approach works, ideally the first step is to identify the problem or question and only then identify and obtain the set of data that can help to solve or answer the problem.
# 
# Also, given the current abundance of data, processing power and some particular machine learning methods, there could be a temptation to use ALL the data available. **Quality** is _**better**_ than **Quantity**!
# 
# Part of calling this discipline **Data Science** is that it is supposed to follow a process and not reach conclusions without support from evidence.
# 
# Moreover, it is a creative, exploratory, laborious, iterative and interactive process. It is part of the process to repeat, review and change when finding a dead-end.

# ## Scenario: Predicting Breast Cancer
# The dataset you are going to be using for this laboratory is popularly known as the **Wisconsin Breast Cancer** dataset. The task related to it is Classification.
# 
# The dataset contains a total number of _10_ features labelled in either **benign** or **malignant** classes. The features have _699_ instances out of which _16_ feature values are missing. The dataset only contains numeric values.

# # Step 1: Define the problem or question
# Identify the subject matter and the given or obvious questions that would be relevant in the field.
# 
# ## Potential Questions
# List the given or obvious questions.
# 
# ## Actual Question
# Choose the **one** question that should be answered.

# In[26]:


## Import Libraries
import itertools

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import warnings
warnings.filterwarnings('ignore')


# # Step 2: Find the Data
# ### Wisconsin Breast Cancer DataSet
# - **Citation Request**
# 
#     This breast cancer databases was obtained from the **University of Wisconsin Hospitals**, **Madison** from **Dr. William H. Wolberg**. If you publish results when using this database, then please include this information in your acknowledgements.
# 
# - **Title**
# 
#     Wisconsin Breast Cancer Database (January 8, 1991)
# 
# - **Sources**
#     - **Creator**
#             Dr. William H. Wolberg (physician)
#             University of Wisconsin Hospitals
#             Madison, Wisconsin
#             USA
#     - **Donor**
#             Olvi Mangasarian (mangasarian@cs.wisc.edu)
#             Received by David W. Aha (aha@cs.jhu.edu)
#     - **Date**
#             15 July 1992
#         
# ### UCI - Machine Learning Repository
# - Center for Machine Learning and Intelligent Systems
# 
# The [**UCI Machine Learning Repository**](http://archive.ics.uci.edu/ml/about.html) is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.

# # Step 3: Read the Data
# - Read the data
# - Perform some basic structural cleaning to facilitate the work

# In[27]:


## reading and inspect the data

columns = [
    'Sample_Number',
    'Clump_Thickness',
    'Uniformity_of_Cell_Size',
    'Uniformity_of_Cell_Shape',
    'Marginal_Adhesion',
    'Single_Epithelial_Cell_Size',
    'Bare_Nuclei',
    'Bland_Chromatin',
    'Normal_Nucleoli',
    'Mitoses',
    'Class'
]
df = pd.read_csv(
    filepath_or_buffer = 'breast-cancer-wisconsin-data-old.csv',
    header = None,
    names = columns,
    usecols = columns[1:], # do not use the first column
    na_values = '?' # convert the '?' to NA
)


# In[28]:


df.describe()


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

# In[29]:


# Domains, Intervals
print('- Domains, Intervals')
for c in df.columns:
    x = df[c].unique()
    x.sort()
    print('+ %-27s: (%-s)' % (c, df[c].dtypes.name))

    if df[c].dtypes.name != 'object':
        print('  min: %d, max: %d' % (df[c].min(), df[c].max()))
    print('  values: %s' % x)


# # Step 5: Prepare the Data
# - Deal with the data as required by the modelling technique
#     - **Outliers** (remove or adjust if possible or necessary)
#     - **Null** (remove or interpolate if possible or necessary)
#     - **Missing Values** (remove or interpolate if possible or necessary)
#     - **Coded content** (transform if possible or necessary [str to number or vice-versa])
#     - **Normalisation** (if possible or necessary)
#     - **Feature Engeneer** (if useful or necessary)

# In[30]:


print(df['Bare_Nuclei'].value_counts(dropna = False))


# In[31]:


# Convert the DataFrame object into NumPy array otherwise you will not be able to impute
values = df.values

# define the criteria for dealing with the missing values
imputer = SimpleImputer(
    missing_values = np.nan,
    strategy = 'median'
)
# Now impute it
imputedData = imputer.fit_transform(values)


# In[32]:


scaler = MinMaxScaler(feature_range = (0, 1))
normalizedData = scaler.fit_transform(imputedData)


# In[33]:


# move the data back to a dataframe
df_n = pd.DataFrame.from_records(normalizedData, columns = columns[1:])
df_n['Class'] = df_n['Class'].astype(np.int8)
df_n.sample(5)


# In[34]:


# Check for Outliers
for c in df_n.columns:
    fig, ax = plt.subplots(2, 
                           figsize = (10, 5),
                           sharex = True, 
                           gridspec_kw = {'height_ratios': (0.15, 0.85)})

    ax_box = ax[0]
    ax_box = sns.boxplot(df_n[c], ax = ax_box)
    ax_box.set(title = c, yticks = [], xlabel = '')
    sns.despine(ax = ax_box, left = True)

    ax_hist = ax[1]
    ax_hist = sns.distplot(df_n[c], ax = ax_hist)
    ax_hist.set(xlabel = '')
    sns.despine(ax = ax_hist)

plt.show()


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

# In[35]:


df.columns


# In[36]:


# create X and y to match Scikit-Learn parlance

features = columns[1:-1] # discard the first and last columns
outcome = 'Class'

# X include all the features
X = df_n[features].copy()
# y is the target variable
y = df_n[outcome].copy()


# In[37]:


## Check the data

# About X
X.describe().T


# In[38]:


X.sample(5)


# In[39]:


# About y
y.describe().T


# In[49]:


y.sample(5)


# In[50]:


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
print('    - [%s] Regression' % ('x' if (y.dtypes != 'object') & (len(y.unique()) >= 20) else ' '))
print('    - [%s] Classification (binary)' % ('x' if len(y.unique()) == 2 else ' '))
print('    - [%s] Classification (multi-class)' % ('x' if (y.dtypes == 'object') and (len(y.unique()) != 2) else ' '))
print('    - [%s] Clustering' % (' ' if outcome else 'x'))


# In[51]:


colormap = plt.cm.RdBu
plt.figure(figsize = (10, 10))
plt.title('Pearson Correlation of Features', size = 15)
sns.heatmap(df_n.astype(float).corr(),
            linewidths = 0.1,
            vmax = 1.0,
            square = True,
            cmap = colormap,
            linecolor = 'white',
            annot = True)
plt.show()


# In[52]:


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
print('    - [%s] Regression' % ('x' if (y.dtypes != 'object') & (len(y.unique()) >= 20) else ' '))
print('    - [%s] Classification (binary)' % ('x' if len(y.unique()) == 2 else ' '))
print('    - [%s] Classification (multi-class)' % ('x' if (y.dtypes == 'object') and (len(y.unique()) != 2) else ' '))
print('    - [%s] Clustering' % (' ' if outcome else 'x'))


# # Step 7: Split the Data
# 
# Need to check for **Supervised** modelling:
# - Number of known cases or observations
# - Define the split in Training/Test or Training/Validation/Test and their proportions
# - Check for unbalanced classes and how to keep or avoid it when spliting

# In[53]:


## Create training and testing subsets
test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = test_size,
                                                    random_state = 100666001,
                                                    stratify = y)


# # Step 9: Verify and Evaluate the Training Model
# - Use the **training** data to make predictions
# - Check for overfitting
# - What metrics are appropriate for the modelling approach used
# - For **Supervised** models:
#     - Check the **Training Results** with the **Training Predictions** during development
# - Analyse, modify the parameters and hyper-parameters and repeat (within reason) until the model does not improve

# # Step 8: Define and Fit Models
# 
# Define the model and its hyper-parameters.
# 
# Consider the parameters and hyper-parameters of each model at each (re)run and after checking the efficiency of a model against the training and test datasets.

# In[54]:


kfold = KFold(n_splits = 10)
cart = DecisionTreeClassifier()
cart.fit(X_train, y_train)
results = cross_val_score(cart, X_train, y_train, cv = kfold)
print('Decision Tree, Cross-Validation mean: %.4f' % results.mean())


# In[56]:


seed = 7
num_trees = 70
kfold = model_selection.KFold(n_splits = 10)
model = AdaBoostClassifier(n_estimators = num_trees, random_state = seed)
model.fit(X_train, y_train)
results = cross_val_score(model, X_train, y_train, cv = kfold)
print('AdaBoost, Cross-Validation mean: %.4f' % results.mean())


# In[57]:


np.mean(cross_val_score(model, X_test, y_test, cv = kfold))


# In[ ]:


def show_summary_report(actual, prediction):

    if isinstance(actual, pd.Series):
        actual = actual.values
    if actual.dtype.name == 'object':
        actual = actual.astype(int)
    if prediction.dtype.name == 'object':
        prediction = prediction.astype(int)

    accuracy_ = accuracy_score(actual, prediction)
    precision_ = precision_score(actual, prediction)
    recall_ = recall_score(actual, prediction)
    roc_auc_ = roc_auc_score(actual, prediction)

    print('Accuracy : %.4f [TP / N] Proportion of predicted labels that match the true labels. Best: 1, Worst: 0' % accuracy_)
    print('Precision: %.4f [TP / (TP + FP)] Not to label a negative sample as positive.        Best: 1, Worst: 0' % precision_)
    print('Recall   : %.4f [TP / (TP + FN)] Find all the positive samples.                     Best: 1, Worst: 0' % recall_)
    print('ROC AUC  : %.4f                                                                     Best: 1, Worst: < 0.5' % roc_auc_)
    print('-' * 107)
    print('TP: True Positives, FP: False Positives, TN: True Negatives, FN: False Negatives, N: Number of samples')

    # Confusion Matrix
    mat = confusion_matrix(actual, prediction)

    # Precision/Recall
    precision, recall, _ = precision_recall_curve(actual, prediction)
    average_precision = average_precision_score(actual, prediction)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(actual, prediction)
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
    
    return (accuracy_, precision_, recall_, roc_auc_)


# In[ ]:


# Keep the results in a dataframe
results = pd.DataFrame(columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'ROC_AUC'])

models = ['Decision Tree', 'AdaBoost']
for i, m in enumerate([cart, model]):
    print('*' * (len(models[i]) + 4))
    print('* %s *' % models[i])
    print('*' * (len(models[i]) + 4))
    predictions = m.predict(X_train)
    # show the report
    accuracy_, precision_, recall_, roc_auc_ = show_summary_report(y_train, predictions)
    # keep the results
    results.loc[i] = {'Model': models[i], 
                      'Accuracy': accuracy_, 
                      'Precision': precision_,
                      'Recall': recall_,
                      'ROC_AUC': roc_auc_}
    print()


# In[ ]:


results


# # Step 10: Make Predictions and Evaluate the Test Model
# **NOTE**: **Do this only after not making any more improvements in the model**.
# 
# - Use the **test** data to make predictions
# - For **Supervised** models:
#     - Check the **Test Results** with the **Test Predictions**

# In[ ]:


# gbm.fit(xx_train, y_train)
predictions = model.predict(X_test)


# In[ ]:


accuracy_, precision_, recall_, roc_auc_ = show_summary_report(y_test, predictions)
# keep the results
results.loc[len(results)] = {
    'Model': 'AdaBoost (with test)', 
    'Accuracy': accuracy_, 
    'Precision': precision_,
    'Recall': recall_,
    'ROC_AUC': roc_auc_}
print()


# In[ ]:


results


# # Step 11: Solve the Problem or Answer the Question
# The results of an analysis or modelling can be used:
# - As part of a product or process, so the model can make predictions when new input data is available
# - As part of a report including text and charts to help understand the problem
# - As input for further questions

# In[ ]:


r = df.iloc[X_test.index[0]]
for i, c in enumerate(df.columns[:-1]):
    print('%-27s: %d' % (c, int(r[i])))
print('%-27s: %s' % (df.columns[-1], 'Benign' if r[-1] == 2 else 'Malignant'))
print('%-27s: %s' % ('Prediction', 'Benign' if predictions[0] == 0 else 'Malignant'))


# In[ ]:


X_test.index[0]


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
