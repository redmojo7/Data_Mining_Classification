import sqlite3 as sql
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, naive_bayes
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import svm
import time

start = time.process_time()
# connect to a database on disk with a given name
# check the file browser on the left to see that the file has been created.
con = sql.connect('Assignment2022.sqlite')

# Load the data into a DataFrame
df_train = pd.read_sql_query("SELECT * from train", con)
df_test = pd.read_sql_query("SELECT * from test", con)

# look at shape
print(f"df_train shape : {df_train.shape}")
print(f"df_test shape : {df_test.shape}")

# concat df_train and df_test
concat_pf = pd.concat([df_train, df_test])
print(f"concat_pf shape {concat_pf.shape}\n")

#  split the data into groups based on some 'class' criteria.
print(f"\ngroup by 'class: {df_train.groupby('class').size()}\n")
print(f"concat_pf describe : {concat_pf.describe()}")

# Identify and remove irrelevant attributes.
print(f"\ngroup by 'Att17: {concat_pf.groupby('Att17').size()}\n")
print(f"group by 'Att26: {concat_pf.groupby('Att26').size()}\n")
# compute correlation matrix
corr_matrix = df_train.corr(numeric_only=True)
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
class_corr = upper['class']
class_corr_sorted = class_corr.sort_values(ascending=True, key=abs)
print(class_corr_sorted)

threshold = 0.09828981875734498
irrelevant_cols_arr = class_corr.loc[abs(class_corr) <= threshold]
irrelevant_cols = irrelevant_cols_arr.index.values.tolist()
old = len(irrelevant_cols)
if "index" in irrelevant_cols:
    irrelevant_cols.remove("index")
print(f"{threshold} : \n {irrelevant_cols} {old} -> {len(irrelevant_cols)}")



# use seaborn to do the plot
plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)

# drop columns with correlation less than |+-0.09828981875734498|
print(f"\n Drop column {irrelevant_cols} which correlation less than |+-{threshold}|")
concat_pf.drop(irrelevant_cols, axis=1, inplace=True)
# drop column 'index'
# concat_pf.drop(['index'], axis=1, inplace=True)

# nunique
print(f"\nconcat_pf nunique : {concat_pf.nunique()}")


# Detect and handle missing entries.
# find all rows with missing data(more than 2 columns value missing)
def missing_on_row(df):
    missing_list = []
    total = df.shape[1]
    for i in range(len(df.index)):
        missing = df.iloc[i].isnull().sum()
        if missing > 2:
            missing_list.append([i, round(missing, 4)])
    return missing_list


# find columns which have missing data and percentage of missing data
def missing_on_col(df):
    """
    For each attribute/column in the dataframe `df`, count the number of missing entries.
    Return a list of all the columns and missing rate.
    """
    missing_list = []
    total = df.shape[0]
    for attribute in df.columns:
        missing = df[attribute].isna().sum()
        if missing != 0:
            missing_list.append([attribute, missing / total])
    return missing_list


print(f"\n df_train missing data columns:")
[print(x) for x in missing_on_col(df_train)]

print(f"\n df_train missing data rows(more than 2 columns value missing):")
[print(x) for x in missing_on_row(df_train)]

#print(df_train[["Att09", "Att23", "Att25"]].dtypes)
#print(df_train[["Att09", "Att23", "Att25"]].describe())


# decide to replace missing data with mean for columns "Att09" and "Att25"
# and drop column "Att23"

# replace these missing values with either the mean or the mode of the attribute.
def replace_missing_with_mean(df, cols_to_fill):
    for attribute in cols_to_fill:
        print(f"{attribute}: has ", df[attribute].isna().sum(), "NaN")
        # compute the mean
        mean = df_train[attribute].mean()
        print(f"df_train[{attribute}]'s mean is {mean}")
        # now use the fillna function to replace the NaN avalues with the mean value
        df[attribute].fillna(mean, inplace=True)
        print(f"Replace NaN at {attribute} with {mean}")
        # check that the replacement has worked.
        print(f"{attribute}: has ", df[attribute].isna().sum(), "NaN")


# replace missing data for concat_pf column 'Att25'
print("\nReplace missing data for concat_pf column 'Att25'")
replace_missing_with_mean(concat_pf, ["Att25"])

# check
print(f"\n df_train missing data columns:")
[print(x) for x in missing_on_col(df_train)]

# drop columns 'Att09', 'Att23'
print("Drop column 'Att09', 'Att23' for df_train & df_test "
      "(because data missing)\n['Att09', 0.1996] ['Att23', 0.5964]")
drop_cols = ["Att09", "Att23"]
for col in drop_cols:
    if col in concat_pf.columns:
        print(f" drop column {col}")
        concat_pf.drop(columns=col, axis=1, inplace=True)

# look at shape
print(f"\nconcat_pf shape : {concat_pf.shape}")

# Detect and handle duplicates (both instances and attributes).
# count all duplicate rows across all columns
print(f"\n df_train duplicated rows: {df_train.duplicated().sum()}")
# count all duplicated columns
print(f"df_train shape: {df_train.shape}")
df_train_dup = df_train.T.drop_duplicates().T
print(f"df_train_dup shape: {df_train_dup.shape}")
# same shape, that means there is no duplicated columns
print(f"df_train duplicated columns: {df_train.shape[1]-df_train_dup.shape[1]}")

# Select suitable data types for attributes.
# checking for numeric (is it should be numeric?)
# class is float64?  object categorical data
# it doest effect the result, but it would make sense to use data type int
# only for df_train
print(f"\ndf_train column 'class' data type : {df_train['class'].dtype}")
# reset type for class as int64
df_train['class'] = df_train['class'].astype('int64', copy=False)
# check
print(f"train_df column 'class' data type : {df_train['class'].dtype}")

# get all numeric attributes except columns "index" and "class"
train_num_columns = concat_pf.loc[:, ~concat_pf.columns.isin(["index", "class"])] \
    .select_dtypes(exclude=object).columns
print(f"concat_pf numeric attributes: {train_num_columns}")

# convert all categorical columns to numeric
#
obj_columns = concat_pf.select_dtypes(include=object).columns
print(f"\nconcat_pf categorical columns \n: {obj_columns}")

# have a look columns: ['Att01', 'Att11', 'Att12']
print(f"\n{concat_pf[['Att01', 'Att11', 'Att12']].head()}")

# look at shape
print(f"\ndf_train obj_columns : {obj_columns.size}")

# before encoding, we have to concat df_train_scaled and df_test_scaled
# to make sure pd.get_dummies to encode categorical columns with same strategy
print(f"\nconcat_pf shape {concat_pf.shape}\n")
print(f"concat_pf head: \n {concat_pf.head()}")
print(f"concat_pf tail: \n {concat_pf.tail()}")
# encode with One-Hot Encoding
print()
#   Convert categorical variable into dummy/indicator variables.
#   (except column 'class')
print("\nConvert categorical variable into dummy/indicator variables.")
df_OHE = pd.get_dummies(concat_pf.iloc[:, :-1])
print(f"df_OHE shape: {df_OHE.shape}")
print(f"df_OHE columns:\n: {df_OHE.columns}")

# make sure all are numerical data
print(f"\n df_OHE shape:: {df_OHE.shape}")
obj_columns = df_OHE.select_dtypes(include=object).columns
print(f"df_OHE categorical columns : {len(obj_columns)}")
num_columns = df_OHE.select_dtypes(exclude=object).columns
print(f"df_OHE numeric columns: {len(num_columns)}")

# Perform data transformation (such as scaling/standardization) if needed.
# At this stage you will want to rescale your variable to bring them to a similar numeric range
# This is particularly important for KNN, as it uses a distance metric

# standardise numeric attributes
# all columns are numeric


# copy df_OHE and df_OHE_scaled
df_OHE_scaled = df_OHE.copy()
# for all columns except index
columns = df_OHE_scaled.iloc[:, 1:].columns
print("Do StandardScaler for df_OHE_scaled")
# describe df_OHE_scaled
print(f"\ndf_OHE_scaled describe: \n {df_OHE_scaled.describe()}")
df_OHE_scaled[columns] = StandardScaler().fit_transform(df_OHE_scaled[columns])
print(f"\ndf_OHE_scaled describe: \n {df_OHE_scaled.describe()}")

print(f"\ndf_OHE_scaled head: \n {df_OHE_scaled.head()}")
print(f"\ndf_OHE_scaled tail: \n {df_OHE_scaled.tail()}")

# split df_OHE_scaled to train and test by index, then remove index
# drop "index"
df_train_final = df_OHE_scaled.loc[df_OHE_scaled['index'] < 5000].iloc[:, 1:]
df_test_final = df_OHE_scaled.loc[df_OHE_scaled['index'] >= 5000].iloc[:, 1:]
print(f"\n df_train_final shape: {df_train_final.shape}")
print(f"\n df_test_final shape: {df_test_final.shape}")
# print(df_train_final.head())
# print(df_train_final.tail())

# Split the data into two subsets:
# A training subset comprising 75% of the data
# A testing subset comprising 25% of the data
X = df_train_final
y = df_train.iloc[:, -1]
print(f"\n X shape: {X.shape}")
print(f"y shape: {y.shape}\n")

# use a teste sieve of 25% # this random state ensures that we get the same subset each time we call this cell
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

print("\nUsing KNN")
# make a KNN classifier object
knn = KNeighborsClassifier()
print("Tune parameters for KNN")
# Create a dictionary of all the parameters we'll be iterating over
parameters = {'weights': ('uniform', 'distance'),  # this should be the different weighting schemes
              'n_neighbors': range(3, 30)  # range(1, 30)
              # ,'p': [1, 2]
              }  # this should be a list of the nearest neigbhours
# create a GridSearchCV object to do the training with cross validation
gscv = GridSearchCV(estimator=knn,
                    param_grid=parameters,
                    cv=10,  # the cross validation folding pattern
                    scoring='accuracy')
# now train our model
best_knn = gscv.fit(X_train, y_train)

print(f'Best params {best_knn.best_params_}')
print(f'Best accuracy = {best_knn.best_score_}')

# y_pred = gscv.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
#
#
#       Decision Tree
#
#
print("\nUsing Decision Tree")
# make a Decision Tree classifier object
dtc = tree.DecisionTreeClassifier()
print("\nTune parameters for Decision Tree")
# Create a dictionary of all the parameters we'll be iterating over
parameters = {'criterion': ("gini", "entropy"),  # this should be the different splitting criteria
              'min_samples_split': range(2, 30)  # range(2, 46)
              }  # this should be the different values for min_samples_split
gscv = GridSearchCV(estimator=dtc,
                    param_grid=parameters,
                    cv=10,  # the cross validation folding pattern
                    scoring='accuracy')
best_dtc = gscv.fit(X_train, y_train)
print(f'Best params {best_dtc.best_params_}')
print(f'Best accuracy = {best_dtc.best_score_}')

#
#
#
#       Naive Bayes
#
#
print("\nUsing Naive Bayes")
# make a Naive Bayes classifier object
nb = naive_bayes.GaussianNB()
print("\nTune parameters for Naive Bayes")
params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
gs_NB = GridSearchCV(estimator=nb,
                     param_grid=params_NB,
                     cv=10,  # the cross validation folding pattern
                     verbose=1,
                     scoring='accuracy')
best_nb = gs_NB.fit(X_test, y_test)
print(f'Best params {best_nb.best_params_}')
print(f'Best accuracy = {best_nb.best_score_}')
#
#
#   Support vector machines (SVM)
#
print("\nUsing Support vector machines (SVM)")
# Create a svm Classifier
svm_Classifier = svm.SVC()  # Linear Kernel
print("\nTune parameters for SVM")
# Create a dictionary of all the parameters we'll be iterating over
parameters = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ["rbf"]}
gscv = GridSearchCV(estimator=svm_Classifier,
                    param_grid=parameters,
                    cv=10,  # the cross validation folding pattern
                    scoring='accuracy')
best_svm = gscv.fit(X_train, y_train)
print(f'Best params {best_svm.best_params_}')
print(f'Best accuracy = {best_svm.best_score_}')

# best two models are : KNN with accuracy = 0.8312000000000002
#                       SVM with accuracy = 0.8733333333333333

# to predict test
# to predict using best KNN for test
best_knn_classifier = KNeighborsClassifier(n_neighbors=4, weights='distance')
# Training the model with whole training data
best_knn_classifier.fit(X, y)
# to predict using KNN for test
knn_y_pred = best_knn_classifier.predict(df_test_final)
# print(kenn_y_pred)

# to predict using SVM for table test
best_svmclassifier = svm.SVC(C=1, gamma=0.1)
# Training the model.
best_svmclassifier.fit(X, y)
svm_y_pred = best_svmclassifier.predict(df_test_final)
# print(svm_y_pred)

print()
# prepare data frame
d = {'index': range(5000, 5500),
     'Predict1': knn_y_pred,
     'Predict2': svm_y_pred}
df_result = pd.DataFrame(data=d, dtype='int64')

# prepare export file
name = "Answers.sqlite"
if os.path.exists(name):
    print(f"remove old file named {name}")
    os.remove(name)
print(f"create file named {name}")
f = open(name, "x")
f.close

# export dataframe to sqlite
database = "Answers.sqlite"
conn = sql.connect(database)
df_result.to_sql(name='answers', con=conn, schema=None, if_exists='replace',
                 index=False, index_label=None)
print("exported dataframe to sqlite")

# check sqlite
diff = df_result.loc[df_result['Predict1'] != df_result['Predict2']]
print(diff)
df_diff = pd.read_sql_query("SELECT * from answers where predict1 != predict2", conn)
print(df_diff)
conn.close()

print(f"\n df_result group by Predict1: {df_result.groupby('Predict1').size()}\n")
print(f"\n df_result group by Predict12: {df_result.groupby('Predict2').size()}\n")

end = time.process_time()
print(end - start)
