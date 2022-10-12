import sqlite3 as sql
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import main2
import statistics

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
class_corr = upper.iloc[:-1, -1]  # from index, att00 to att29
class_corr_sorted = class_corr.sort_values(ascending=True, key=abs)
print(class_corr_sorted)

threshold_list = abs(class_corr_sorted).tolist()
print(threshold_list)
# correlation less than |+-0.01|
knn_accuracy_list = []
dt_accuracy_list = []
nb_accuracy_list = []
#threshold_list = [0.01, 0.02, 0.03]
knn_best_threshold = 0
knn_best_accuracy = 0
best_threshold = 0
best_accuracy = 0
for threshold in threshold_list:
    '''
    irrelevant_cols_arr = class_corr.loc[abs(class_corr) <= threshold]
    irrelevant_cols = irrelevant_cols_arr.index.values.tolist()
    old = len(irrelevant_cols)
    if "index" in irrelevant_cols:
        irrelevant_cols.remove("index")
    print(f"{threshold} : \n {irrelevant_cols} {old} -> {len(irrelevant_cols)}")
    '''
    accuracy_list = main2.running(threshold)
    knn_accuracy_list.append(accuracy_list[0])
    dt_accuracy_list.append(accuracy_list[1])
    nb_accuracy_list.append(accuracy_list[2])

    if accuracy_list[0] > knn_best_accuracy:
        knn_best_threshold = threshold
        knn_best_accuracy = accuracy_list[0]
    print(f"knn threshold = {threshold}: accuracy_list = {accuracy_list}")
    if statistics.mean(accuracy_list) > best_accuracy:
        best_threshold = threshold
        best_accuracy = statistics.mean(accuracy_list)
    print(f"all threshold = {threshold}: accuracy_list = {accuracy_list}")

print(f"best_threshold = {knn_best_threshold}: best_accuracy = {knn_best_accuracy}")
# plot lines
fig, ax = plt.subplots()
plt.plot(threshold_list, knn_accuracy_list, label="knn accuracy")
plt.plot(threshold_list, dt_accuracy_list, label="dt accuracy")
plt.plot(threshold_list, nb_accuracy_list, label="nb accuracy")
#plt.axvline(x=knn_best_threshold, linestyle="--", color="hotpink")
plt.axvline(x=best_threshold, linestyle="--", color="hotpink")
plt.axvline(x=0.09828981875734498, linestyle="--", color="hotpink")
#ax.annotate(knn_best_threshold, xy=(knn_best_threshold, knn_best_accuracy),
#            xytext=(0.01, knn_best_accuracy), arrowprops={"arrowstyle":"->", "color":"gray"})
ax.annotate(best_threshold, xy=(best_threshold, 0.76),
            xytext=(0.03, 0.8), arrowprops={"arrowstyle":"->", "color":"gray"})
ax.annotate(0.09828981875734498, xy=(0.09828981875734498, 0.7),
            xytext=(0.12, 0.65), arrowprops={"arrowstyle":"->", "color":"gray"})

plt.legend(loc='best')

plt.show()

# prepare data frame
d = {'threshold': threshold_list,
     'knn': knn_accuracy_list,
     'td': dt_accuracy_list,
     'nb': nb_accuracy_list
     }
df_result = pd.DataFrame(data=d, dtype='int64')
df_result.to_csv('out.csv', index=True)
