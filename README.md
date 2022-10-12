# Data_Mining_Resource
## Synopsis
Practical of Claasification
## Contents
- README – readme file for the Classification project
- Assignment2022.sqlite - Original data, including `train` and `test` two tables.	
- main.py	- Read data, do data prepaeration and train models(`KNN`, `Decision Tree`, `Naive Bayes` and `Support vector machines(SVM)`)		
- main2.py - edited version of main.py
- threshold.py - Loop for each threshold of correlation to run main2.py to get best accuracy
## Dependencies
- sqlite3
- numpy 
- pandas
- seaborn
- matplotlib
- sklearn
- statistics
- time
## Version information
12 Oct 2022 - last version of Practical of Claasification
## How to run
Find the best threshold
```
python3 threshold.py
```
Run the main program
```
python3 main.py
```
## Result
- Accrucacy with different thresholds

![accuracy_with_diff_thresholds](https://user-images.githubusercontent.com/20329677/195328147-304bc6fa-49cf-46a6-9c55-c93912085111.jpg)
