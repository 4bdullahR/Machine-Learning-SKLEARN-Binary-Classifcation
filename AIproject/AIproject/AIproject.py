import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #line 13
from sklearn.linear_model import LogisticRegression #line 17
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from random import randrange #line 31

#laod the data from the file
df = pd.read_csv("datafile.csv").dropna()

#load all features
X = df[["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke",
             "prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]].to_numpy()
y = df["TenYearCHD"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34)



logReg = LogisticRegression(solver='lbfgs', max_iter=999999)
logReg.fit(X_train,y_train)
y_predict = logReg.predict(X_test)

print(f"Accuracy score is: {accuracy_score(y_test,y_predict)*100:.02f}%")
print(f"precision score is: {precision_score(y_test,y_predict)*100:.02f}%")

#print(f"recall_score is: {recall_score(y_test,y_predict)*100:.02f}%")
#print(f"f1_score is: {f1_score(y_test,y_predict)*100:.02f}%")
print("accual \t: predicted **10 random samples**")
for i in range(10):
    index = randrange(len(y_test))
    print(f'{y_test[index]} \t : {y_predict[index]}')

