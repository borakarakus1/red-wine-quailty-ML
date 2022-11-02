from statistics import correlation
import numpy as np
import pandas as pd

data = pd.read_csv("winequality-red.csv")
data.head()

#print(wine.isna().sum()) # no avaible null value

#print(wine.corr()) # checking correlation

# data visualition

import matplotlib.pyplot as plt  
import seaborn as sns

data.hist(figsize=(8,8),bins=50)
plt.show()

# heat map to see correlation

correlation = data.corr()
sns.heatmap(data,annot=True)

data['bestofones'] = [1 if x>= 8 else 0 for x in data['quality']]
X = data.drop(['quality','bestofones'], axis = 1)
y = data['bestofones']

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=5)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))

confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)