import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

districtRainfall = pd.read_csv('district wise rainfall normal.csv')
rainfall = pd.read_csv('rainfall in india 1901-2015.csv')
Temp = pd.read_csv('Chennai_1990_2022_Madras.csv')
recomandation = pd.read_csv('Crop_recommendation.csv')
#agri = pd.read_csv('all.csv')

x = Temp.drop(columns=['time'])
y = Temp['time']
X_train, X_test, Y_train, y_test = train_test_split(x,y,test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x,y)
predictions=model.predict(X_test )

score=accuracy_score(y_test, predictions)
print(score)