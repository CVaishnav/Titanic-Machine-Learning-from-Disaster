import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC

train_url="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train=pd.read_csv(train_url)

test_url="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test=pd.read_csv(test_url)

train.drop(['PassengerId','Name','Ticket'],axis=1)
test.drop(['Name','Ticket'],axis=1)

train["Embarked"]=train["Embarked"].fillna("S")

embark_dummies_train=pd.get_dummies(train["Embarked"])
embark_dummies_train.drop(['S'],axis=1,inplace=True)

embark_dummies_test=pd.get_dummies(test["Embarked"])
embark_dummies_test.drop(['S'],axis=1,inplace=True)

train=train.join(embark_dummies_train)
test=test.join(embark_dummies_test)

train.drop(['Embarked'],axis=1,inplace=True)
test.drop(['Embarked'],axis=1,inplace=True)

test["Fare"].fillna(test["Fare"].median(),inplace=True)


train["Fare"]=train["Fare"].astype(int)
test["Fare"]=test["Fare"].astype(int)

avg_age_train=train["Age"].mean()
std_age_train=train["Age"].std()
count_nan_age_train=train["Age"].isnull().sum()

avg_age_test=test["Age"].mean()
std_age_test=test["Age"].std()
count_nan_age_test=test["Age"].isnull().sum()


rand1=np.random.randint(avg_age_train-std_age_train,avg_age_train+std_age_train,size=count_nan_age_train)
rand2=np.random.randint(avg_age_test-std_age_test,avg_age_test+std_age_test,size=count_nan_age_test)

train["Age"][np.isnan(train["Age"])]=rand1
test["Age"][np.isnan(test["Age"])]=rand2

train['Age'] = train['Age'].astype(int)
test['Age']= test['Age'].astype(int)


train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train=train.drop(['SibSp','Parch'],axis=1)
test=test.drop(['SibSp','Parch'],axis=1)


def get_person(passenger):
    age,sex=passenger
    return 'child' if age<16 else sex


train['Person']=train[["Age","Sex"]].apply(get_person,axis=1)
test['Person']=test[["Age","Sex"]].apply(get_person,axis=1)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

dummies_train_person=pd.get_dummies(train["Person"])
dummies_train_person.columns = ['Child','Female','Male']
dummies_train_person.drop(['Male'], axis=1, inplace=True)

dummies_test_person=pd.get_dummies(test["Person"])
dummies_test_person.columns = ['Child','Female','Male']
dummies_test_person.drop(['Male'], axis=1, inplace=True)

train= train.join(dummies_train_person)
test= test.join(dummies_test_person)

train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)


pclass_dummies_train  = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)

train = train.join(pclass_dummies_train)
test= test.join(pclass_dummies_test)


X_train = train[["Class_1","Class_2", "Age","Child","Female" ,"Fare", "Family","C","Q"]].values

#X_train = train.drop(["Survived"],axis=1)

Y_train = train["Survived"]
X_test = test[["Class_1","Class_2", "Age","Child","Female" ,"Fare", "Family","C","Q"]].values

#random_forest = RandomForestClassifier(n_estimators=100)
#random_forest.fit(X_train, Y_train)
#prediction = random_forest.predict(X_test)




model=LogisticRegression()
model=model.fit(X_train,Y_train)
prediction=model.predict(X_test)

train=train.drop(['Survived','Name','Ticket','Cabin'],axis=1)

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(model.coef_[0])
print(coeff_df)


solution=pd.DataFrame({
     'PassengerId':test["PassengerId"],
     'Survived':prediction
 })

solution.to_csv("C:/Users/dell/Desktop/test.csv",index=False)














