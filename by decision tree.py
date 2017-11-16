import pandas as pd
import numpy as np
from sklearn import tree

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


train["Age"]=train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"]=="male"]=0
train["Sex"][train["Sex"]=="female"]=1
train["Embarked"]=train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"]=="S"]=0
train["Embarked"][train["Embarked"]=="C"]=1
train["Embarked"][train["Embarked"]=="Q"]=2
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
target=np.array(train["Survived"])
features_one=np.array(train[["Pclass","Age","Sex","Fare","SibSp","Parch", "Embarked"]])                  
my_tree_one=tree.DecisionTreeClassifier(max_depth=10,min_samples_split=5,random_state=1)
my_tree_one=my_tree_one.fit(features_one,target)                  

test["Age"]=test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"]=="male"]=0                  
test["Sex"][test["Sex"]=="female"]=1    
test["Embarked"]=test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"]=="S"]=0
test["Embarked"][test["Embarked"]=="C"]=1
test["Embarked"][test["Embarked"]=="Q"]=2
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
features_test=np.array(test[["Pclass","Age","Sex","Fare","SibSp","Parch", "Embarked"]])                  
my_prediction=my_tree_one.predict(features_test)                  
                  
            
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("C:/Users/dell/Desktop/vash.csv",index_label=["PassengerId"])  


                 
                  