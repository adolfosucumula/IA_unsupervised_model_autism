import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# MY CLASSES 
from read_database import ReadDatabase as DB
db = DB().getDatabase("/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff");

df = pd.read_csv(db)

#df = df.iloc[:,-0]

df.columns = df.columns.str.replace("/",'_')
df.rename(columns={"austim": "hasAUTISM"}, inplace=True)
df.rename(columns={"jundice": "jaundice"}, inplace=True)
df.rename(columns={"relation": "who_is_talking"}, inplace=True)

df.loc[df.age=='?', 'age'] = 0
df.age = df.age.astype(int)

print("____________________DATASETS__________________")
print('')
print(df)
print("____________________DATASETS SHAPED__________________")
print('')
print(df.shape)
print("____________________DATASETS COLUMNS__________________")
print('')
print(df.columns)
#print("____________________DATASETS HEAD__________________")
print('')
#print(df.head())
print("____________________DATASETS DESCRIBED__________________")
print('')
#print(df.describe())
print("____________________DATASETS CORR__________________")
print('')
#print(df.corr())
print('')

df = df.drop_duplicates()
#print(df.shape)
#print(df.isnull().sum())
df = df.dropna()
#print(df.isnull().sum())


gender = {'m': 1, 'f': 0, '?': -1}
df.gender = [gender[item] for item in df.gender]

#print(df.jaundice)
bornwithjaundice = {'yes': 1, 'no': 0}
df.jaundice = [bornwithjaundice[item] for item in df.jaundice]

familymemberwithpdd = {'YES': 1, 'NO': 0}
df.Class_ASD = [familymemberwithpdd[item] for item in df.Class_ASD]
#print(df.Class_ASD)

useda_pp_before = {'yes': 1,'no': 0}
df.used_app_before = [useda_pp_before[item] for item in df.used_app_before]

whos_is_talking = {'Self': 1, 'Parent': 2, "'Health care professional'": 3, 'Relative': 4, 'Others': 5, '?': 6}
df.who_is_talking = [whos_is_talking[item] for item in df.who_is_talking]

#age = {'?': 0}
#df.age = [age[item] for item in df.age]

hasAUTISM = {'yes': 1,'no': 0}
df.hasAUTISM = [hasAUTISM[item] for item in df.hasAUTISM]

df = df[['age', 'gender', 'jaundice', 'Class_ASD', 'used_app_before', 'who_is_talking', 
      'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 
      'A8_Score', 'A9_Score', 'A10_Score', 'hasAUTISM']]

#for i in range(0,df.shape[0]):
    #if(df.iloc[i,-1] > 5):
        #df.iloc[i-1] = 1
    #else:
        #df.iloc[i,-1] = 0

features = df.iloc[:,:-1]
target = df.iloc[:,-1]

#print(features)


rf = RandomForestClassifier()
grid_space = {'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,500,1000],
              'min_samples_split':[1,2,3,5],
              'min_samples_leaf':[1,2,3,5]}


grid = GridSearchCV(rf, param_grid = grid_space, cv=3, scoring='accuracy')
model_grid = grid.fit(features, target)



