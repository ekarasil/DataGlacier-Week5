import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("Cab_Data.csv")

def encoder(x):
  if x=="Pink Cab":
    return(0)
  else:
    return(1)

df["Company"]=df["Company"].apply(encoder)
print(df.head())

# Select independent and dependent variable
X = df[["KM Travelled", "Cost of Trip","Company"]]
y = df["Price Charged"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

# Instantiate the model
classifier = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 7, random_state = 18)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))