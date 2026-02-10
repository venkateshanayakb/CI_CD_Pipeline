import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

#Load data
df = pd.read_csv(r"G:\IIHMR\Notes\Vinay Sir\MLOps_Pipeline\data\data.csv")

X = df[["area", "bedrooms"]]
y = df["price"]

#Train the model 
model= LinearRegression()
model.fit(X,y)

#Save the model
with open(r"G:\IIHMR\Notes\Vinay Sir\MLOps_Pipeline\backend\models\model.pkl", "wb") as f:
    pickle.dump(model,f)