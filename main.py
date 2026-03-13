
import pandas as pd
from src.data_preprocessing import clean_text
from src.feature_engineering import create_features
from src.train import train_model
from src.evaluate import evaluate_model

print("NEWS ARTICLE CLASSIFICATION PIPELINE\n")

train=pd.read_csv("data/raw/train.csv")
test=pd.read_csv("data/raw/test.csv")

train["text"]=train["Title"]+" "+train["Description"]
test["text"]=test["Title"]+" "+test["Description"]

train=train[["text","Class Index"]]
test=test[["text","Class Index"]]

train.columns=["text","label"]
test.columns=["text","label"]

df=pd.concat([train,test])

df["clean_text"]=df["text"].apply(clean_text)

df.to_csv("data/processed/news_clean.csv",index=False)

X,y=create_features("data/processed/news_clean.csv")

model,X_test,y_test=train_model(X,y)

evaluate_model(model,X_test,y_test)
