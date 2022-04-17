from genericpath import exists
from operator import truediv
import sys
import os
import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import csv

def LoadTweets():
    Data = {}
    Target = "tweets.pkl"
    if exists(Target):
    	with open(Target,"rb") as f:
            unpickler = pickle.Unpickler(f)
            Data = unpickler.load()
            df  = pd.DataFrame(Data)
            return df
    else:
      print("Error ,tweets.pkl not found!")
      exit()

def CleanTweets():

  Data = LoadTweets()
  #Data["user_description"] = Data['user_description'].transform(lambda x: x.lower() if isinstance(x,str) else x)
 # Data.transform(lambda x: x.lower() if isinstance(x,str) else x)
  Data = Data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
  for columns in Data.columns:
    Data[columns] = Data[columns].str.lower()
  #tokenizer = nltk.RegexpTokenizer(r"\w+")
  #User_Description = tokenizer.tokenize(User_Description)
  print(Data)


CleanTweets()
