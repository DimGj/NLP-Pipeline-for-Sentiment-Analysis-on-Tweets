import os
import numpy as np
import pickle
import pandas as pd
import nltk
import csv

def LoadTweets():
    Data = {}
    Target = "tweets.pkl"
    if os.path.getsize(Target) > 0 :
        with open(Target,"rb") as f:
            unpickler = pickle.Unpickler(f)
            Data = unpickler.load()
           # print(Data)
            df  = pd.DataFrame(Data)
          #  df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
            User_Description = df['user_description'].map(lambda x: x.lower() if isinstance(x,str) else x)
            print(User_Description)
    else:
        print("Error,file not found")
LoadTweets()
