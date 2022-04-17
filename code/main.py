from genericpath import exists
from operator import truediv
import sys
import os
import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import csv

def LoadTweets():
    Data = {}
    Target = "tweets.pkl"
    if exists(Target):
    	with open(Target,"rb") as f:
            unpickler = pickle.Unpickler(f)
            Data = unpickler.load()
            df  = pd.DataFrame(Data)
            f.close()
            return df
    else:
        print("Error ,tweets.pkl not found!")
        exit()

def CleanTweets():

    Data = LoadTweets()
    #Data["user_description"] = Data['user_description'].transform(lambda x: x.lower() if isinstance(x,str) else x)
    #Data.transform(lambda x: x.lower() if isinstance(x,str) else x)
    Data = Data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    Data['text'] = Data['text'].str.lower()
    #tokenizer = nltk.RegexpTokenizer(r"\w+")
    #User_Description = tokenizer.tokenize(User_Description)
    #re.sub(combined_pat, '', Data)
    cleaned = []
    elements = list(Data['text'])
    for i in elements:
        cleanedText = re.sub("[^a-zA-Z0-9]", " ",i)
        cleaned.append(re.sub(r'^RT[\s]+', '', cleanedText))
    Data['text'] = cleaned
    print(Data['text'])


CleanTweets()
