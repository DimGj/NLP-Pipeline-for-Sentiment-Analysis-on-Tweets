from genericpath import exists
from operator import truediv
from pathlib import Path
import sys
import os
import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import csv
import sklearn.model_selection

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
    Data = Data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    Data['text'] = Data['text'].str.lower()
    stop  = stopwords.words('english')
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    Data['text'] = Data['text'].str.replace(pat, '',regex = True)
    Data['text'] = Data['text'].str.replace(r'\s+', ' ',regex = True)
    cleaned = []
    elements = list(Data['text'])
    for i in elements:
        FlteredText = re.sub('https?://[A-Za-z0-9./]+','',i)
        FlteredText = re.sub("[^a-zA-Z0-9]", " ",FlteredText)
        cleaned.append(re.sub(r'^RT[\s]+', '', FlteredText))
    Data['text'] = cleaned
    #print(Data['text'][2])
    return Data

def SplitDataFrame():
  Data = CleanTweets()
  Train,Test = sklearn.model_selection.train_test_split(Data,test_size=0.2, random_state=42, shuffle=True)

  train_path = Path(os.getcwd(),'Train.tsv')
  test_path = Path(os.getcwd(),'Test.tsv')

  Train.to_csv(train_path, sep='\t', index=False)
  Test.to_csv(test_path, sep='\t', index=False)


SplitDataFrame()
