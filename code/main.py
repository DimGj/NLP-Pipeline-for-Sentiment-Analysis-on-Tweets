from ast import Load
from email import header
from fileinput import filename
from genericpath import exists
from operator import pos, truediv
from pathlib import Path
import sys
import os
from tkinter import W
from turtle import color, title
import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import csv
import sklearn.model_selection
from collections import Counter
from matplotlib import pyplot as plt
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer

import data_analysis
import VectorizationTweets

def LoadTweets():
    Data = {}
    Target = "../tweets/tweets.pkl"
    if exists(Target):
    	with open(Target,"rb") as f:
            unpickler = pickle.Unpickler(f)
            Data = unpickler.load()
            df  = pd.DataFrame(Data)
            f.close()
            return df
    else:
        print("Error ,tweets.pkl not found!")
        exit(1)

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

Data = CleanTweets()

def SplitDataFrame():
  Train,Test = sklearn.model_selection.train_test_split(Data,test_size=0.2, random_state=42, shuffle=True)
  del Test['sentiment']

  train_path = Path(os.getcwd(),'Train.tsv')
  test_path = Path(os.getcwd(),'Test.tsv')
  if not exists(train_path):
    Train.to_csv(train_path, sep='\t', index=False)
  if not exists(test_path):
    Test.to_csv(test_path, sep='\t', index=False)

def SaveFile(DataFrame,Filename):
    if not exists(Filename):
        with open(Filename,'wb') as handle:
            pickle.dump(DataFrame,handle,protocol=pickle.HIGHEST_PROTOCOL)

def OpenFile(Filename):
    if exists(Filename):
        Data = pd.read_table(Filename)
        return Data
    else:
        print("Requested File does not exist!")
        exit(1)

def SplitTuple(TupleArray):
    TupleStr = []
    TupleValues = []
    for items in TupleArray:
        for moreitems in items:
            if isinstance(moreitems,str):
                TupleStr.append(moreitems)
            else:
                TupleValues.append(moreitems)
    return TupleStr,TupleValues

def PrintDataFrame():
    for items in Data:
        print(items)

VectorizationTweets.TF_IDF()
#BagOfWords()
#PrintDataFrame()
#DataAnalysis_v()
#SplitDataFrame()
