from ast import Load
from genericpath import exists
from operator import truediv
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
  del Test['sentiment']

  train_path = Path(os.getcwd(),'Train.tsv')
  test_path = Path(os.getcwd(),'Test.tsv')

  Train.to_csv(train_path, sep='\t', index=False)
  Test.to_csv(test_path, sep='\t', index=False)

def DataAnalysis():
    Data = CleanTweets()
    MostCommon = Counter(" ".join(Data["text"]).split()).most_common(10)
    ExplodeData = (0.1, 0.0, 0.2, 0.3, 0.0, 0.0, 0.3,0.6, 0.4 ,0.0)
    Colors = ( "orange", "cyan", "brown",
          "grey", "indigo", "beige","black","red","pink","blue")
    WedgeProperties = { 'linewidth' : 1, 'edgecolor' : "green" }
    MostCommonStr = []
    MostCommonValues = []
    MostCommonStr,MostCommonValues = SplitTuple(MostCommon)
    Fig,Ax = plt.subplots(figsize=(10,7))
    wedges,texts,autotexts = Ax.pie(MostCommonValues,autopct=lambda pct: autocpt(pct, MostCommonValues),
                                    explode=ExplodeData,labels=MostCommonStr,shadow=True,
                                    colors=Colors,startangle=90,wedgeprops=WedgeProperties,
                                    textprops=dict(color = "magenta"))
    Ax.legend(wedges,MostCommonStr,title = "Legend",loc = "center left",
              bbox_to_anchor = (1,0,0.5,1))
    plt.setp(autotexts,size = 7,weight = "bold")
    Ax.set_title("Most common words in the DataFrame")
    plt.show()
    Neg = Data.groupby('sentiment').get_group("NEG")
    Pos = Data.groupby('sentiment').get_group("POS")
    Neu = Data.groupby('sentiment').get_group("NEU")
    MostCommonNeg = Counter(" ".join(Neg["text"]).split()).most_common(3)
    MostCommonPos = Counter(" ".join(Pos["text"]).split()).most_common(3)
    MostCommonNeu = Counter(" ".join(Neu["text"]).split()).most_common(3)

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

def autocpt(pct,allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)




DataAnalysis()
#SplitDataFrame()
