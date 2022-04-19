from ast import Load
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

def DataAnalysis_i():
    Data =  CleanTweets()
    Neg = Data.groupby('sentiment').get_group("NEG")
    Pos = Data.groupby('sentiment').get_group("POS")
    Neu = Data.groupby('sentiment').get_group("NEU")
    DataStr = []
    DataValues = []
    DataStr.append("NEG")
    DataStr.append("POS")
    DataStr.append("NEU")
    DataValues.append(len(Neg))
    DataValues.append(len(Pos))
    DataValues.append(len(Neu))
    ExplodeData = (0.1, 0.0, 0.2)
    Colors = ( "orange", "cyan", "brown")
    Preview = "Distribution of sentiments in the tweets"
    CreatePlot(ExplodeData,Colors,DataValues,DataStr,DataStr,Preview)

def DataAnalysis_ii():
    Data = CleanTweets()
    MostCommon = Counter(" ".join(Data["text"]).split()).most_common(10)
    MostCommonStr = []
    MostCommonValues = []
    MostCommonStr,MostCommonValues = SplitTuple(MostCommon)
    ExplodeData = (0.1, 0.0, 0.2, 0.3, 0.0, 0.0, 0.3,0.6, 0.4 ,0.0)
    Colors = ( "orange", "cyan", "brown",
          "grey", "indigo", "beige","black","red","pink","blue")
    Preview = "Most common words in the DataFrame"
    CreatePlot(ExplodeData,Colors,MostCommonValues,MostCommonStr,MostCommonStr,Preview)

def DataAnalysis_iii():
    Data = CleanTweets()
    MostCommonStr = []
    MostCommonValues = [] 
    Neg = Data.groupby('sentiment').get_group("NEG")
    Pos = Data.groupby('sentiment').get_group("POS")
    Neu = Data.groupby('sentiment').get_group("NEU")
    MostCommonNeg = Counter(" ".join(Neg["text"]).split()).most_common(3)
    MostCommonPos = Counter(" ".join(Pos["text"]).split()).most_common(3)
    MostCommonNeu = Counter(" ".join(Neu["text"]).split()).most_common(3)
    MostCommonNegStr,MostCommonNegValues = SplitTuple(MostCommonNeg)
    MostCommonPosStr,MostCommonPosValues = SplitTuple(MostCommonPos)
    MostCommonNeuStr,MostCommonNeuValues = SplitTuple(MostCommonNeu)
    for items in MostCommonNegStr:
        MostCommonStr.append(items)
    for items in MostCommonNegValues:
        MostCommonValues.append(items)
    for items in MostCommonPosStr:
        MostCommonStr.append(items)
    for items in MostCommonPosValues:
        MostCommonValues.append(items)
    for items in MostCommonNeuStr:
        MostCommonStr.append(items)
    for items in MostCommonNeuValues:
        MostCommonValues.append(items)
    SentimentArray = ["NEG is orange","POS is cyan","NEU is brown"]
    ExplodeData = (0.1,0.1,0.1, 0.0,0.0,0.0, 0.2,0.2,0.2)
    Colors = ( "orange","orange","orange", "cyan","cyan","cyan","brown","brown","brown")
    Preview = "Most common words in the DataFrame by sentiment"
    CreatePlot(ExplodeData,Colors,MostCommonValues,MostCommonStr,SentimentArray,Preview)

def DataAnalysis_iv():
    Data = CleanTweets()
    SentimentValues = []
    SentimentStr = []
    Astra = Data.loc[Data["text"].str.contains("astrazeneca",case=True)]
    Pfizer = Data.loc[Data["text"].str.contains(pat = "astrazeneca|pfizer|biontech",case=True)]
    AstraNeg = Astra.groupby('sentiment').get_group("NEG")
    AstraPos = Astra.groupby('sentiment').get_group("POS")
    AstraNeu = Astra.groupby('sentiment').get_group("NEU")
    SentimentValues.append(len(AstraNeg))
    SentimentValues.append(len(AstraNeu))
    SentimentValues.append(len(AstraPos))
    SentimentStr.append("Astra NEG")
    SentimentStr.append("Astra NEU")
    SentimentStr.append("Astra POS")
    PfizerNeg = Pfizer.groupby('sentiment').get_group("NEG")
    PfizerPos = Pfizer.groupby('sentiment').get_group("POS")
    PfizerNeu = Pfizer.groupby('sentiment').get_group("NEU")
    SentimentValues.append(len(PfizerNeg))
    SentimentValues.append(len(PfizerNeu))
    SentimentValues.append(len(PfizerPos))
    SentimentStr.append("Pfizer NEG")
    SentimentStr.append("Pfizer NEU")
    SentimentStr.append("Pfizer POS")
    Colors = ("orange","orange","orange","yellow","yellow","yellow")
    ExplodeData = (0.1,0.1,0.1,0.2,0.2,0.2)
    SentimentArray = ["Astra is orange","Pfizer etc. is yellow"]
    Preview = "Sentiment Comparison between Astrazeneca tweets and Pfizer etc"
    CreatePlot(ExplodeData,Colors,SentimentValues,SentimentStr,SentimentArray,Preview)

def DataAnalysis_v():
    Data = CleanTweets()
    Data['month'] = pd.DatetimeIndex(Data['date']).month
    ExplodeData = (0.1, 0.0, 0.2, 0.3, 0.0, 0.0, 0.1,0.2, 0.2 ,0.0,0.2,0.3)
    Colors = ( "orange", "cyan", "brown",
          "grey", "indigo", "beige","purple","red","pink","blue","yellow","green")
    Preview = "Tweets per month"
    Date = [dict() for i in range(12)]
    DateStr = ["Jan","Feb","March","April","May","June","July","Aug","Sept","Oct","Nov","Dec"]
    DateValues = []
    for i in range(1,13):
        Date[i - 1] = Data.groupby('month').get_group(i)
        DateValues.append(len(Date[i - 1]))
    CreatePlot(ExplodeData,Colors,DateValues,DateStr,DateStr,Preview)
    AverageTweets = int(len(Data)/12)
    ImportantMonths = []
    for i in range(1,13):
        if DateValues[i - 1] > AverageTweets:
            ImportantMonths.append(i)

def CreatePlot(ExplodeData,Colors,Values,StrArray,LegendPreview,LegendTitle):
    Fig,Ax = plt.subplots(figsize=(10,7))
    WedgeProperties = { 'linewidth' : 1, 'edgecolor' : "green" }
    wedges,texts,autotexts = Ax.pie(Values,autopct=lambda pct: autocpt(pct, Values),
                                    explode=ExplodeData,labels=StrArray,shadow=True,
                                    colors=Colors,startangle=90,wedgeprops=WedgeProperties,
                                    textprops=dict(color = "black"))
    Ax.legend(wedges,LegendPreview,title = "Legend",loc = "center left",
              bbox_to_anchor = (1,0,0.5,1))
    plt.setp(autotexts,size = 7,weight = "bold")
    Ax.set_title(LegendTitle)
    plt.show()

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

def PrintDataFrame():
    Data = CleanTweets()
    print(Data)

#PrintDataFrame()
DataAnalysis_v()
#SplitDataFrame()