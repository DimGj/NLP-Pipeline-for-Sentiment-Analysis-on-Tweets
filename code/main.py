import os
import numpy
from ossaudiodev import SNDCTL_COPR_RESET
import pickle
import pandas
import nltk
import csv

def LoadTweets():
    Scores = {}
    Target = "tweets.pkl"
    if os.path.getsize(Target) > 0 :
        with open(Target,"rb") as f:
            unpickler = pickle.Unpickler(f)
            Scores = unpickler.load()
            dict((k.lower(), v.lower()) for k,v in Scores.iteritems())
            for i in Scores:
                print(i,Scores[i])

    else:
        print("Error,file not found")
LoadTweets()
