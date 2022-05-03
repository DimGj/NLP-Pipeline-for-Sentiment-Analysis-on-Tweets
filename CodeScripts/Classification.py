from math import gamma
from re import T
from sklearn import datasets
from VectorizationTweets import TF_IDF
import imports
import FileHandlers

def SVM():
    Train = FileHandlers.OpenFile('../TSVFiles/Train.tsv')
    Test = FileHandlers.OpenFile('../TSVFiles/Test.tsv')
    BOWTrain = FileHandlers.LoadFile('../PickleFiles/bagofwordsTrain.pkl')
    BOWTest = FileHandlers.LoadFile('../PickleFiles/bagofwordsTest.pkl')
    clf = imports.svm.LinearSVC(C=1.0,dual=False)
    clf.fit(BOWTrain,Train['sentiment'])
    ScoresBOW = imports.cross_val_score(clf, BOWTest,Test['sentiment'], scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the Bag Of Words are: ", ScoresBOW)
    TFIDTrain = FileHandlers.LoadFile('../PickleFiles/TF-IDFTrain.pkl')
    TFIDTest = FileHandlers.LoadFile('../PickleFiles/TF-IDFTest.pkl')
    clf.fit(TFIDTrain,Train['sentiment'])
    ScoresTFIDF = imports.cross_val_score(clf, TFIDTest, Test['sentiment'], scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the TF-IDF are: ", ScoresTFIDF)

def SVM2():
    WETrain = FileHandlers.LoadFile('../PickleFiles/WordEmbeddingsTrain.pkl')
    WETest = FileHandlers.LoadFile('../PickleFiles/WordEmbeddingsTest.pkl')
    words = list(WETrain.wv.key_to_index)
    print(words)
    #clf = imports.svm.SVC()
    #clf.fit(List[:1000],Train['sentiment'][:1000])
    #ScoresWE = imports.cross_val_score(clf, WETest[:500], Test['sentiment'][:500], scoring='accuracy',cv=10, n_jobs=-1)
    #print("Scores for the Word Embeddings are: ", ScoresWE)

def RandomForest():
    Train = FileHandlers.OpenFile('../TSVFiles/Train.tsv')
    Test = FileHandlers.OpenFile('../TSVFiles/Test.tsv')
    BOWTrain = FileHandlers.LoadFile('../PickleFiles/bagofwordsTrain.pkl')
    BOWTest = FileHandlers.LoadFile('../PickleFiles/bagofwordsTest.pkl')
    model = imports.RandomForestClassifier(n_estimators=600,min_samples_split=3,n_jobs=-1)
    model.fit(BOWTrain[:40000],Train['sentiment'][:40000])
    ScoresBOW = imports.cross_val_score(model, BOWTest[:20000], Test['sentiment'][:20000],scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the Bag Of Words are: ", ScoresBOW)
    TFIDTrain = FileHandlers.LoadFile('../PickleFiles/TF-IDFTrain.pkl')
    TFIDTest = FileHandlers.LoadFile('../PickleFiles/TF-IDFTest.pkl')
    model.fit(TFIDTrain[:40000],Train['sentiment'][:40000])
    ScoresTFIDF = imports.cross_val_score(model, TFIDTest[:20000], Test['sentiment'][:20000], scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the TF-IDF are: ", ScoresTFIDF)