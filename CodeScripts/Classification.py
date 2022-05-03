from sklearn import datasets
from VectorizationTweets import TF_IDF
import imports
import FileHandlers

def SVM():
    Train = FileHandlers.OpenFile('../TSVFiles/Train.tsv')
    Test = FileHandlers.OpenFile('../TSVFiles/Test.tsv')
    BOWTrain = FileHandlers.LoadFile('../PickleFiles/bagofwordsTrain.pkl')
    BOWTest = FileHandlers.LoadFile('../PickleFiles/bagofwordsTest.pkl')
    clf = imports.svm.LinearSVC()
    clf.fit(BOWTrain[:20000],Train['sentiment'][:20000])
    ScoresBOW = imports.cross_val_score(clf, BOWTest[:10000], Test['sentiment'][:10000], scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the Bag Of Words are: ", ScoresBOW)
    TFIDTrain = FileHandlers.LoadFile('../PickleFiles/TF-IDFTrain.pkl')
    TFIDTest = FileHandlers.LoadFile('../PickleFiles/TF-IDFTest.pkl')
    clf.fit(TFIDTrain[:20000],Train['sentiment'][:20000])
    ScoresTFIDF = imports.cross_val_score(clf, TFIDTest[:10000], Test['sentiment'][:10000], scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the TF-IDF are: ", ScoresTFIDF)

def SVM2():
    WETrain = FileHandlers.LoadFile('../PickleFiles/WordEmbeddingsTrain.pkl')
    WETest = FileHandlers.LoadFile('../PickleFiles/WordEmbeddingsTest.pkl')
    print(WETrain.wv)
    #clf = imports.svm.SVC()
    #clf.fit(List[:1000],Train['sentiment'][:1000])
    #ScoresWE = imports.cross_val_score(clf, WETest[:500], Test['sentiment'][:500], scoring='accuracy',cv=10, n_jobs=-1)
    #print("Scores for the Word Embeddings are: ", ScoresWE)

def RandomForest():
    Train = FileHandlers.OpenFile('../TSVFiles/Train.tsv')
    Test = FileHandlers.OpenFile('../TSVFiles/Test.tsv')
    BOWTrain = FileHandlers.LoadFile('../PickleFiles/bagofwordsTrain.pkl')
    BOWTest = FileHandlers.LoadFile('../PickleFiles/bagofwordsTest.pkl')
    model = imports.RandomForestClassifier(n_estimators=600,random_state=1,n_jobs=-1)
    model.fit(BOWTrain[:40000],Train['sentiment'][:40000])
    ScoresBOW = imports.cross_val_score(model, BOWTest[:20000], Test['sentiment'][:20000],scoring='accuracy',cv=10, n_jobs=-1)
    print(ScoresBOW)
    TFIDTrain = FileHandlers.LoadFile('../PickleFiles/TF-IDFTrain.pkl')
    TFIDTest = FileHandlers.LoadFile('../PickleFiles/TF-IDFTest.pkl')
    model.fit(TFIDTrain[:40000],Train['sentiment'][:40000])
    ScoresTFIDF = imports.cross_val_score(model, TFIDTest[:20000], Test['sentiment'][:20000], scoring='accuracy',cv=10, n_jobs=-1)
    print("Scores for the TF-IDF are: ", ScoresTFIDF)