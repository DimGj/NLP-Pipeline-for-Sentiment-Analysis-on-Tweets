from matplotlib.pyplot import clf
from VectorizationTweets import TF_IDF
import imports
import FileHandlers

def SVM():
    BOWTrain = FileHandlers.LoadFile('../PickleFiles/bagofwordsTrain.pkl')
    BOWTest = FileHandlers.LoadFile('../PickleFiles/bagofwordsTest.pkl')
   # TF_IDF = FileHandlers.LoadFile('../PickleFiles/TF-IDFTrain.pkl')
   # WE = FileHandlers.LoadFile('../PickleFiles/WordEmbeddingsTrain.pkl')
    clf = imports.svm.SVC(gamma=0.001,C=100.)
    print(BOWTrain)
   # clf.fit(BOWTrain,BowTrainSen)
   # y_pred = clf.predict(BOWTest)
   # accuracy = imports.accuracy_score(BOWTest,y_pred)*100
   # print(accuracy)
