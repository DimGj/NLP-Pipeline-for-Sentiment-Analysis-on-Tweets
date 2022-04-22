import imports
import FileHandlers

def BagOfWords():
    from sklearn.feature_extraction.text import CountVectorizer
    Train = FileHandlers.OpenFile("../TSVFiles/Train.tsv")
    Test = FileHandlers.OpenFile("../TSVFiles/Test.tsv")
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=1000,stop_words='english')
    TrainVector = vectorizer.fit_transform(Train['text'])
    TestVector = vectorizer.transform(Test['text'])
    dfTrain_bow_sklearn = imports.pd.DataFrame(TrainVector.toarray(),columns=vectorizer.get_feature_names_out())
    dfTest_bow_sklearn = imports.pd.DataFrame(TestVector.toarray(),columns=vectorizer.get_feature_names_out())
    FileHandlers.SaveFile(dfTrain_bow_sklearn,'../PickleFiles/bagofwordsTrain.pkl')
    FileHandlers.SaveFile(dfTest_bow_sklearn,'../PickleFiles/bagofwordsTest.pkl')

def TF_IDF():
    from sklearn.feature_extraction.text import TfidfVectorizer
    Train = FileHandlers.OpenFile("../TSVFiles/Train.tsv")
    Test = FileHandlers.OpenFile("../TSVFiles/Test.tsv")
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000,stop_words='english')
    tfidfTrain_vector = tfidf_vectorizer.fit_transform(Train['text'])
    tfidfTest_vector = tfidf_vectorizer.transform(Test['text'])
    tfidfTrain_df = imports.pd.DataFrame(tfidfTrain_vector.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
    tfidfTest_df = imports.pd.DataFrame(tfidfTest_vector.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
    FileHandlers.SaveFile(tfidfTrain_df,'../PickleFiles/TF-IDFTrain.pkl')
    FileHandlers.SaveFile(tfidfTest_df,'../PickleFiles/TF-IDFTest.pkl')