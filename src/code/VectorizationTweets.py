import imports
import FileHandlers

def BagOfWords(Data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=1000,stop_words='english')
    vector = vectorizer.fit_transform(Data['text'])
    df_bow_sklearn = imports.pd.DataFrame(vector.toarray(),columns=vectorizer.get_feature_names_out())
    FileHandlers.SaveFile(df_bow_sklearn,'../PickleFiles/bagofwords.pkl')

def TF_IDF(Data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=1000,stop_words='english')
    tfidf_vector = tfidf_vectorizer.fit_transform(Data['text'])
    tfidf_df = imports.pd.DataFrame(tfidf_vector.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
    FileHandlers.SaveFile(tfidf_df,'../PickleFiles/TF-IDF.pkl')