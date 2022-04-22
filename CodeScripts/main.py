import imports
import FileHandlers
import VectorizationTweets
import DataAnalysis

def CleanTweets():

    Data = FileHandlers.LoadFile("../PickleFiles/tweets.pkl")
    Data = Data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    Data['text'] = Data['text'].str.lower()
    stop  = imports.stopwords.words('english')
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    Data['text'] = Data['text'].str.replace(pat, '',regex = True)
    Data['text'] = Data['text'].str.replace(r'\s+', ' ',regex = True)
    cleaned = []
    elements = list(Data['text'])
    for i in elements:
        FlteredText = imports.re.sub('https?://[A-Za-z0-9./]+','',i)
        FlteredText = imports.re.sub("[^a-zA-Z0-9]", " ",FlteredText)
        cleaned.append(imports.re.sub(r'^RT[\s]+', '', FlteredText))
    Data['text'] = cleaned
    return Data

def PrintDataFrame():
    print(Data)

Data = CleanTweets()

FileHandlers.SplitDataFrame(Data)
VectorizationTweets.TF_IDF()
VectorizationTweets.BagOfWords()
#PrintDataFrame()
#DataAnalysis.DataAnalysis_i(Data)
#DataAnalysis.DataAnalysis_ii(Data)
#DataAnalysis.DataAnalysis_iii(Data)
#DataAnalysis.DataAnalysis_iv(Data)
#DataAnalysis.DataAnalysis_v(Data)