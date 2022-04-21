import main

def DataAnalysis_i():
    Data =  main.CleanTweets()
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
    Colors = ( "brown", "cyan", "orange")
    Preview = "Distribution of sentiments in the tweets"
    CreatePlot(ExplodeData,Colors,DataValues,DataStr,DataStr,Preview)

def DataAnalysis_ii():
    Data = main.CleanTweets()
    MostCommon = main.Counter(" ".join(Data["text"]).split()).most_common(10)
    MostCommonStr = []
    MostCommonValues = []
    MostCommonStr,MostCommonValues = main.SplitTuple(MostCommon)
    ExplodeData = (0.1, 0.15, 0.2, 0.3, 0.05, 0.25, 0.3,0.4, 0.4 ,0.0)
    Colors = ( "orange", "cyan", "brown",
          "grey", "indigo", "beige","yellow","red","pink","blue")
    Preview = "Most common words in the DataFrame"
    CreatePlot(ExplodeData,Colors,MostCommonValues,MostCommonStr,MostCommonStr,Preview)

def DataAnalysis_iii():
    Data = main.CleanTweets()
    MostCommonStr = []
    MostCommonValues = [] 
    Neg = Data.groupby('sentiment').get_group("NEG")
    Pos = Data.groupby('sentiment').get_group("POS")
    Neu = Data.groupby('sentiment').get_group("NEU")
    MostCommonNeg = main.Counter(" ".join(Neg["text"]).split()).most_common(3)
    MostCommonPos = main.Counter(" ".join(Pos["text"]).split()).most_common(3)
    MostCommonNeu = main.Counter(" ".join(Neu["text"]).split()).most_common(3)
    MostCommonNegStr,MostCommonNegValues = main.SplitTuple(MostCommonNeg)
    MostCommonPosStr,MostCommonPosValues = main.SplitTuple(MostCommonPos)
    MostCommonNeuStr,MostCommonNeuValues = main.SplitTuple(MostCommonNeu)
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
    SentimentArray = ["NEG is brown","POS is cyan","NEU is orange"]
    ExplodeData = (0.1,0.1,0.1, 0.0,0.0,0.0, 0.2,0.2,0.2)
    Colors = ( "brown","brown","brown","cyan","cyan","cyan","orange","orange","orange",)
    Preview = "Most common words in the DataFrame by sentiment"
    CreatePlot(ExplodeData,Colors,MostCommonValues,MostCommonStr,SentimentArray,Preview)    #legend not implemented right!

def DataAnalysis_iv():
    Data = main.CleanTweets()
    SentimentValues = []
    SentimentStr = []
    Astra = Data.loc[Data["text"].str.contains("astrazeneca",case=True)]
    Pfizer = Data.loc[Data["text"].str.contains(pat = "astrazeneca|pfizer|biontech",case=True)]
    AstraNeg = Astra.groupby('sentiment').get_group("NEG")
    AstraPos = Astra.groupby('sentiment').get_group("POS")
    AstraNeu = Astra.groupby('sentiment').get_group("NEU")
    PfizerNeg = Pfizer.groupby('sentiment').get_group("NEG")
    PfizerPos = Pfizer.groupby('sentiment').get_group("POS")
    PfizerNeu = Pfizer.groupby('sentiment').get_group("NEU")
    SentimentValues.append(len(AstraNeg))
    SentimentValues.append(len(PfizerNeg))
    SentimentValues.append(len(AstraNeu))
    SentimentValues.append(len(PfizerNeu))
    SentimentValues.append(len(AstraPos))
    SentimentValues.append(len(PfizerPos))
    SentimentStr.append("Astra NEG")
    SentimentStr.append("Pfizer NEG")
    SentimentStr.append("Astra NEU")
    SentimentStr.append("Pfizer NEU")
    SentimentStr.append("Astra POS")
    SentimentStr.append("Pfizer POS")
    #Colors = ("orange","orange","orange","yellow","yellow","yellow")
    Colors = ('orange','yellow','orange','yellow','orange','yellow')
    ExplodeData = (0.1,0.1,0.1,0.2,0.2,0.2)
    SentimentArray = ["Astra is orange","Pfizer etc. is yellow"]
    Preview = "Sentiment Comparison between Astrazeneca tweets and Pfizer etc"
    CreatePlot(ExplodeData,Colors,SentimentValues,SentimentStr,SentimentArray,Preview)

def DataAnalysis_v():
    Data = main.CleanTweets()
    Data['month'] = main.pd.DatetimeIndex(Data['date']).month
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
    Fig,Ax = main.plt.subplots(figsize=(10,7))
    WedgeProperties = { 'linewidth' : 1, 'edgecolor' : "green" }
    wedges,texts,autotexts = Ax.pie(Values,autopct=lambda pct: autocpt(pct, Values),
                                    explode=ExplodeData,labels=StrArray,shadow=True,
                                    colors=Colors,startangle=90,wedgeprops=WedgeProperties,
                                    textprops=dict(color = "black"))
    Ax.legend(wedges,LegendPreview,title = "Legend",loc = "center left",
              bbox_to_anchor = (1,0,0.5,1))
    main.plt.setp(autotexts,size = 7,weight = "bold")
    Ax.set_title(LegendTitle)
    main.plt.show()

def autocpt(pct,allvalues):
    absolute = int(pct / 100.*sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

DataAnalysis_v()
