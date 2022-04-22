import imports

def LoadFile(Filename):
    Data = {}
    if imports.exists(Filename):
        Data = imports.pd.read_pickle(Filename)
        return(Data)
    else:
        print("Directory not found!")

def SaveFile(DataFrame,Filename):
    if not imports.exists(Filename):
        DataFrame.to_pickle(Filename)

def OpenFile(Filename):
    if imports.exists(Filename):
        Data = imports.pd.read_csv(Filename,sep='\t',lineterminator='\n')
        return Data
    else:
        print("Requested File does not exist!")

def SplitDataFrame(Data):
  Train, Test = imports.sklearn.model_selection.train_test_split(Data,test_size=0.2,random_state=42, shuffle=True)
  del Test['sentiment']
  train_path = "../TSVFiles/Train.tsv"
  test_path = "../TSVFiles/Test.tsv"
  Train.to_csv(train_path,sep='\t')
  Test.to_csv(test_path, sep='\t')

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
