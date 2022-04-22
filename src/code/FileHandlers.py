import imports

def LoadFile(Filename):
    Data = {}
    if imports.exists(Filename):
        with open(Filename,'rb') as f:
            unpickler = imports.pickle.Unpickler(f)
            Data = unpickler.load()
            df = imports.pd.DataFrame(Data)
            f.close()
            return df
    else:
        print("Error file,with directory: ",Filename," not found!")

def SaveFile(DataFrame,Filename):
    if not imports.exists(Filename):
        with open(Filename,'wb') as handle:
            imports.pickle.dump(DataFrame,handle,protocol=imports.pickle.HIGHEST_PROTOCOL)

def OpenFile(Filename):
    if imports.exists(Filename):
        Data = imports.pd.read_table(Filename)
        return Data
    else:
        print("Requested File does not exist!")

def SplitDataFrame(DataFrame):
  Train,Test = imports.sklearn.model_selection.train_test_split(DataFrame,test_size=0.2, random_state=42, shuffle=True)
  del Test['sentiment']

  train_path = "../TSVFiles/Train.tsv"
  test_path = "../TSVFiles/Test.tsv"
  if not imports.exists(train_path):
    Train.to_csv(train_path, sep='\t', index=False)
  else:
      print("Train.tsv file already exists!")
  if not imports.exists(test_path):
    Test.to_csv(test_path, sep='\t', index=False)
  else:
      print("Test.tsv file already exists!")

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