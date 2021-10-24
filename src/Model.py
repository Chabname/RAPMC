from nltk.corpus import stopwords
import pandas as pd
import re


class Model:
    model_path="results/"

    def __init__(self, data_file):
        self.data_file = data_file

    def pre_processing(data):
        sw = stopwords.words("english")
        # lowercase text
        data = data.apply(lambda x: " ".join(i.lower() for i in  str(x).split()))
#         # remove numeric values
#         data = data.str.replace("\d","")
#         # remove punctuations
#         data = data.str.replace("[^\w\s]","")
        # remove stopwords: the,a,an etc.
        data = data.apply(lambda x: " ".join(i for i in x.split() if i not in sw))
        data = data.apply(lambda x: re.sub("⇓","",x))
    
        return data


    def get_data(self, data_file):
        dtf = pd.read_csv(data_file, sep = "\|\|", engine = "python")
        X = self.pre_processing(dtf["Text"])
        dataset = dtf.drop(columns = ["Score"], axis = 0)
        return X, dataset
    
    