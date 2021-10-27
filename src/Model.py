from nltk.corpus import stopwords
import pandas as pd
import re


class Model:
    model_path="results/"

    def __init__(self, data_file):
        """ 
        Initialize Model Object before completing

        function : 
           __init__
        input :
            data_file : (str) path of concatenate clean datas (training or test)
        output :
            Object
        details : 
            Prepare the object
        """
        self.data_file = data_file

    def pre_processing(data):
        """ 
        Preprocess the articles. simple preprocessing

        function : 
           pre_processing
        input :
            data : (dataframe) article
        output :
            data : (vector) vecotrs of words
        """
        sw = stopwords.words("english")
        # lowercase text
        data = data.apply(lambda x: " ".join(i.lower() for i in  str(x).split()))  
        data = data.apply(lambda x: " ".join(i for i in x.split() if i not in sw))
        data = data.apply(lambda x: re.sub("⇓","",x))
    
        return data


    def get_data(self, data_file):
        """ 
        Getting all datas concatenated from clean article file

        function : 
           get_data
        input :
            data_file : (str) path of concatenate clean datas (training or test)
        output :
            Vector of words and Dataset of all datas
        """
        dtf = pd.read_csv(data_file, sep = "\|\|", engine = "python")
        X = self.pre_processing(dtf["Text"])
        dataset = dtf.drop(columns = ["Score"], axis = 0)
        return X, dataset
    
    