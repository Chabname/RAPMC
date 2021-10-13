import pandas as pd

class Articles:
    datas = []

    def __init__(self, f_path):
        """ Initialize the datas
        """
        self.datas = pd.read_csv(f_path, sep = "\|\|", engine = 'python')
        self.datas.columns = ["ID","Gene","Variation","Class","Text","Score"]


         
        

class Variants:
    datas = []

    def __init__(self, f_path):
        """ Initialize the datas
        """
        self.datas = pd.read_csv(f_path, engine = 'python')
        self.datas.set_index("ID", inplace = True)
