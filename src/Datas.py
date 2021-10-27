import pandas as pd

class Articles:
    datas = []

    def __init__(self, f_path, is_training):
        """ 
        Get all datas (Gene variantions, class and text, then create a dataframe

        function : 
           __init__
        input :
            f_path : (str) path of concatenate clean datas
            is_training : (bool) training mode
        output :
            dataframe with all datas
        details : 
            Get all datas, if "is_raining" mode, there is a "Class" column
        """
        self.datas = pd.read_csv(f_path, sep = "\|\|\|", engine = 'python')
        if is_training:
            self.datas.columns = ["ID","Gene","Variation","Class","Text","Score"]
        else:
            self.datas.columns = ["ID","Gene","Variation","Text","Score"]


         
        


