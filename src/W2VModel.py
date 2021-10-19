from Model import Model
from Datas import Articles
from EmbW2V import EmbW2V

class W2V(Model):
         
    def __init__(self, data_file):
        """ Initialize Low matrix before completing

        Keyword arguments:
                data_file -- 
        """

        super().__init__(data_file)

    def pre_processing(self,  win_size, epoch, batch, concat, repeat, clean_stopword):
        articles = Articles(self.data_file)
        learned_model = EmbW2V(articles.datas, win_size, epoch, batch, concat, repeat)
        learned_model.preprocess_datas(clean_stopword)
        return learned_model