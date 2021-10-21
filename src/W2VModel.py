from Model import Model
from Datas import Articles
from EmbW2V import EmbW2V

from gensim.models import Word2Vec
import time

class W2V(Model):
    model_path="results/"

    def __init__(self, data_file, learned_mod_path, win_size, epoch, batch, stopword, repeat, concat):
        """ Initialize Word2Vec model

        Keyword arguments:
                data_file -- 
        """

        super().__init__(data_file)
        self.learned_mod_path = learned_mod_path
        self.win_size = win_size
        self.epoch = epoch
        self.batch = batch
        self.concat = concat
        self.repeat = repeat
        self.stopword = stopword


    def pre_processing(self):
        articles = Articles(self.data_file, False)
        emb_model = EmbW2V(articles.datas, self.win_size, self.epoch, self.batch, self.concat, self.repeat)
        emb_model.preprocess_datas(self.stopword)
        return emb_model


    def copy_embeded_model(self):
        model = Word2Vec.load(self.learned_mod_path)
        split_filename = self.learned_mod_path.split("/")
        filename = split_filename[len(split_filename) - 1]
        self.model_path += "Copy_" + filename
        model.save(self.model_path)
    

    def train_embeded_model(self):
        emb_model = self.pre_processing()
        emb_model.model_path = self.model_path
        print("Path of the new model : " + emb_model.model_path)
        emb_model.train_model()


def main(data_file, learned_mod_path, win_size, epoch, batch, stopword, repeat, concat):
    print("__________________Training W2V Model with new datas_________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    w2vmodel = W2V(data_file, learned_mod_path, win_size, epoch, batch, concat, repeat, stopword)
    w2vmodel.copy_embeded_model()
    w2vmodel.train_embeded_model()

    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Training finished in {} seconds".format(stop_time-start_time))


#main("datas/sample_test_clean", 
#        "results/cbow_A701_WS20_E15_B10000_R20_CTrue.model",
#        type = "cbow", 
#        win_size = 20, 
#        epoch = 15, 
#        batch = 10000, 
#        stopword = True, 
#        repeat = 20, 
#        concat = True)
#

def test():
    model = Word2Vec.load("datas/cbow_A3316_WS20_E20_B10000_R2000_CTrue.model")

    print(model.wv.get_vector("egfr"))
    #model.wv.index_to_key()

test()