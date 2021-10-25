from Model import Model
from Datas import Articles
from EmbW2V import EmbW2V

from gensim.models import Word2Vec
import time

class W2V(Model):
    model_path="results/"

    def __init__(self, data_file, learned_mod_path, win_size, epoch, batch, stopword, repeat, concat):
        """ 
        Initialize Word2Vec model Object before completing

        function : 
           __init__
        input :
            data_file : (str) path of concatenate clean datas (training or test)
            learned_mod_path :  (str) path of model to use
            win_size : (int) windo size (of context) for the training the model
            epoch : (int) number of epoch for training the model
            batch : (int) size of the batch for training the model
            repeat : (int) number of repetition
            concat : (bool) concatenating articles
        output :
            Object
        details : 
            Prepare the object
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
        """ 
        Preprocessing the Article

        function : 
           pre_processing
        input :
            NA
        output :
            model with preprocessed data
        details : 
            Use EmbW2V preprocess data function
        """
        articles = Articles(self.data_file, False)
        emb_model = EmbW2V(articles.datas, self.win_size, self.epoch, self.batch, self.concat, self.repeat)
        emb_model.preprocess_datas(self.stopword)
        return emb_model


    def copy_embeded_model(self):
        """ 
        Create a copy of the model to not loose the original

        function : 
           copy_embeded_model
        input :
            NA
        output :
            New model copied and new path model
        """
        model = Word2Vec.load(self.learned_mod_path)
        split_filename = self.learned_mod_path.split("/")
        filename = split_filename[len(split_filename) - 1]
        self.model_path += "Copy_" + filename
        model.save(self.model_path)
    

    def train_embeded_model(self):
        """ 
        Train a model with new datas

        function : 
           train_embeded_model
        input :
            NA
        output :
            Model trained and saved
        """
        emb_model = self.pre_processing()
        emb_model.model_path = self.model_path
        print("Path of the new model : " + emb_model.model_path)
        emb_model.train_model()


def main(data_file, learned_mod_path, win_size, epoch, batch, stopword, repeat, concat):
    """ 
    Process to get get the new datas, prepare the datas, copy the model and train the new (copy) model.
    
    function : 
       main
    input :
        data_file : (str) path of concatenate clean datas (training or test)
        learned_mod_path :  (str) path of model to use
        win_size : (int) windo size (of context) for the training the model
        epoch : (int) number of epoch for training the model
        batch : (int) size of the batch for training the model
        repeat : (int) number of repetition
        concat : (bool) concatenating articles
    output :
        Model(s) created, trained and saved
    """
    print("__________________Training W2V Model with new datas_________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    w2vmodel = W2V(data_file, learned_mod_path, win_size, epoch, batch, concat, repeat, stopword)
    w2vmodel.copy_embeded_model()
    w2vmodel.train_embeded_model()

    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Training finished in {} seconds".format(stop_time-start_time))


######################################################
##                       TEST                       ##
######################################################
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
######################################################
##                      /TEST                       ##
######################################################