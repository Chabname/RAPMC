import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from pickle import TRUE
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings(action = 'ignore')
from gensim.models import Word2Vec
import time
import sys
import multiprocessing
import string
import numpy as np

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#from Epoch import EpochLogger


from Datas import Articles




class EmbW2V:
    model_path="results/"

    def __init__(self, datas, win_size, epoch, batch):
        self.datas = datas
        self.win_size = win_size
        self.epoch = epoch
        self.batch = batch
        self.data_size = len(self.datas)



    ## function : 
    #       preprocess_datas
    ## input :
    #       articles : (dataframe) 
    ## output :
    #       articles : (dataframe)
    ## details : 
    #       
    def preprocess_datas(self, clean_stopword, repeat):
        print("_______________________________Preprocessing_____________________________")
        start_time = time.perf_counter()

        prog = 0

        # Replaces escape character with space
        articles = self.datas.replace("\n", " ")
        all_vect = []
        lemmatizer = WordNetLemmatizer()
        # Init the Wordnet Lemmatizer
        for article in articles.iterrows():
            self.progress(prog, len(articles), status='Preparing Datas')
            word_vector = []
            # iterate through each sentence in the file
            for sentence in sent_tokenize(article[1]["Text"]):
                sent_clean = sentence.translate(str.maketrans('', '', string.punctuation))
                temp = []
                # tokenize the sentence into words
                if clean_stopword : 
                    list_words = list(set(word_tokenize(sent_clean)) - set(stopwords.words('english')))
                else:
                    list_words = list(set(word_tokenize(sentence)))
                for word in list_words:
                    temp.append(lemmatizer.lemmatize(word))

                word_vector.append(temp)
                #word_vector = np.repeat(word_vector, repeat, axis=0)
            #repeats = np.tile(word_vector, (repeat, 1))
            all_vect.append(np.repeat(word_vector, repeat, axis=0))
            #print(all_vect)
            #all_vect.append(word_vector)   
            prog += 1

        
        articles["Text"] = all_vect

        self.datas = articles
        

        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Preprocessing finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")



    

    ## function : 
    #       cbow
    ## input :
    #       data : 
    ## output :
    #       
    ## details : 
    #       
    def cbow(self, workers, vec_size):
        """"
    
        Parameters:

        Returns:

        """
        print("_______________________________CBOW_____________________________")
        start_time = time.perf_counter()
        self.model_path = self.model_path + "cbow_A" + str(self.data_size) + "_WS" + str(self.win_size) + "_E" + str(self.epoch) +"_B" + str(self.batch) +".model"

        # Create CBOW model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = vec_size,
                        window = self.win_size,  workers=workers, batch_words = self.batch)
        model.save(self.model_path)            
            
        self.train_model()

        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("CBOW finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")


    ## function : 
    #       skipgram
    ## input :
    #       data : 
    ## output :
    #       
    ## details : 
    #       
    def skipgram(self, workers, vec_size): 
        """"
    
        Parameters:

        Returns:

        """
        print("_______________________________Skip Gram_____________________________")  
        start_time = time.perf_counter()

        self.model_path = self.model_path + "skipgram_A" + str(self.data_size) + "_WS" + str(self.win_size) + "_E" + str(self.epoch) +"_B" + str(self.batch) +".model"

        # Create Skip Gram model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = vec_size,
                            window = self.win_size, sg = 1,  workers=workers, batch_words = self.batch)
        model.save(self.model_path)

        self.train_model()

        
        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Skip Gram finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")


    ## function : 
    #       skipgram
    ## input :
    #       data : 
    ## output :
    #       
    ## details : 
    # 
    def train_model(self):
        prog = 0
        for article in self.datas.iterrows():
            self.progress(prog, self.data_size, status='Training the model')
            #if article[0]==1:
            #    print(article[1]["Text"])
            #    break
            model = Word2Vec.load(self.model_path)
            model.train(article[1]["Text"], 
                    total_examples = len(article), 
                    epochs = self.epoch, 
                    compute_loss = True)
            
            prog +=1

    ## function : 
    #       mod_path
    ## input :
    #       data : 
    ## output :
    #       
    ## details : 
    # 
    def plot_similarities(self, words_sim, top_number):
        """"

        Parameters:

        Returns:

        """
        model = Word2Vec.load(self.model_path)
        sims_mut = []
        
        for word in words_sim:
            try:
                sims_mut.append(model.wv.most_similar(word, topn = top_number))
            except:
                str_err = "this word does not exist into the model : {word}".format(word=word)
                print(str_err)
        for similarity in sims_mut:
            text = ""
            for word, dist in similarity:
                word_freq = [word] * int(dist*1000)
                text += " ".join(word_freq)
            wordcloud = WordCloud(collocations=False).generate(text)
            plt.figure(figsize=(20,10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()



    ## function : 
    #       mod_path
    ## input :
    #       data : 
    ## output :
    #       
    ## details : 
    #   
    def progress(self, count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '.' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()


## function : 
#       skipgram
## input :
#       data : 
## output :
#       
## details : 
# 
def show_similarities(mod_path, word_sim, top_number):
    """"
    
    Parameters:
        
    Returns:
        
    """
    model = Word2Vec.load(mod_path)
    try:
        sims_mut = model.wv.most_similar(word_sim, topn = top_number) 
        #print("____________________________________________________________________")
        return ("List of words most similar to {word} :\n {sim} \n").format(word=word_sim, sim=sims_mut)
    except:
        str_err = "this word does not exist into the model : {word}".format(word=word_sim)
        print(str_err)
        return str_err
   


## function : 
#       main
## input :
#       f_path : (string) 
## output :
#       na 
## details : 
#       
def main(f_path, type, win_size, epoch, batch, stop_word, repeat): 
    print("_______________________________Word2Vec_____________________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    cores = multiprocessing.cpu_count() 

    """Launch Word2Vec"""
    articles = Articles(f_path)

    if type in( "cbow", "both"):
        cbow_test = EmbW2V(articles.datas, win_size, epoch, batch)
        cbow_test.preprocess_datas(stop_word, repeat)
        #print(cbow_test.datas["Text"][33])

        cbow_test.cbow(workers = cores, vec_size = 100)
        print("__________________________Word similarities_________________________")

        print(show_similarities(cbow_test.model_path,"mutation" ,20))

        print("____________________________________________________________________")


    if type in ("skipgram", "both") :
        skipgram_test = EmbW2V(articles.datas, win_size, epoch, batch)
        skipgram_test.preprocess_datas(stop_word, repeat)

        skipgram_test.skipgram(workers = cores, vec_size = 100)
        print("__________________________Word similarities_________________________")

        print(show_similarities(skipgram_test.model_path,"mutation" ,20))

        print("____________________________________________________________________")

    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Word2vec finished in {} seconds".format(stop_time-start_time))





######################################################
##                      TEST                       ##
######################################################

#main("datas/701_mix_data_clean.txt","both", 20, 20, 10000, True, 2000)
#main("datas/all_data_clean.txt", type = "both", win_size = 50, epoch = 500, batch = 1000, False)
#print(show_similarities("results/cbow_701.model","cell" ,20))

#cbow_test = EmbW2V([], 0, 0, 0)
#cbow_test.model_path = "results/cbow_3316_5.model"
#cbow_test.plot_similarities(["mutation", "cell", "amplification", "egfr"] , 20)

#cbow_test2 = EmbW2V([], 0, 0, 0)
#cbow_test2.model_path = "results/cbow_A3316_WS50_E500_B1000.model"
#cbow_test2.plot_similarities(["mutation", "cell", "amplification", "egfr", "cancer", "heterozygous", "variant"] , 20)

#cbow_test3 = EmbW2V([], 0, 0, 0)
#cbow_test3.model_path = "results/cbow_3284.model"
#cbow_test3.plot_similarities(["mutation", "cell", "amplification", "egfr"] , 20)
#cbow_test3.plot_similarities(["cancer", "heterozygous", "variant"] , 20)
#
#skipgram_test = EmbW2V([], 0, 0, 0)
#skipgram_test.model_path = "results/skipgram_3284.model"
#skipgram_test.plot_similarities(["mutation", "cell", "amplification", "egfr"] , 20)
#skipgram_test.plot_similarities(["cancer", "heterozygous", "variant"] , 20)


#model = Word2Vec.load("results/cbow_700_lem.model")
#print(model)
#print(model.wv.vocab)
#print(model.wv.index_to_key)
#print(model.wv.key_to_index)

######################################################
##                      /TEST                       ##
######################################################