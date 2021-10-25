import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from pickle import TRUE
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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

from Epoch import LossLogger


from Datas import Articles




class EmbW2V:
    model_path="results/"

    def __init__(self, datas, win_size, epoch, batch, concat, repeat):
        """ 
        Initialize model Object before completing

        function : 
           __init__
        input :
            datas : (dataframe) all the datas from files
            win_size : (int) windo size (of context) for the training the model
            epoch : (int) number of epoch for training the model
            batch : (int) size of the batch for training the model
            concat : (bool) concatenating articles
            repeat : (int) number of repetition
        output :
            Object
        details : 
            Prepare the object
        """
        self.datas = datas
        self.win_size = win_size
        self.epoch = epoch
        self.batch = batch
        self.data_size = len(self.datas)
        self.concat = concat
        self.repeat = repeat



    def preprocess_datas(self, clean_stopword):
        """ 
        Preprocess the articles using different types of preprocessing 

        function : 
           preprocess_datas
        input :
            clean_stopword : (bool) choose to clean stopwords or not
        output :
            Vector of articles into the object
        details : 
            The cleaning is different using the object parameters :
                Not concatening + not repeating (=0) : each row only have a vector of his own article
                Not concatening + repeating : each row have a vector of his own article repeated `self.repeat` times
                Concatening : each row have a vector of all articles. The `self.repeat` is used to repeat the learning 
                    (of the first article, it contains all articles)
                
        """
        print("_______________________________Preprocessing_____________________________")
        start_time = time.perf_counter()
        prog = 0

        # Replaces escape character with space
        articles = self.datas.replace("\n", " ")
        all_vect = []
        word_vector = []
        lemmatizer = WordNetLemmatizer()

        # Init the Wordnet Lemmatizer
        for article in articles.iterrows():
            self.progress(prog, len(articles), status='Preparing Datas')

            if not self.concat:
                word_vector = []
            # iterate through each sentence in the file
            for sentence in sent_tokenize(article[1]["Text"]):
                sent_clean = sentence.translate(str.maketrans('', '', string.punctuation))
                temp = []
                # tokenize the sentence into words and if nedded delete stopwords
                if clean_stopword : 
                    list_words = list(set(word_tokenize(sent_clean)) - set(stopwords.words('english')))
                else:
                    list_words = list(set(word_tokenize(sentence)))
                for word in list_words:
                    temp.append(lemmatizer.lemmatize(word))

                word_vector.append(temp)
                
            # Checking object parameters
            if not self.concat : 
                if self.repeat != 0:
                    # For non concat mode, if repetition, we concatenate the same vector
                    all_vect.append(np.repeat(word_vector, self.repeat, axis=0))
                else:
                     # For non concat mode, if not repetition, the vector is only one article vector
                    all_vect.append(word_vector)
              
            prog += 1

        
        if self.concat :
            for i,_ in enumerate(articles["Text"]):
                # For concat mode we concat all the articles
                articles["Text"][i] = word_vector
        else:
            articles["Text"] = all_vect

        self.datas = articles
        

        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Preprocessing finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")



    
     
    def cbow(self, workers, vec_size):
        """ 
        Create and save the model cbow. 
        This model architecture tries to predict the current target word 
        (the center word) based on the source context words (surrounding words).  

        function : 
           cbow
        input :
            workers : (int) number of CPU to use
            vec_size : (int) size of the vector for the model
        output :
            model cbow initiate and saved
                
        """

        print("_______________________________CBOW_____________________________")
        start_time = time.perf_counter()
        self.model_path = (self.model_path + "cbow_A" + str(self.data_size) 
        + "_WS" + str(self.win_size) + "_E" + str(self.epoch) + "_B" + str(self.batch) 
        + "_R" + str(self.repeat) + "_C" + str(self.concat) + ".model")

        # Create CBOW model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = vec_size,
                        window = self.win_size,  workers=workers, batch_words = self.batch)
        model.save(self.model_path)            
            
        self.train_model()

        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("CBOW finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")




    def skipgram(self, workers, vec_size): 
        """ 
        Create and save the model skipgram. 
        This model architecture is a deep learning classification model 
        such that we take in the target word as our input and try to predict the context words.   

        function : 
           skipgram
        input :
            workers : (int) number of CPU to use
            vec_size : (int) size of the vector for the model
        output :
            model skipgram initiate and saved
                
        """
        print("_______________________________Skip Gram_____________________________")  
        start_time = time.perf_counter()

        self.model_path = (self.model_path + "skipgram_A" + str(self.data_size) 
        + "_WS" + str(self.win_size) + "_E" + str(self.epoch) + "_B" + str(self.batch) 
        + "_R" + str(self.repeat) + "_C" + str(self.concat) + ".model")
       
        # Create Skip Gram model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = vec_size,
                            window = self.win_size, sg = 1,  workers=workers, batch_words = self.batch)
        model.save(self.model_path)

        self.train_model()

        
        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Skip Gram finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")


 
    def train_model(self):
        """ 
        Training the model using object parameters 

        function : 
           train_model
        input :
            NA
        output :
            Train model saved and log loss file saved
        details : 
            The training is different using the object parameters :
                Not concatening + not repeating (=0) : each row only have a vector of his own article
                Not concatening + repeating : each row have a vector of his own article repeated `self.repeat` times
                Concatening : each row have a vector of all articles. The `self.repeat` is used to repeat the learning 
                    (of the first article, it contains all articles)
                
        """
        prog = 0
        split_filename = self.model_path.split("/")
        filename = split_filename[len(split_filename) - 1]
        loss_logger = LossLogger(filename)
        
        # Training the model using object parameters
        if not self.concat:
            # For non concat mode, we iterrate into all rows
            for article in self.datas.iterrows():
                #progress bar
                self.progress(prog, self.data_size, status='Training the model')
                model = Word2Vec.load(self.model_path)
                # The first row is already learned while creating and saving the model
                model.train(article[1]["Text"], 
                    total_examples = len(article),
                    epochs = self.epoch, 
                    callbacks=[loss_logger], 
                    compute_loss = True)
                #reinitiate the epoch for the log loss file (because of repetiions)
                if loss_logger.epoch == self.epoch +1:
                    loss_logger.epoch=1
                prog +=1

        # Concat mode       
        else:
            # For concat mode, we iterrate self.repeat times
            for _ in range(self.repeat):
                self.progress(prog, self.repeat, status='Training the model')
                model = Word2Vec.load(self.model_path)
                # Concat mode has all articles concatenated, the first row is enough
                model.train(self.datas.iloc[0]["Text"], 
                        total_examples = self.repeat, 
                        epochs = self.epoch, 
                        callbacks=[loss_logger], 
                        compute_loss = True)
                #reinitiate the epoch for the log loss file (because of repetiions)
                if loss_logger.epoch == self.epoch +1:
                    loss_logger.epoch=1
                prog +=1




    def plot_similarities(self, words_sim, top_number):
        """ 
        Plot the "top_number" most similar words of the input word by computing cosine similarity

        function : 
           plot_similarities
        input :
            words_sim : (list) List of words wanted to plot
            top_number : (int) number of most isimlar words
        output :
            One graphique for each word in the list
        """
        model = Word2Vec.load(self.model_path)
        sims_mut = []
        
        for word in words_sim:
            try:
                # Gettinf all similar word vectors for each input words
                sims_mut.append(model.wv.most_similar(word, topn = top_number))
            except:
                # if a word is not found, print the word
                str_err = "this word does not exist into the model : {word}".format(word=word)
                print(str_err)
        #For each similarities
        for similarity in sims_mut:
            text = ""
            for word, dist in similarity:
                # repeat a word dist*1000 times to get a frequencie of word (conversion of distance similarity)
                word_freq = [word] * int(dist*1000)
                text += " ".join(word_freq)
            # generate the graph (normally using frequence of word)
            wordcloud = WordCloud(collocations=False).generate(text)
            plt.figure(figsize=(20,10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()



    def progress(self, count, total, status=''):
        """ 
        Print a progress bar

        function : 
           progress
        input :
            count : (int) current count progression
            total : (nt) Total of itertions 
            status : (str) Status for the progress bar
        output :
              print into a terminal
        details : 
            Only for terminal, doesn't work for notebooks
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '.' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()




def show_similarities(mod_path, word_sim, top_number):
    """ 
    Print the "top_number" most similar words of the input word by computing cosine similarity 

    function : 
       show_similarities
    input :
        words_sim : (list) List of words wanted to plot
        top_number : (int) number of most isimlar words
    output :
        Print the list of similarites (word, distance) for "to_number) similar words
    """
    model = Word2Vec.load(mod_path)
    try:
        sims_mut = model.wv.most_similar(word_sim, topn = top_number) 
        return ("List of words most similar to {word} :\n {sim} \n").format(word=word_sim, sim=sims_mut)
    except:
        str_err = "this word does not exist into the model : {word}".format(word=word_sim)
        print(str_err)
        return str_err
   


     
def main(f_path, type, win_size, epoch, batch, stop_word, repeat, concat): 
    """ 
    Process to get get the datas, prepare the datas, create and train the model.
    
    function : 
       main
    input :
        f_path : (str) data clean file path
        type : (str) Type of model to create
            `cbow`
            `skipgram`
            `both`
        win_size : (int) windo size (of context) for the training the model
        epoch : (int) number of epoch for training the model
        batch : (int) size of the batch for training the model
        stop_word : (bool) 
        repeat : (int) number of repetition
        concat : (bool) concatenating articles
    output :
        Model(s) created, trained and saved
    """

    print("_______________________________Word2Vec_____________________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    cores = multiprocessing.cpu_count() 

    # Get the articles
    articles = Articles(f_path, True)

    # create cbow model
    if type in( "cbow", "both"):
        cbow_test = EmbW2V(articles.datas, win_size, epoch, batch, concat, repeat)
        cbow_test.preprocess_datas(stop_word)

        cbow_test.cbow(workers = cores, vec_size = 100)
        print("__________________________Word similarities_________________________")

        print(show_similarities(cbow_test.model_path,"mutation" ,20))

        print("____________________________________________________________________")

    # create skipgram model
    if type in ("skipgram", "both") :
        skipgram_test = EmbW2V(articles.datas, win_size, epoch, batch, concat, repeat)
        skipgram_test.preprocess_datas(stop_word)

        skipgram_test.skipgram(workers = cores, vec_size = 100)
        print("__________________________Word similarities_________________________")

        print(show_similarities(skipgram_test.model_path,"mutation" ,20))

        print("____________________________________________________________________")

    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Word2vec finished in {} seconds".format(stop_time-start_time))





######################################################
##                       TEST                       ##
######################################################

#main("datas/701_mix_data_clean.txt", 
#       type = "cbow", 
#       win_size = 20, 
#       epoch = 15, 
#       batch = 10000, 
#       stop_word = True, 
#       repeat = 2000, 
#       concat = True)
#main("datas/701_mix_data_clean.txt", 
#        type = "cbow", 
#        win_size = 20, 
#        epoch = 15, 
#        batch = 10000, 
#        stop_word = True, 
#        repeat = 200, 
#        concat = True)
#main("datas/701_mix_data_clean.txt", 
#       type = "both", 
#       win_size = 50, 
#       epoch = 15, 
#       batch = 10000, 
#       stop_word = True, 
#       repeat = 0, 
#       concat = False)
#main("datas/701_mix_data_clean.txt", 
#       type = "cbow", 
#       win_size = 50, 
#       epoch = 15, 
#       batch = 10000, 
#       stop_word = True, 
#       repeat = 2000, 
#       concat = False)



#model = Word2Vec.load("results/cbow_700_lem.model")
#print(model)
#print(model.wv.vocab)
#print(model.wv.index_to_key)
#print(model.wv.key_to_index)

######################################################
##                      /TEST                       ##
######################################################