import nltk
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings(action = 'ignore')
#import gensim
from gensim.models import Word2Vec
import time


######################################################
##                TEST Imports                      ##
######################################################

from Datas import Articles

######################################################
##                /TEST Imports                     ##
######################################################



class EmbW2V:
    model_path="results/"
    
    def __init__(self, datas):
        self.datas = datas




    ## function : 
    #       preprocess_datas
    ## input :
    #       articles : (dataframe) 
    ## output :
    #       articles : (dataframe)
    ## details : 
    #       
    def preprocess_datas(self):
        print("_______________________________Preprocessing_____________________________")
        start_time = time.perf_counter()
        # Replaces escape character with space
        articles = self.datas.replace("\n", " ")

        data = []
        lemmatizer = WordNetLemmatizer()

        # Init the Wordnet Lemmatizer
        for article in articles.iterrows():
            # iterate through each sentence in the file
            for i in sent_tokenize(article[1]["Text"]):
                temp = []

                # tokenize the sentence into words
                for j in word_tokenize(i):
                    temp.append(lemmatizer.lemmatize(j.lower()))

                data.append(temp)
            article[1]["Text"] = data

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
    def cbow(self):

        print("_______________________________CBOW_____________________________")
        start_time = time.perf_counter()
        length_datas = len(self.datas)
        self.model_path = self.model_path + "cbow_"+str(length_datas)+".model"

        # Create CBOW model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = 100,
                        window = 5,  workers=4)
        model.save(self.model_path)

        for article in self.datas.iterrows():
            model = Word2Vec.load(self.model_path)
            model.train(article[1]["Text"], total_examples = len(article), epochs=15)

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
    def skipgram(self): 
        print("_______________________________Skip Gram_____________________________")  
        start_time = time.perf_counter()

        length_datas = len(self.datas)
        self.model_path = self.model_path + "skipgram_"+str(length_datas)+".model"

        # Create Skip Gram model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = 100,
                            window = 5, sg = 1,  workers=4)
        model.save(self.model_path)

        for article in self.datas.iterrows():
            model = Word2Vec.load(self.model_path)
            model.train(article[1]["Text"], total_examples = len(article), epochs=15)

        
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
def show_similarities(mod_path, word_sim, top_number):
    model = Word2Vec.load(mod_path)
    try:
        sims_mut = model.wv.most_similar(word_sim, topn = top_number) 
        print("List of words most similar to '" + word_sim + "' :")
        print(sims_mut)
        print("____________________________________________________________________")
    except:
        print("this wrd does not exist into the model : " + word_sim)
        pass
   

######################################################
##                TEST Functions                    ##
######################################################

## function : 
#       model_test
## input :
#       f_path : (string) 
## output :
#       n 
## details : 
#       This function is used to see the result of 
#       To run it : 
#               Uncomment the line into the "TEST" section and run this file only
#       DON'T FORGET to comment again after testing !
def model_test(f_path): 
    print("_______________________________Word2Vec_____________________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    """Launch Word2Vec"""
    articles = Articles(f_path)

    cbow_test = EmbW2V(articles.datas)
    cbow_test.preprocess_datas()

    cbow_test.cbow()
    print("__________________________Word similarities_________________________")

    show_similarities(cbow_test.model_path,"mutation" ,20)

    print("____________________________________________________________________")


    
    skipgram_test = EmbW2V(articles.datas)
    skipgram_test.preprocess_datas()

    skipgram_test.skipgram()
    print("__________________________Word similarities_________________________")

    show_similarities(skipgram_test.model_path,"mutation" ,20)

    print("____________________________________________________________________")

    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Word2vec finished in {} seconds".format(stop_time-start_time))

######################################################
##                 /TEST  functions                 ##
######################################################



######################################################
##                      TEST                       ##
######################################################

#model_test("datas/cbl_clean_article.txt")
#model_test("datas/sample_data_clean.txt")



#print("__________________________CBOW SIMILARITIES_________________________")
#show_similarities("results/cbow_23.model","mutation" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_23.model","cbl" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_23.model","ptprt" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_23.model","brca1" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_23.model","rheb" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_23.model","tert" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_23.model","mycn" ,20)
#print("____________________________________________________________________")
#
#
#print("________________________SkipGram SIMILARITIES_______________________")
#show_similarities("results/skipgram_23.model","mutation" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_23.model","cbl" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_23.model","ptprt" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_23.model","brca1" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_23.model","rheb" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_23.model","tert" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_23.model","mycn" ,20)
#print("____________________________________________________________________")


#print("__________________________CBOW SIMILARITIES_________________________")
#show_similarities("results/cbow_700.model","mutation" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","cbl" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","ptprt" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","brca1" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","rheb" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","tert" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","mycn" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","v391i" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","truncating mutations" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","r1095h" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","f1088sfs*2" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","deletion" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","fusion" ,20)
#print("____________________________________________________________________")
#show_similarities("results/cbow_700.model","insertion" ,20)
#print("____________________________________________________________________")
#
#
#print("________________________SkipGram SIMILARITIES_______________________")
#show_similarities("results/skipgram_700.model","mutation" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","cbl" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","ptprt" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","brca1" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","rheb" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","tert" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","mycn" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","v391i" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","truncating" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","r1095h" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","f1088sfs*2" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","deletion" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","fusion" ,20)
#print("____________________________________________________________________")
#show_similarities("results/skipgram_700.model","insertion" ,20)
#print("____________________________________________________________________")


#model = Word2Vec.load("results/cbow_700_lem.model")
#print(model.wv.vocab)
#print(model.wv.index())

######################################################
##                      /TEST                       ##
######################################################