from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings(action = 'ignore')
from gensim.models import Word2Vec
import time

from wordcloud import WordCloud
import matplotlib.pyplot as plt

######################################################
##                TEST Imports                      ##
######################################################

from Datas import Articles

######################################################
##                /TEST Imports                     ##
######################################################



class EmbW2V:
    model_path="results/"
    
    def __init__(self, datas, win_size):
        self.datas = datas
        self.win_size = win_size



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
        """"
    
        Parameters:

        Returns:

        """
        print("_______________________________CBOW_____________________________")
        start_time = time.perf_counter()
        length_datas = len(self.datas)
        self.model_path = self.model_path + "cbow_"+str(length_datas)+".model"

        # Create CBOW model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = 100,
                        window = self.win_size,  workers=4)
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
        """"
    
        Parameters:

        Returns:

        """
        print("_______________________________Skip Gram_____________________________")  
        start_time = time.perf_counter()

        length_datas = len(self.datas)
        self.model_path = self.model_path + "skipgram_"+str(length_datas)+".model"

        # Create Skip Gram model
        model = Word2Vec(self.datas.iloc[0]["Text"], min_count = 1, vector_size = 100,
                            window = self.win_size, sg = 1,  workers=4)
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
   

def plot_similarities(mod_path, words_sim, top_number):
    """"
    
    Parameters:
        
    Returns:
        
    """
    model = Word2Vec.load(mod_path)
    sims_mut = []
    for word in words_sim:
        sims_mut.append(model.wv.most_similar(word, topn = top_number))
    for similarity in sims_mut:
        text = ""
        for word, dist in similarity:
            word_freq = [word] * int(dist*1000)
            text += " ".join(word_freq)
        wordcloud = WordCloud(collocations=False).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


######################################################
##                TEST Functions                    ##
######################################################

## function : 
#       model_test
## input :
#       f_path : (string) 
## output :
#       na 
## details : 
#       This function is used to see the result of 
#       To run it : 
#               Uncomment the line into the "TEST" section and run this file only
#       DON'T FORGET to comment again after testing !
def model_test(f_path, type, win_size): 
    print("_______________________________Word2Vec_____________________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    """Launch Word2Vec"""
    articles = Articles(f_path)

    if type in( "cbow", "both"):
        cbow_test = EmbW2V(articles.datas, win_size)
        cbow_test.preprocess_datas()

        cbow_test.cbow()
        print("__________________________Word similarities_________________________")

        print(show_similarities(cbow_test.model_path,"mutation" ,20))

        print("____________________________________________________________________")


    if type in ("skipgram", "both") :
        skipgram_test = EmbW2V(articles.datas, win_size)
        skipgram_test.preprocess_datas()

        skipgram_test.skipgram()
        print("__________________________Word similarities_________________________")

        print(show_similarities(skipgram_test.model_path,"mutation" ,20))

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

#model_test("datas/10_score1_data_clean.txt","both", 20)

#plot_similarities("results/cbow_3284.model",["l861p"] , 20)

#with open('results/cbow_3284.txt', 'w') as f:
#    print("__________________________CBOW SIMILARITIES_________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","mutation" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","cbl" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","ptprt" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","brca1" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","rheb" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","tert" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","mycn" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","v391i" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","truncating mutations" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","r1095h" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","f1088sfs*2" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","deletion" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","fusion" ,20)))
#    print("\n____________________________________________________________________", file=f)
#    f.write(str(show_similarities("results/cbow_3284.model","insertion" ,20)))
#    print("\n____________________________________________________________________", file=f)



#model = Word2Vec.load("results/cbow_700_lem.model")
#print(model)
#print(model.wv.vocab)
#print(model.wv.index_to_key)
#print(model.wv.key_to_index)

######################################################
##                      /TEST                       ##
######################################################