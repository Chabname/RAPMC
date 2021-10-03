#nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
import time

from Datas import Articles

class EmbW2V:

    def __init__(self, datas):
        self.datas = datas


    ## function : 
    #       prepare_datas
    ## input :
    #       articles : (dataframe) 
    ## output :
    #       articles : (dataframe)
    ## details : 
    #       
    def prepare_datas(self, articles):
        
        # Replaces escape character with space
        articles = articles.replace("\n", " ")

        data = []

        for article in articles.iterrows():
            # iterate through each sentence in the file
            for i in sent_tokenize(article[1]["Text"]):
                temp = []

                # tokenize the sentence into words
                for j in word_tokenize(i):
                    temp.append(j.lower())

                data.append(temp)
            article[1]["Text"] = data

        return articles



    ## function : 
    #       cbow
    ## input :
    #       data : 
    ## output :
    #       
    ## details : 
    #       
    def cbow(self, articles):

        print("_______________________________CBOW_____________________________")
        start_time = time.perf_counter()

        # Create CBOW model
        model = Word2Vec(articles.iloc[0]["Text"], min_count = 1, vector_size = 100,
                        window = 5,  workers=4)
        model.save("results/cbow.model")

        for article in articles.iterrows():
            model = Word2Vec.load("results/cbow.model")
            model.train(article[1]["Text"], total_examples = len(article), epochs=15)

        # Print results
        print("Cosine similarity between 'cbl' " + 
                       "and 'w802' - CBOW : ",
            model.wv.similarity('cbl', 'w802'))

        print("Cosine similarity between 'cbl' " +
                         "and 'mutation' - CBOW : ",
              model.wv.similarity('cbl', 'mutation'))

        print("Cosine similarity between 'w802' " +
                         "and 'mutation' - CBOW : ",
              model.wv.similarity('w802', 'mutation'))

        print("Cosine similarity between 'w802' " +
                         "and 'transfection' - CBOW : ",
              model.wv.similarity('w802', 'transfection'))

        sims_mut = model.wv.most_similar('mutation', topn=10) 
        print("List of words most similar to 'mutation' :")
        print(sims_mut)
        sims_cbl = model.wv.most_similar('cbl', topn=10) 
        print("List of words most similar to 'cbl' :")
        print(sims_cbl)
        sims_minutes = model.wv.most_similar('minutes', topn=10) 
        print("List of words most similar to 'minutes' :")
        print(sims_minutes)

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
    def skipgram(self, articles): 
        print("_______________________________Skip Gram_____________________________")  
        start_time = time.perf_counter()

        # Create Skip Gram model
        model = Word2Vec(articles.iloc[0]["Text"], min_count = 1, vector_size = 100,
                            window = 5, sg = 1,  workers=4)
        model.save("results/skipgram.model")

        for article in articles.iterrows():
            model = Word2Vec.load("results/skipgram.model")
            model.train(article[1]["Text"], total_examples = len(article), epochs=15)

        # Print results
        print("Cosine similarity between 'cbl' " +
                  "and 'W802' - Skip Gram : ",
            model.wv.similarity('cbl', 'w802'))

        print("Cosine similarity between 'cbl' " +
                         "and 'mutation' - CBOW : ",
              model.wv.similarity('cbl', 'mutation'))

        print("Cosine similarity between 'w802' " +
                         "and 'mutation' - Skip Gram : ",
              model.wv.similarity('w802', 'mutation'))

        print("Cosine similarity between 'w802' " +
                         "and 'transfection' - Skip Gram : ",
              model.wv.similarity('w802', 'transfection'))

        sims_mut = model.wv.most_similar('mutation', topn=10) 
        print("List of words most similar to 'mutation' :")
        print(sims_mut)
        sims_cbl = model.wv.most_similar('cbl', topn=10) 
        print("List of words most similar to 'cbl' :")
        print(sims_cbl)
        sims_minutes = model.wv.most_similar('minutes', topn=10) 
        print("List of words most similar to 'minutes' :")
        print(sims_minutes)
        
        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Skip Gram finished in {} seconds".format(stop_time-start_time))
        print("____________________________________________________________________")


    
   

######################################################
##                TEST Functions                    ##
######################################################

## function : 
#       main_test
## input :
#       f_path : (string) 
## output :
#       n 
## details : 
#       This function is used to see the result of 
#       To run it : 
#               Uncomment the line into the "TEST" section and run this file only
#       DON'T FORGET to comment again after testing !
def main_test(f_path): 
    print("_______________________________Word2Vec_____________________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    """Launch Word2Vec"""
    articles = Articles(f_path)

    w2v = EmbW2V(articles.datas)
    

    prep_articles = w2v.prepare_datas(articles.datas)

    w2v.cbow(prep_articles)
    w2v.skipgram(prep_articles)

    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Word2vec finished in {} seconds".format(stop_time-start_time))

######################################################
##                 /TEST  functions                 ##
######################################################



######################################################
##                      TEST                       ##
######################################################

main_test("datas/cbl_clean_article.txt")

######################################################
##                      /TEST                       ##
######################################################