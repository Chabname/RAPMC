from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
#warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

def test_word2vec():
    #  Reads ‘alice.txt’ file
    sample = open("datas/article_sample", "r")
    s = sample.read()

    # Replaces escape character with space
    f = s.replace("\n", " ")

    data = []

    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
    
        data.append(temp)

    # Create CBOW model
    model1 = gensim.models.Word2Vec(data, min_count = 1, 
                                  size = 100, window = 5)
    
    # Print results
    print("Cosine similarity between 'CBL' " + 
                   "and 'W802*' - CBOW : ",
        model1.similarity('CBL', 'W802*'))

    print("Cosine similarity between 'CBL' " +
                     "and 'mutation' - CBOW : ",
          model1.similarity('CBL', 'mutation'))

    print("Cosine similarity between 'W802*' " +
                     "and 'mutation' - CBOW : ",
          model1.similarity('W802*', 'mutation'))

    print()
    
    # Create Skip Gram model
    model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100,
                                                 window = 5, sg = 1)
    
    # Print results
    print("Cosine similarity between 'alice' " +
              "and 'wonderland' - Skip Gram : ",
        model2.similarity('alice', 'wonderland'))

    print("Cosine similarity between 'alice' " +
                "and 'machines' - Skip Gram : ",
          model2.similarity('alice', 'machines'))



def main():
    """Launch the application"""


    

main()