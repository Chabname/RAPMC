import pandas as pd
import numpy as np

import string
import time
import sys 

from Datas import Articles

from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class Vector():
    def __init__(self, data_file, model_path, is_training):
        """ 
        Initialize Vector Object before completing

        function : 
           __init__
        input :
            data_file : (str) path of concatenate clean datas (training or test)
            model_path : (str) path of the model to useto get vectors
            is_training : (bool) training mode
        output :
            Object
        details : 
            Prepare the object
        """
        self.model_path = model_path
        self.model = Word2Vec.load(model_path)
        self.data_file = data_file
        self.is_training = is_training



    def get_vector_datas(self, is_notebook, type_sum, top_number = 50):
        """
        Initialize Vector Object before completing

        function : 
           get_vector_datas
        input :
            is_notebook : (bool) Mode calling from Notebook
            type_sum : (str) Different mode for Sum
                'gene_var' : Sum only Gene and variant vectores
                'best_sim' : Sum of vector Variation and all top_number vectors 
                    most similar to Variation without vectors of other existing 
                    variations for the gene um
                    (similarty by computing cosine similarity)
                'clean_art' : Sum of all word vector from the cleaned article 
                    having preprocessing
                'sim_gene_var' : Sum of vector Variation and vector Gene and all 
                    top_number vectors most similar to Variation and Gene without 
                    the vector of other existing variation for the gene 
                    (similarty by computing cosine similarity)
            top_number : (int) number of similarties desired
        output :
            Vector of Vector Gene, Vairation, and Sum
            Log File of Gene or Variations not found
        details : 
            Top_number only use with best_sim or sim_gene_var type
        """
        print("________________________Getting datas vectors_______________________")
        print("____________________________________________________________________")
        start_time = time.perf_counter()

        # Getting all the datas
        all_datas = Articles(self.data_file, self.is_training)
        # Doesn't need Id and Score column
        datas_df = all_datas.datas.drop(['ID', 'Score'], axis=1)
        # Not clean article mode, doesnt need the Text coumns
        if type_sum != "clean_art":
            datas_df = datas_df.drop(['Text'], axis=1)
        # Preparaing datas
        datas_df["Gene"] = datas_df["Gene"].str.lower()
        datas_df["Variation"] = datas_df["Variation"].str.lower()
        size_datas_df = len(datas_df)
        prog = 0
        new_data = []
        not_found_var_list =[]
        not_found_art_list =[]
        sum_vec = []

        log_file_path = ""
        
        # Self training mode needs column class
        if self.is_training:
            datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum', 'Class'])
        else:
            datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum'])

        # Preparing Vectors
        for _, row in datas_df.iterrows():
            self.progress(prog, size_datas_df, "Getting datas vectors")

            var_clean=""
            # Get a clean word (prepocess)
            gene_clean = clean_word(row['Gene'])
            vec_var =  np.full((1, 100), 0)

            try:
                vec_gene = self.model.wv.get_vector(gene_clean)
            except:
                # If the word is not found, it's concatenate for a log list
                line = row
                line['Target'] = var_clean
                not_found_var_list.append(line)
                prog += 1
                continue
            
            # Separate all variation words
            var_list = row['Variation'].split(" ")
            vect_var_clean = []

            for ind_var in range(len(var_list)):
                # Clean the word
                var_clean = clean_word(var_list[ind_var])
                try:
                    # if found, the vector is sum with all precedent variation vectors
                    vec_var = vec_var + self.model.wv.get_vector(var_clean)
                    # Keep the variation name into a vector
                    vect_var_clean.append(var_clean)
                except:
                    # If the word is not found, it's concatenate for a log list
                    line = row
                    line['Target'] = var_clean
                    not_found_var_list.append(line)
                    continue

            # Preparing Sum coluns  Using different modes
            if type_sum == "gene_var":
                # Simple sum of vector gene and vector variations
                sum_vec = vec_gene + vec_var

            if type_sum == "best_sim" or type_sum == "sim_gene_var":
                # Getting all line for the current "Gene"
                df_bool_gene = (datas_df["Gene"] == row["Gene"])
                vect_neg_clean = []
                sum_best_sim = np.full((1, 100), 0)
                for ind_df, bool_val in enumerate(df_bool_gene):
                    # Checking if the variations are not the current variation and concatenate into negative list
                    if bool_val and (datas_df.loc[ind_df]["Variation"] != row['Variation']):
                        neg_list = datas_df.loc[ind_df]["Variation"].split(" ")
                        for ind_neg_var in range(len(neg_list)):
                            neg_var_clean = clean_word(neg_list[ind_neg_var])
                            try:
                                vec_gene = self.model.wv.get_vector(neg_var_clean)
                                vect_neg_clean.append(neg_var_clean)
                            except:
                                # if the negative word doesn"t exist, continues
                                continue
                # For similarities Gene and variaiton, we append to the variation list (positive words) the gene vector
                if type_sum == "sim_gene_var":
                    vect_var_clean.append(gene_clean)
                # Check if the vector is not empty
                if vect_var_clean:
                    most_simlar = self.model.wv.most_similar(positive = vect_var_clean, negative = vect_neg_clean, topn = top_number)
                    # Sum of all positive vectors 
                    for sim_word in most_simlar :
                        sum_best_sim = sum_best_sim + self.model.wv.get_vector(sim_word[0])
                    sum_vec = sum_best_sim

            if type_sum == "clean_art":
                # Preprocessing article to have the same format used into training model (src/EmbW2V.py)
                article = preprocess_article(row['Text'])
                vec_art =  np.full((1, 100), 0)
                for sentence in article :
                    for word in sentence:
                        try:
                            # Sum all words vectors of the article
                            vec_art = vec_art + self.model.wv.get_vector(word)
                        except:
                            # if word not found, appending into a list to count them
                            line = row
                            line['Target'] = word
                            not_found_art_list.append(line)
                            continue
                sum_vec = vec_art


            # Incrementation de la progress bar
            prog += 1

            if self.is_training:
                new_data.append([vec_gene, vec_var, sum_vec, row['Class']])
            else:
                new_data.append([vec_gene, vec_var, sum_vec])

            
        # Log the gene or variant not found
        split_filename = self.model_path.split("/")
        modele_name = split_filename[len(split_filename) - 1]
        if is_notebook:
            log_file_path = "../../../"
        log_file_path += "results/log_not_found_words_" + modele_name + ".txt"

        if self.is_training:
            not_found_var = pd.DataFrame(not_found_var_list, columns =['Gene', 'Variation', 'Class', 'Target'])
            np.savetxt(log_file_path, 
                        not_found_var, 
                        fmt = "%s\t\t\t%s\t\t\t%d\t\t\t%s", 
                        header= "\t\t\t".join(not_found_var.columns), 
                        comments='')
            not_found_art = pd.DataFrame(not_found_art_list, columns =['Gene', 'Variation', 'Text', 'Class', 'Target'])
            datas_vect = pd.DataFrame(new_data, columns =['Gene', 'Variation', 'Sum', 'Class'])
        else:
            not_found_var = pd.DataFrame(not_found_var_list, columns =['Gene', 'Variation', 'Target'])
            np.savetxt(log_file_path, 
                        not_found_var, 
                        fmt = "%s\t\t\t%s\t\t\t%s", 
                        header= "\t\t\t".join(not_found_var.columns), 
                        comments='')

            not_found_art = pd.DataFrame(not_found_art_list, columns =['Gene', 'Variation', 'Text', 'Target'])
            datas_vect = pd.DataFrame(new_data, columns =['Gene', 'Variation', 'Sum'])

        # Print number off Gene and Variation not found 
        print("Number of genes/variations not foud : " + str(len(not_found_var)))
        if type_sum == "clean_art":
            # Print number of word into articles not found
            print("Number of article words not foud : " + str(len(not_found_art)))
        
        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Getting vectors finished in {} seconds".format(stop_time-start_time))

        # Put the vecto into the object
        self.vectors = datas_vect


    def progress(self, count, total, status=''):
        """ 
        Print a progress bar

        function : 
           progress
        input :
            count : (int) current count progression
            total : (int) Total of itertions 
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


def preprocess_article(article):
    """ 
    Prepocess the article to have vectors of word cleaned :
    meaning lematized, without stop words and punctuation

    function : 
       preprocess_article
    input :
        article : (str) article text
    output :
        vector of article words
    """
    # Replaces escape character with space
    word_vector = []
    lemmatizer = WordNetLemmatizer()
    # Init the Wordnet Lemmatizer
    word_vector = []
    # iterate through each sentence in the file
    for sentence in sent_tokenize(article):
        sent_clean = sentence.translate(str.maketrans('', '', string.punctuation))
        temp = []
        # tokenize the sentence into words
        list_words = list(set(word_tokenize(sent_clean)) - set(stopwords.words('english')))
        for word in list_words:
            temp.append(lemmatizer.lemmatize(word))
        word_vector.append(temp)
    return word_vector


def clean_word(word):
    """ 
    Clean the wordto have it in the same mode than the model learned :
    meaning lematized, without punctuation

    function : 
       clean_word
    input :
        word : (str) gene or variant word
    output :
        Word in the same mode than what the model learned
    """
    lemmatizer = WordNetLemmatizer()
    clean_var = word.translate(str.maketrans('', '', string.punctuation))
    clean_var = lemmatizer.lemmatize(clean_var)
    return clean_var


def main(data_file, model_path, is_training, is_notebook):
    """
        Process to get the vectors of datas.

        function : 
           main
        input :
            data_file : (str) path of concatenate clean datas (training or test)
            model_path : (str) path of the model to useto get vectors
            is_training : (bool) training mode
            is_notebook : (bool) Mode calling from Notebook
        output :
            Get the vectors of datas into the object parameter
        details : 
            type_sum values : gene_var, best_sim, clean_art, sim_gene_var
    """
    datas = Vector(data_file, model_path, is_training)
    datas.get_vector_datas(is_notebook, type_sum = "gene_var")
    #print (datas.vectors['Sum'])
    
    

######################################################
##                       TEST                       ##
######################################################

#main("datas/training_clean", "datas/cbow_A3316_WS20_E20_B10000_R2000_CTrue.model", True, False)
#main("datas/training_clean", "datas/cbow_3284.model", True, False)
#main("datas/training_clean", "datas/skipgram_3284.model", True, False)
#main("datas/test_clean", "results/Copy_cbow_A701_WS20_E15_B10000_R200_CTrue.model", False, False)

######################################################
##                      /TEST                       ##
######################################################