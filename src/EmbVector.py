import pandas as pd
import numpy as np

import string
import time
import multiprocessing
import sys 

from Datas import Articles

from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer

class Vector():
    def __init__(self, data_file, model_path, is_training):
        """ Initialize Vector before completing

        Keyword arguments:
                data_file -- 
        """
        self.model_path = model_path
        self.model = Word2Vec.load(model_path)
        self.data_file = data_file
        self.is_training = is_training

    def get_vector_datas(self, is_notebook, type_sum, top_number = 50, ):
        """
        type_sum values : gene_var, best_sim, clean_art, sim_gene_var
        top_number only use with best_sim or sim_gene_var
        """
        print("________________________Getting datas vectors_______________________")
        print("____________________________________________________________________")
        start_time = time.perf_counter()

        all_datas = Articles(self.data_file, self.is_training)
        datas_df = all_datas.datas.drop(['ID', 'Score'], axis=1)
        if type_sum != "clean_art":
            datas_df = datas_df.drop(['Text'], axis=1)
        datas_df["Gene"] = datas_df["Gene"].str.lower()
        datas_df["Variation"] = datas_df["Variation"].str.lower()
        size_datas_df = len(datas_df)
        prog = 0
        new_data = []
        not_found_var_list =[]
        not_found_art_list =[]
        sum_vec = []

        log_file_path = ""
        
        if self.is_training:
            datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum', 'Class'])
        else:
            datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum'])

        for _, row in datas_df.iterrows():
            self.progress(prog, size_datas_df, "Getting datas vectors")
            var_clean=""
            gene_clean = clean_word(row['Gene'])
            vec_var =  np.full((1, 100), 0)

            try:
                vec_gene = self.model.wv.get_vector(gene_clean)
            except:
                line = row
                line['Target'] = var_clean
                not_found_var_list.append(line)
                prog += 1
                continue
            
            var_list = row['Variation'].split(" ")
            vect_var_clean = []

            for ind_var in range(len(var_list)):
                var_clean = clean_word(var_list[ind_var])
                try:
                    vec_var = vec_var + self.model.wv.get_vector(var_clean)
                    vect_var_clean.append(var_clean)
                except:
                    line = row
                    line['Target'] = var_clean
                    not_found_var_list.append(line)
                    continue


            if type_sum == "gene_var":
                sum_vec = vec_gene + vec_var

            if type_sum == "best_sim" or type_sum == "sim_gene_var":
                df_bool_gene = (datas_df["Gene"] == row["Gene"])
                vect_neg_clean = []
                sum_best_sim = np.full((1, 100), 0)
                for ind_df, bool_val in enumerate(df_bool_gene):
                    if bool_val and (datas_df.loc[ind_df]["Variation"] != row['Variation']):
                        neg_list = datas_df.loc[ind_df]["Variation"].split(" ")
                        for ind_neg_var in range(len(neg_list)):
                            neg_var_clean = clean_word(neg_list[ind_neg_var])
                            try:
                                vec_gene = self.model.wv.get_vector(neg_var_clean)
                                vect_neg_clean.append(neg_var_clean)
                            except:
                                continue
                if type_sum == "sim_gene_var":
                    vect_var_clean.append(gene_clean)
                if vect_var_clean:
                    most_simlar = self.model.wv.most_similar(positive = vect_var_clean, negative = vect_neg_clean, topn = top_number)
                for sim_word in most_simlar :
                    sum_best_sim = sum_best_sim + self.model.wv.get_vector(sim_word[0])
                sum_vec = sum_best_sim


            if type_sum == "clean_art":
                art_list = row['Text'].split(" ")
            
                art_clean = clean_word(art_list[0])
                vec_art =  np.full((1, 100), 0)

                for ind_art in range(len(art_list)):
                    art_clean = clean_word(art_list[ind_art])
                    try:
                        vec_art = vec_art + self.model.wv.get_vector(art_clean)
                    except:
                        line = row
                        line['Target'] = art_clean
                        not_found_art_list.append(line)
                        continue
                sum_vec = vec_art

            prog += 1
            if self.is_training:
                new_data.append([vec_gene, vec_var, sum_vec, row['Class']])
            else:
                new_data.append([vec_gene, vec_var, sum_vec])

            

        split_filename = self.model_path.split("/")
        modele_name = split_filename[len(split_filename) - 1]
        if is_notebook:
            log_file_path = "../"
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
            
        print("Number of genes/variations not foud : " + str(len(not_found_var)))
        if type_sum == "clean_art":
            print("Number of article words not foud : " + str(len(not_found_art)))
        
        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Getting vectors finished in {} seconds".format(stop_time-start_time))

        self.vectors = datas_vect


    def progress(self, count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '.' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()



def clean_word(word):
    lemmatizer = WordNetLemmatizer()
    clean_var = word.translate(str.maketrans('', '', string.punctuation))
    clean_var = lemmatizer.lemmatize(clean_var)
    return clean_var


def main(data_file, model_path, is_training, is_notebook):
    """
        type_sum values : gene_var, best_sim, clean_art, sim_gene_var
        top_number only use with best_sim or sim_gene_var
    """
    datas = Vector(data_file, model_path, is_training)
    datas.get_vector_datas(is_notebook,"sim_gene_var")
    print (datas.vectors['Sum'])
    
    



#main("datas/training_clean", "datas/cbow_A3316_WS20_E20_B10000_R2000_CTrue.model", True, False)
#main("datas/training_clean", "datas/cbow_3284.model", True, False)
#main("datas/training_clean", "datas/skipgram_3284.model", True, False)
#main("datas/test_clean", "results/Copy_cbow_A701_WS20_E15_B10000_R200_CTrue.model", False, False)

