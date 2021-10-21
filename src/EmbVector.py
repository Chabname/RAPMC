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
        """ Initialize Low matrix before completing

        Keyword arguments:
                data_file -- 
        """
        self.model_path = model_path
        self.model = Word2Vec.load(model_path)
        self.data_file = data_file
        self.is_training = is_training

    def get_vector_datas(self):
        print("________________________Getting datas vectors_______________________")
        print("____________________________________________________________________")
        start_time = time.perf_counter()

        all_datas = Articles(self.data_file, self.is_training)
        datas_df = all_datas.datas.drop(['ID', 'Text', 'Score'], axis=1)
        datas_df["Gene"] = datas_df["Gene"].str.lower()
        datas_df["Variation"] = datas_df["Variation"].str.lower()
        size_datas_df = len(datas_df)
        prog = 0
        new_data = []
        not_found_var_list =[]
        
        if self.is_training:
            datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum', 'Class'])
        else:
            datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum'])

        for _, row in datas_df.iterrows():
            self.progress(prog, size_datas_df, "Getting datas vectors")
            var_clean=""
            gene_clean = clean_word(row['Gene'])

            try:
                vec_gene = self.model.wv.get_vector(gene_clean)
            except:
                line = row
                line['Target'] = var_clean
                not_found_var_list.append(line)
                prog += 1
                continue

            var_list = row['Variation'].split(" ")
            
            var_clean = clean_word(var_list[0])

            try:
                vec_var = self.model.wv.get_vector(var_clean)
            except:
                line = row
                line['Target'] = var_clean
                not_found_var_list.append(line)
                prog += 1
                continue

            if(len(var_list)>1):
                for ind_var in range(1, len(var_list)):
                    var_clean = clean_word(var_list[ind_var])
                    try:
                        vec_var = vec_var + self.model.wv.get_vector(var_clean)
                    except:
                        line = row
                        line['Target'] = var_clean
                        not_found_var_list.append(line)
                        prog += 1
                        continue
            if self.is_training:
                new_data.append([vec_gene, vec_var, vec_gene + vec_var, row['Class']])
            else:
                new_data.append([vec_gene, vec_var, vec_gene + vec_var])
            prog += 1

            

        split_filename = self.model_path.split("/")
        modele_name = split_filename[len(split_filename) - 1]
        log_file_path = "results/log_not_found_words_" + modele_name + ".txt"

        if self.is_training:
            not_found_var = pd.DataFrame(not_found_var_list, columns =['Gene', 'Variation', 'Class', 'Target'])
            np.savetxt(log_file_path, 
                        not_found_var, 
                        fmt = "%s\t\t\t%s\t\t\t%d\t\t\t%s", 
                        header= "\t\t\t".join(not_found_var.columns), 
                        comments='')
            datas_vect = pd.DataFrame(new_data, columns =['Gene', 'Variation', 'Sum', 'Class'])
        else:
            not_found_var = pd.DataFrame(not_found_var_list, columns =['Gene', 'Variation', 'Target'])
            datas_vect = pd.DataFrame(new_data, columns =['Gene', 'Variation', 'Sum'])
            np.savetxt(log_file_path, 
                        not_found_var, 
                        fmt = "%s\t\t\t%s\t\t\t%s", 
                        header= "\t\t\t".join(not_found_var.columns), 
                        comments='')
            
        print(len(not_found_var))
        
        stop_time = time.perf_counter()
        print("____________________________________________________________________")
        print("Getting vectors finished in {} seconds".format(stop_time-start_time))

        return datas_vect


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


def main(data_file, model_path, is_training):
    vector = Vector(data_file, model_path, is_training)
    vect = vector.get_vector_datas()
    
    

#main("datas/training_clean", "datas/cbow_A3316_WS20_E20_B10000_R2000_CTrue.model", True)
#main("datas/training_clean", "datas/cbow_3284.model", True)
#main("datas/training_clean", "datas/skipgram_3284.model", True)
#main("datas/test_clean", "results/Copy_cbow_A701_WS20_E15_B10000_R200_CTrue.model", False)