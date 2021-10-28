
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.model_selection import train_test_split
import re

import warnings
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine.input_layer import Input

from tensorflow.python.keras.layers.core import Dropout, Flatten
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.layers.wrappers import Bidirectional
from transformers.utils.dummy_pt_objects import Conv1D, NoBadWordsLogitsProcessor

warnings.filterwarnings('ignore')

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download("stopwords")
from nltk.corpus import stopwords
import time
import torch
from keras.preprocessing.text import Tokenizer

import os
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed

import transformers as ppb
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import plotly.express as px

def plot_matrices(model, x_test, y_test): 
    probas = model.predict(x_test)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.array(range(1, 10))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], probas)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))

    confusion  = confusion_matrix(classes[np.argmax(y_test, axis=1)], preds)
    precision =(confusion/confusion.sum(axis=0))
    recall =(((confusion.T)/(confusion.sum(axis=1))).T)
    f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,0.08]},figsize=(22,6))

    labels = [1,2,3,4,5,6,7,8,9]
    g1 = sns.heatmap(confusion,cbar=False,ax=ax1,annot=True, cmap="Blues", fmt=".3f", xticklabels=labels, yticklabels=labels,)
    g1.set_ylabel('Original Class')
    g1.set_xlabel('Predicted Class')
    g1.set_title('Confusion')
    g2 = sns.heatmap(precision,cmap="Blues",cbar=False,ax=ax2, annot=True,fmt=".3f", xticklabels=labels, yticklabels=labels)
    g2.set_ylabel('Original Class')
    g2.set_xlabel('Predicted Class')
    g2.set_yticks(labels)
    g2.set_title('Precision')
    g3 = sns.heatmap(recall,cmap="Blues",ax=ax3, cbar_ax=axcb, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)
    g3.set_ylabel('Original Class')
    g3.set_xlabel('Predicted Class')
    g3.set_title('Recall')
    g3.set_yticks(labels)

    for ax in [g1,g2,g3]:
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl, rotation=0)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly, rotation=0)

    plt.savefig(("confusion_scibert.PNG"))
    plt.show()

def pre_processing(data):
    sw = stopwords.words("english")
    # lowercase text
    data = data.apply(lambda x: " ".join(i.lower() for i in  str(x).split()))
#     # remove numeric values
#     data = data.str.replace("\d","")
#     # remove punctuations
#     data = data.str.replace("[^\w\s]","")
    # remove stopwords: the,a,an etc.
    data = data.apply(lambda x: " ".join(i for i in x.split() if i not in sw))
    data = data.apply(lambda x: re.sub("⇓","",x))
    data = data.apply(lambda x: re.sub(",|\)|\(|\.","",x))
    return data



def get_data(data_file):
    
    dtf = pd.read_csv(data_file, sep = "\|\|", engine = "python")
    X = pre_processing(dtf["Text"])
    # dataset = dtf.drop(columns = ["Score"], axis = 0)
    return X, dtf


def lemm(text):
    lemmatizer = WordNetLemmatizer()
    temp = []
    list_words = list(set(word_tokenize(text)))
    for word in list_words:
        if len(word) > 2:
            temp.append(lemmatizer.lemmatize(word))
    return list(set(temp))



def get_chunks(text_split):
    length_text = len(text_split)
    size = int(len(text_split) / 20)
    final = []
    for i in range(0, length_text, size):
        final += [" ".join(text_split[i:i+size])]
    return final

def word_vec(word, tokenizer, model):

    features = []
    for w in word.split():
        print(w)
        max_len = 0
        token = []
        token.append(tokenizer.encode(w, add_special_tokens=True))
        for i in token:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0]*(max_len-len(i)) for i in token])
        attention_mask = np.where(padded != 0, 1, 0)    

        input_ids = torch.tensor(padded)  
        attention_mask = torch.tensor(attention_mask)
        
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features.append(last_hidden_states[0][:,0,:].numpy()[0])
    array = np.array(sum(features))
    array = array.reshape(array.shape[0],1)
    print("SHAPE", array.shape)
    return array.flatten()

def get_features(article, tokenizer, model):
    nb_chunk = 20
    chunked_article = get_chunks(article)

    features = []
    for c in range(nb_chunk):
        token = []
        max_len = 0
        token.append(tokenizer.encode(chunked_article[c], add_special_tokens=True))
        for i in token:
            if len(i) > max_len:
                max_len = len(i)


        padded = np.array([i + [0]*(max_len-len(i)) for i in token])

        if len(padded[0]) > 500:
            padded = np.array([padded[0][:500]])
            if padded[0][-1] != 103:
                padded = np.append(padded[0],(103))

            try:
                padded.shape[1]
            except:
                padded = padded.reshape(1, padded.shape[0])
            
        input_ids = torch.tensor(padded)  
        attention_mask = np.where(padded != 0, 1, 0)                       
        attention_mask = torch.tensor(attention_mask)
        try:
            attention_mask.shape[1]
        except:
            attention_mask = attention_mask.reshape(1, attention_mask.shape[0])

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features.append(last_hidden_states[0][:,0,:].numpy()[0])
    array = np.array(features)
    print("SHAPE", array.shape)
    return array.flatten()

def get_outlier(line, tokenizer):
    nb_chunk = 20
    if len(line) < 20:
        return 0
    return 1
    #     chunked_article = get_chunks(line["Text"].split())

    #     for c in range(nb_chunk):
    #         token = []
    #         max_len = 0
    #         token.append(tokenizer.encode(chunked_article[c], add_special_tokens=True))
    #         for i in token:
    #             if len(i) > max_len:
    #                 max_len = len(i)

    #         padded = np.array([i + [0]*(max_len-len(i)) for i in token])
    #         if len(padded[0]) > 500:
    #             padded = np.array([padded[0][:500]])
    #             if padded[0][-1] != 103:
    #                 padded = np.append(padded[0],(103))
            
                
    #         attention_mask = np.where(padded != 0, 1, 0)                       
    #         attention_mask = torch.tensor(attention_mask)
    #         try:
    #             attention_mask.shape[1]
    #         except:
    #             attention_mask = attention_mask.reshape(1, attention_mask.shape[0])
            
    #     if attention_mask.shape[1] > 512:
    #         return 2
    #     return 1
    # return 0

def do_features_parallel(all_articles, args):

    # #model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    # # Load pretrained model/tokenizer
    # tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    # model = model_class.from_pretrained(pretrained_weights)
    tokenizer_scibert = ppb.AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model_scibert = ppb.AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
    # out = all_articles.apply(lambda ar: get_outlier(ar, tokenizer_scibert))
    # to_keep = out[out == 1].index
    
    start = time.time()
    all_vec = Parallel(n_jobs= args.cpu, verbose = 0, prefer="processes")(delayed(get_features)
    (article, tokenizer_scibert, model_scibert) for article in all_articles)
    print("TIME : ",time.time() - start)
    final_dtf = pd.DataFrame(list(all_vec))
    return final_dtf


def get_input_data(args):
    clean_text, clean_dtf = get_data(args.data_file)
    print(clean_dtf)
    print(clean_text)

    lemmatized_text = clean_text.apply(lambda x : lemm(x))
    clean_dtf["Text"] = lemmatized_text
    print(len(lemmatized_text[0]))

    features = do_features_parallel(clean_dtf["Text"], args)
    labels = pd.get_dummies(clean_dtf.loc[features.index,'Class']).values

    features.to_pickle("../../../data/array_full_data_scibert.pkl")
    np.save("../../../data/full_labels_scibert.npy",labels)
    return features, labels

def model(args):
    # features, labels = get_input_data(args)

    
    features = pd.read_pickle("../../../data/array_full_data_scibert_nolem_p4.pkl")
    labels = np.load("../../../data/full_labels_scibert_nolem.npy")
    # pca = PCA(n_components = 0.8)
    # df_merge = pd.read_csv("../../datas/all_data_clean.txt",sep="\|\|", engine="python")
    # df_merge.head()
    # best_text = df_merge[df_merge["Score"] == 1]
    # print(features.shape)
    # features = features.loc[best_text.loc[:3253].index]
    # y = df_merge.loc[best_text.loc[:3253].index, "Class"]
    # labels = pd.get_dummies(y).values

    # pca.fit(features)
    # exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    # # plot_var = px.area(
    # #     title = "Variance explained for non lemmatized full article",
    # #     x=range(1, exp_var_cumul.shape[0] + 1),
    # #     y=exp_var_cumul,
    # #     labels={"x": "# Components", "y": "Explained Variance"}
    # # )
    # # plot_var.show()
    # pca_results = pca.fit_transform(features)

    # # print(pca_results.shape)

    # features = pd.DataFrame(pca_results)
    # # Load data

    # features = pd.read_pickle("../../../data/features_full_pca_500.pkl")
    # features = pd.read_pickle("../../../data/array_full_data_scibert.pkl")
    # labels = np.load("../../../data/full_labels_scibert.npy")
    

    # y_class = []
    # for x in labels:
    #     y_class.append(np.argmax(x) + 1)

    # class_weights = dict(zip(np.unique(y_class)-1, class_weight.compute_class_weight(class_weight = 'balanced',
                                                                            #    classes = np.unique(y_class), y = y_class)))
    # Shapes
    XD_train, XD_test, YD_train, YD_test = train_test_split(features,
     labels, test_size = 0.2, stratify=labels)
    # Select correct data
   
    # Test train split
    train_df, cv_df, y_train, y_cv = train_test_split(XD_train, YD_train, stratify=YD_train, test_size=0.2)

    # Oversampling
    oversample = RandomOverSampler(sampling_strategy='minority')
    train_df, y_train = oversample.fit_resample(train_df, y_train)
    train_df, y_train = oversample.fit_resample(train_df, y_train)
 
    XD_train = XD_train.values.reshape(XD_train.shape[0],XD_train.shape[1],1)

    XD_test = XD_test.values.reshape(XD_test.shape[0],XD_test.shape[1],1)

    train_df = train_df.values.reshape(train_df.shape[0],train_df.shape[1],1)
    cv_df = cv_df.values.reshape(cv_df.shape[0],cv_df.shape[1],1)
    
    # print(XD_train.shape, YD_train.shape)
    # print(gene_train.shape, var_train.shape)


    # print(train_df.shape, y_train.shape)

    # YD_train = YD_train.reshape(YD_train.shape[0],7)

    model = Sequential(
    [
        layers.Input(shape =(XD_train.shape[1],1)),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(9, activation='softmax')
    ]
    )



    model.summary()
    # Best models with oversampling: 

    #    model = Sequential(
    # [
    #     layers.Input(shape =(XD_train.shape[1],1)),
    #     layers.Conv1D(64, 5, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.MaxPooling1D(2),
    #     layers.Dropout(0.5),

    #     layers.Conv1D(64, 5, activation='relu'),
    #     layers.MaxPooling1D(2),
    #     layers.Dropout(0.5),
    #     layers.Flatten(),
    #     layers.Dense(9, activation='softmax')
    # ]
    # )

    model.compile(  
    loss = 'categorical_crossentropy',
    optimizer = keras.optimizers.Adam(0.0001),
    metrics = ["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(('model_SCIBERT.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    train_sh = model.fit(
        train_df, y_train,
        # XD_train, YD_train,
        epochs=35,
        callbacks=[checkpoint,earlystopping],
        batch_size=64,
        validation_data = (cv_df, y_cv),
        # validation_split=0.2,
        # class_weight = class_weights,
        verbose=1

    )
    # model.save("SCIBERT.h5")    

    loss, accuracy = model.evaluate(XD_train, YD_train, verbose=False)
    print("Training Accuracy: > %.3f"  % (accuracy * 100.0))
    print("loss : ",loss)

    loss, accuracy = model.evaluate(XD_test, YD_test, verbose=False)
    print("Testing Accuracy:  > %.3f"  % (accuracy * 100.0))
    print("loss : ",loss)
    plot_matrices(model, XD_test, YD_test)

    plt.figure(1)
    plt.plot(train_sh.history['loss'])
    plt.plot(train_sh.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig(("loss_plot_scibert.PNG"))
    plt.figure(1).clear()

    plt.plot(train_sh.history['accuracy'])
    plt.plot(train_sh.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(("acc_plot_scibert.jpg"))
    plt.figure(1).clear()




if __name__ == "__main__":
    configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    session = tf.compat.v1.Session(config=configuration)    
    data_file = "../../datas/all_data_clean_011.txt"
    model(data_file)