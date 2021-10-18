
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re

import warnings

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import transformers as ppb
import matplotlib.pyplot as plt


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

    return data



def get_data(data_file):
    
    dtf = pd.read_csv(data_file, sep = "\|\|", engine = "python")
    X = pre_processing(dtf["Text"])
    dataset = dtf.drop(columns = ["Score"], axis = 0)
    return X, dataset


def lemm(text):
    lemmatizer = WordNetLemmatizer()
    temp = []
    list_words = list(set(word_tokenize(text)))
    for word in list_words:
        temp.append(lemmatizer.lemmatize(word))
    return list(set(temp))
            

def get_features(article, tokenizer, model):
    token = []
    max_len = 0
    
    for chunk in lemm(article):
        token.append(tokenizer.encode(chunk, add_special_tokens=True))
        
    for i in token:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in token])
        
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)


    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        
    features = last_hidden_states[0][:,0,:].numpy()
    
    print("SHAPE" ,features.shape)
    tsne = PCA(n_components=50)
    tsne_results = tsne.fit_transform(features.T)
    
    return tsne_results.T.flatten()

def do_features_parallel(all_articles, args):

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    start = time.time()
    all_vec = Parallel(n_jobs= args.cpu, verbose = 0, prefer="processes")(delayed(get_features)
    (article, tokenizer, model) for article in all_articles)
    print("TIME : ",time.time() - start)
    final_dtf = pd.DataFrame(list(all_vec))
    return final_dtf

# def testing_parallel(args):
#     all_articles, dataset = get_data(args.data_file)

#     dtf = do_features_parallel(all_articles.head(100), args)
#     print(dtf)

def model(args):
    clean_text, clean_dtf = get_data(args.data_file)

    labels = pd.get_dummies(clean_dtf['Class'].head(10)).values
    features = do_features_parallel(clean_text.head(10), args)
    features.to_csv("../../datas/tsne_data.csv")
    XD_train, XD_test, YD_train, YD_test = train_test_split(features,
     labels, test_size = 0.2, random_state = 42)

    print(XD_train.shape, YD_train.shape)
    XD_train = XD_train.values.reshape(XD_train.shape[0],XD_train.shape[1],1)
    # YD_train = YD_train.reshape(YD_train.shape[0],7)

    model = Sequential([
    layers.Conv1D(32, kernel_size = [3], activation = "relu", input_shape = (XD_train.shape[1],1)),
    layers.MaxPooling1D(pool_size = [2]),
    layers.Conv1D(64, kernel_size = [3], activation = "relu"),
    layers.MaxPooling1D(pool_size = [2]),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(units = YD_train.shape[1],  activation = "softmax")      
    ])
    model.summary()



    model.compile(
    loss = 'categorical_crossentropy',
    optimizer = "adam",
    metrics = ["accuracy"]
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint('../../results/model_BERT.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    train_sh = model.fit(
        XD_train, YD_train,
        validation_split=0.2,
        epochs=35,
        callbacks=[checkpoint,earlystopping],
        batch_size=32,
        verbose=1
    )


    
    plt.figure(1)
    plt.plot(train_sh.history['loss'])
    plt.plot(train_sh.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig("../../results/loss_plot.PNG")

    plt.figure(1).clear()
    plt.plot(train_sh.history['accuracy'])
    plt.plot(train_sh.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("../../results/acc_plot.jpg")




if __name__ == "__main__":
    configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    session = tf.compat.v1.Session(config=configuration)    
    data_file = "../../datas/all_data_clean.txt"
    model(data_file)