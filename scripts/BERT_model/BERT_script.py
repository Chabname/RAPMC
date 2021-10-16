import tokenization # tokenization.py
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk
import transformers
import matplotlib.pyplot as plt
import re
nltk.download("stopwords")
from nltk.corpus import stopwords



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

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    #clf_output = sequence_output[:, 0, :]
    
    # CNN model
    
    net = tf.keras.layers.Conv1D(124, (5), activation='relu')(sequence_output)
    net = tf.keras.layers.MaxPooling1D(2)(net)
    
#     net = tf.keras.layers.Conv1D(64, (5), activation='relu')(net)
#     net = tf.keras.layers.GlobalMaxPooling1D()(net)
    

    net = tf.keras.layers.Dense(20, activation="relu")(net)
    net = tf.keras.layers.Flatten()(net)    
    net = tf.keras.layers.Dropout(0.1)(net)
    out = tf.keras.layers.Dense(9, activation="softmax")(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_data(data_file):
    
    dtf = pd.read_csv(data_file, sep = "\|\|", engine = "python")
    X = pre_processing(dtf["Text"])
    dataset = dtf.drop(columns = ["Score"], axis = 0)
    return X, dataset


def model(data_file):
    X, dataset = get_data(data_file)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    bert_layer = hub.KerasLayer(m_url, trainable=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    Y_D = pd.get_dummies(dataset['Class']).values
    XD_train, XD_test, YD_train, YD_test = train_test_split(X, Y_D, test_size = 0.2, random_state = 42, stratify=Y_D)
    print(XD_train.shape, YD_train.shape)
    print(XD_test.shape, YD_test.shape)
    
    train_input = bert_encode(XD_train, tokenizer, max_len=500)
    test_input = bert_encode(XD_test, tokenizer, max_len=500)  

    model = build_model(bert_layer, max_len=500)
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../../results/model_bert.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,verbose=1)

    train_sh = model.fit(
        train_input, YD_train,
        validation_split=0.2,
        epochs=20,
        callbacks=[checkpoint, earlystopping],
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
    data_file = "../../../data/data_stat.txt"
    model(data_file)