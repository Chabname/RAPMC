import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf




def predict_class(model, text_x):
    """Make prediction and create one hot and normal prediction format"""

    pred = model.predict(text_x)    
    one_hot_pred = []
    class_pred = []
    for n in range(len(pred)):
        temp = np.zeros(9)
        max_arg = np.argmax(pred[n], axis=0)
        temp[max_arg] = 1
        class_pred.append(max_arg+1)
        one_hot_pred.append(temp)
    return one_hot_pred, class_pred

    
def get_sub(one_hot_pred):
    """Get the dataframe in Kaggle format"""

    # Dataframe with class columns
    all_pred = pd.DataFrame(one_hot_pred,
                            columns = ["class1","class2","class3","class4","class5","class6","class7","class8","class9"],
                           dtype=int)

    # Adding ID column
    pred_dtf = pd.concat([pd.DataFrame(all_pred.index, columns = ["ID"]), all_pred], axis = 1)

    # Kaggle need only ID 1:986
    pred_kaggle = pred_dtf.loc[1:986]
    return pred_kaggle

    
def plot_pred(class_pred):
    """Plot the predicted values with normal prediction format (not one hot)"""

    # Matplotlib 3.4.3
    
    plt.figure(figsize=(20,10))
    g = sns.countplot(x = class_pred)
    plt.title("Prediction of class", fontsize = 25)
    g.bar_label(g.containers[0], fontsize = 25)

    plt.xlabel("Class predicted",fontsize = 25)
    plt.ylabel("Count",fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()
    
def kaggle_dtf(model, text_x):
    """Get the dataframe in Kaggle format"""
    one_hot_pred, class_pred = predict_class(model, text_x)
    return get_sub(one_hot_pred)

if __name__ == "__main__":

    # Load test text data (vectorized / emmbedded)

    # Data vectorized with Scibert
    features_kaggle = pd.read_pickle("../../data/array_full_data_scibert_KAGGLE.pkl")

    # Correct shape
    features_kaggle = features_kaggle.values.reshape(features_kaggle.shape[0],features_kaggle.shape[1],1)

    # Load model
    model = tf.keras.models.load_model("../results/SCIBERT.h5")

    # Get class prediction (one hot and normal)
    one_hot_scibert, pred_scibert  = predict_class(model,features_kaggle)

    # print(pred_scibert, one_hot_scibert)

    # Convert the one hot prediction into dataframe compatible with Kaggle format
    dtf = get_sub(one_hot_scibert)
    print(dtf)

    # Plotting the predicted classes
    plot_pred(pred_scibert)

    # Saving the dataframe (without index !)
    # dtf.to_csv("../results/sci_bert_kaggle.csv",index = False)



    # Direct way : 
    # dtf = kaggle_dtf(model, features_kaggle)
    # dtf.to_csv("../results/sci_bert_kaggle.csv",index = False)