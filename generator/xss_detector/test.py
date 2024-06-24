import time
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten, Input
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from processing import build_dataset
import numpy as np
from utils import init_session
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
init_session()

batch_size = 500
epochs_num = 1
#log_dir = "log/MLP.log/"
model_dir = "G:/file/MLP_model.h5"

def test(model_dir,test_generator,test_size,input_num,dims_num,batch_size):
    model=load_model(model_dir)
    labels_pre=[]
    labels_true=[]
    batch_num=test_size//batch_size+1
    steps=0
    for batch,labels in test_generator:
        if len(labels)==batch_size:
            labels_pre.extend(model.predict_on_batch(batch))
        else:
            batch=np.concatenate((batch,np.zeros((batch_size-len(labels),input_num,dims_num))))
            labels_pre.extend(model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps+=1
        print("%d/%d batch"%(steps,batch_num))
    labels_pre=np.array(labels_pre).round()
    def to_y(labels):
        y=[]
        for i in range(len(labels)):
            if labels[i][0]==1:
                y.append(0)
            else:
                y.append(1)
        return y
    y_true=to_y(labels_true)
    y_pre=to_y(labels_pre)
    precision=precision_score(y_true,y_pre)
    recall=recall_score(y_true,y_pre)
    print("Precision score is :",precision)
    print("Recall score is :",recall)

if __name__ == "__main__":
    train_generator, test_generator, train_size, test_size, input_num, dims_num = build_dataset(batch_size)
    
    test(model_dir, test_generator, test_size, input_num, dims_num, batch_size)