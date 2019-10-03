import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from sklearn.utils import shuffle
import pickle
from keras.utils import to_categorical
import random
from IntegratedGradients import *

input_sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
dict_label = {"B cells":0,"CD14+ Monocytes":1,"CD4+ T cells":2,"CD8+ T cells":3,"Dendritic cells":4,"NK cells":5,"Other":6}

def out_label():
    """Return the inital labels
    """
    return dict_label

def process_reference(reference_bed):
    """process the reference bed file, if altered also please
    change the model.py chr_list, input a reference such as
    hg19.sorted.bed.
    """
    res = {}
    with open(reference_bed,"r") as f:
        for i in f.readlines():
            container = i.rstrip().split("\t")
            if container[0] not in res.keys():
                tmp = []
                tmp.append(container[1])
                res[container[0]] = tmp
            else:
                res[container[0]].append(container[1])
    return res

class Generator(Sequence):
    """A generator for providing objects for model.fit_generator
    """
    # Class is a dataset wrapper for better training performance
    def __init__(self,x_set,y_set,validation=False,batch_size=34):
        self.x,self.y = x_set,np.array(y_set)
        self.batch_size = batch_size
        self.indices = np.arange(self.x[0].shape[0])
        self.validation = validation

    def __len__(self):
        return int(np.floor(self.x[0].shape[0] / self.batch_size))

    def __getitem__(self,idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.x[i][inds].reshape(self.batch_size,-1,1) for i in range(len(self.x))]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        if not self.validation:
            np.random.shuffle(self.indices)

def Ten_fold_validation_split(training_path):
    """split a tsv file with gene by sample to 10 folds and
    save with pickle.
    """
    print("reading tsv")
    df = pd.read_csv(training_path,sep="\t",header=0,index_col=0)
    df = shuffle(df)
    samples = len(df.index)
    batch_size = int(np.floor(samples / 10))
    for i in range(10):
        out = df.iloc[i*batch_size:(i+1)*batch_size,:]
        out.to_pickle("fold_{}.pkl".format(i))

def folding_Generator(index,batch_size,reference_bed,categorical=False):
    """Convert fold pickles in the current directory for fit_generator.
    """
    training = []
    validation = None
    for i in range(10):
        out = pd.read_pickle("fold_{}.pkl".format(i))
        if i == int(index):
            validation = out
        else:
            training.append(out)
    training = pd.concat(training)
    print("processing labels")
    dict_label = out_label()
    print("processing reference")
    dict_reference = process_reference(reference_bed)
    print("reordering")
    t_x,t_y = reorder(training,dict_reference,dict_label)
    v_x,v_y = reorder(validation,dict_reference,dict_label)
    if categorical:
        t_y = to_categorical(t_y)
        v_y = to_categorical(v_y)
    return Generator(t_x,t_y,validation=False,batch_size=batch_size),Generator(v_x,v_y,validation=True,batch_size=batch_size)

def split_df(df,percentage=0.9):
    """Random splitting dataframe according to a percentage.
    """
    table = {}
    gb = df.groupby(df.index)
    unique = set(df.index)
    for i in unique:
        table[i] = gb.get_group(i)
    training = []
    validation = []
    for i in unique:
        index = np.floor(percentage * table[i].shape[0]).astype(np.int64)
        training.append(table[i].iloc[:index,:])
        validation.append(table[i].iloc[index:,:])
    training_res = pd.concat(training)
    validation_res = pd.concat(validation)
    indices = random.sample(range(training_res.shape[0]),training_res.shape[0])
    training_res = training_res.iloc[indices,:]
    indices = random.sample(range(validation_res.shape[0]),validation_res.shape[0])
    validation_res = validation_res.iloc[indices,:]
    return training_res,validation_res

def reorder(df,dr,dl,test=False):
    """Reordering dataframe to fit in fit_generator. If test, the
    will output the original labels, else will output the labels
    according to dict_label's value(int).
    """
    samples = df.shape[0]
    x_set = [np.zeros((samples,len(dr[i]))) for i in input_sequence]
    y_set = []
    for h,i in enumerate(input_sequence):
        for k,j in enumerate(dr[i]):
            if j in df.columns:
                x_set[h][:,k] = np.copy(df[j])
    if test == True:
        for i in df.index:
            y_set.append(i)
        return x_set,y_set
    for i in df.index:
        y_set.append(dl[i])
    return x_set,y_set

def TrainBatchGenerator(batch_size,training_path,reference_bed,categorical=False,number=None):
    """Initalize the parameters with batch_size, dataframe directory,
    reference_bed directory. If categorical, the value is converted to
    y to one-hot vectors. If number, the first `number` of set inputted
    will be applied to fit_generator.
    """
    print("processing labels")
    dict_label = out_label()
    print("reading tsv")
    training_set = pd.read_csv(training_path,sep="\t",header=0,index_col=0)
    #training_set,df_mean = pre_processing(training_set)
    if number is not None:
        training_set = training_set.iloc[:int(number),]
    print("processing reference")
    dict_reference = process_reference(reference_bed)
    #need to perform validation split manually
    print("splitting df")
    df_train,df_validation = split_df(training_set,percentage=0.9)
    print("reordering")
    #pdb.set_trace()
    t_x,t_y = reorder(df_train,dict_reference,dict_label)
    v_x,v_y = reorder(df_validation,dict_reference,dict_label)
    if categorical:
        t_y = to_categorical(t_y)
        v_y = to_categorical(v_y)
    return Generator(t_x,t_y,validation=False,batch_size=batch_size),Generator(v_x,v_y,validation=True,batch_size=batch_size)

def pickle_predict(df,reference_bed,model,out_name,dataframe=False,score=False,test=False):
    """Predict with the trained model. The file will be named
     `out_name`. If dataframe, the prediction will be output of
     csv and a pickle file (both containing the same result, a
     pickle file for a quicker input to python). Else, if score,
     the prediction score will be provided. Else, the real labels
     will be provided.
    """
    print("processing labels")
    dict_label = out_label()
    print("reading tsv")
    t_y = None
    if df.endswith("tsv"):
        out = pd.read_csv(df,sep="\t",header=0,index_col=0)
    elif df.endswith("pkl"):
        out = pd.read_pickle(df)
    else:
        raise Exception("Not a valid tsv or pkl file")
    print("processing reference")
    dict_reference = process_reference(reference_bed)
    t_x,t_y = reorder(out,dict_reference,dict_label,test=test)
    batch_x = [t_x[i].reshape(t_x[i].shape[0],-1,1) for i in range(len(t_x))]
    res = model.predict_on_batch(batch_x)
    if dataframe:
        df_res = pd.DataFrame(res)
        df_res.columns = [i[0] for i in dict_label.items()]
        df_res.index = out.index
        df_res.to_csv(out_name + ".csv")
        df_res.to_pickle(out_name + ".pkl")
    else:
        if score:
            with open(out_name,"wb") as w:
                pickle.dump({"true_label":t_y,"predict_label":res},w)
        else:
            res_index = [np.argmax(i) for i in res]
            with open(out_name,"wb") as w:
                pickle.dump({"true_label":t_y,"predict_label":res_index},w)

if __name__ == '__main__':
    pass
