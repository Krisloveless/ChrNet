import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from sklearn.utils import shuffle
import pickle
from keras.utils import to_categorical
import random
from IntegratedGradients import *
from itertools import chain
from scipy import stats
from statsmodels.stats.multitest import multipletests

input_sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
# dict_label can be changed here for customized training
dict_label = {"B cells":0,"CD14+ Monocytes":1,"CD4+ T cells":2,"CD8+ T cells":3,"Dendritic cells":4,"NK cells":5,"Other":6}

def out_label():
    """Return the inital labels.
    """
    return dict_label

def process_reference(reference_bed):
    """Process the reference bed file, if altered also please
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
    """Split a tsv file with gene by sample to 10 folds and
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
    will be applied to fit_generator. Input training file should be
    a tsv file where rows should be cells and columns should be genes
    , first column indicating the cell type and first row indicating gene
    name (Ensembl ID).
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
     pickle file for a quicker input to python). If score,
     the prediction score will be provided. If test, the real labels
     will be provided. Input file should be a tsv file or pkl a pkl
     that is holding a pandas dataframe where rows should be cells
     and columns should be genes.
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

def IG_reorder(input_batch):
    """A function for reordering into the right reorder
    that is accepted by intergrated gradients.
    """
    out = []
    samples = input_batch[0].shape[0]
    for i in range(samples):
        tmp = []
        for j in input_batch:
            tmp.append(j[i,:,:])
        out.append(tmp)
    return out

def pickle_predict_with_IG(df,reference_bed,model,out_name):
    """Prediction with intergrated gradients. Input should be a
    tsv or a pkl that is holding a pandas dataframe. Output csv
    is the prediction result, while pkl file is the attribute of
    different cell types.
    """
    print("processing labels")
    dict_label = out_label()
    print("reading csv")
    t_y = None
    if df.endswith("tsv"):
        out = pd.read_csv(df,sep="\t",header=0,index_col=0)
    elif df.endswith("pkl"):
        out = pd.read_pickle(df)
    else:
        raise Exception("Not a valid tsv or pkl file")
    print("processing reference")
    dict_reference = process_reference(reference_bed)
    t_x,t_y = reorder(out,dict_reference,dict_label,test=True)
    batch_x = [t_x[i].reshape(t_x[i].shape[0],-1,1) for i in range(len(t_x))]
    res = model.predict_on_batch(batch_x)
    ig = integrated_gradients(model)
    res_index = [np.argmax(i) for i in res]
    IG_batch = IG_reorder(batch_x)
    attribute = [ig.explain(i) for i in IG_batch]
    barcode = t_y
    pl = res_index
    value = pd.DataFrame(np.array(barcode,dtype=object).reshape(-1,1))
    prediction = pd.DataFrame(np.array(pl,dtype=object).reshape(-1,1))
    conout = pd.concat([value,prediction],axis=1)
    conout.columns = ["","Cluster"]
    name = os.path.split(out_name)[1].split(".")[0]
    # csv output here
    conout.to_csv("{}.csv".format(name),index=False)
    # resout is sample x gene
    genes = [i.shape[0] for i in attribute[0]]
    resout = np.zeros((len(attribute),sum(genes)))
    resout = pd.DataFrame(resout)
    tmp = [dict_reference[i] for i in input_sequence]
    colnames = [i for i in chain(*tmp)]
    resout.columns = colnames
    resout.index = barcode
    for i in range(len(attribute)):
        tmp = [j[0] for j in chain(*(attribute[i]))]
        resout.iloc[i,] = tmp
    resout.to_pickle("{}_attribute.pkl".format(out_name))

def findMetaFeature(pkl,cluster_csv,out_name,classes=len(dict_label)):
    """Find the metafeatures from the model in each of the
    cell types. Input should be output files from
    pickle_predict_with_IG. classes is the number of class
    for the model, default is 7.
    """
    data = pd.read_pickle(pkl)
    reference = pd.read_csv(cluster_csv,header=0,index_col=0)
    label = reverse_label()
    for i in range(classes):
        outname = "".join(label[i].split(" "))
        current_index = reference[reference["Cluster"] == i].index
        none_index = reference[reference["Cluster"] != i].index
        G_set = abs(data.loc[current_index])
        R_set = abs(data.loc[none_index])
        gene_name_all = data.columns
        MetaName = []
        Pval = []
        for j in range(G_set.shape[1]):
            G_set_single = G_set.iloc[:,j].tolist()
            R_set_single = R_set.iloc[:,j].tolist()
            if G_set_single == R_set_single:
                continue
            P_out = stats.ttest_ind(G_set_single,R_set_single, equal_var=True).pvalue
            if np.isnan(P_out):
                continue
            MetaName.append(gene_name_all[j])
            # two side to one side check for null hypothesis
            P_out = P_out / 2
            Pval.append(P_out)
        if len(Pval) == 0:
            continue
        p_adjusted = multipletests(Pval, alpha=0.05, method='fdr_bh')
        p_adj_value = p_adjusted[1]
        sig = {}
        #print(label[i],":",len(p_adjusted[0][p_adjusted[0]==True]))
        for j in range(len(p_adjusted[0])):
            if p_adjusted[0][j]:
                sig[MetaName[j]] = p_adj_value[j]
        with open("{}_{}_Meta.pkl".format(out_name,outname),"wb") as pkl:
            pickle.dump(sig,pkl)

if __name__ == '__main__':
    pass
