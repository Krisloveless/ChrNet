import os
from data import *
from model import *
from keras.callbacks import ModelCheckpoint,CSVLogger
import pickle

def save(name,c_object):
    with open(name,"wb") as saving:
        pickle.dump(c_object,saving)

def load(name):
    with open(name,"rb") as loading:
        res = pickle.load(loading)
    return res

def run(weight=None,cpu=False):
    if cpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #force cpu
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    # gather the training and validation set.
    train,validation = TrainBatchGenerator(40,"./dataset/training_tpm.tsv","./reference/hg19.sorted.bed")
    model = ChrNet()
    # the generators can be save and load for reuse propose.
    #save("training.pkl",train)
    #save("validation.pkl",validation)
    #train = load("training.pkl")
    #validation = load("validation.pkl")
    model_checkpoint = ModelCheckpoint("ChrNet.hdf5", monitor="val_acc",verbose=1,save_best_only=True)
    csv_logger = CSVLogger("training_ChrNet.log")
    model.fit_generator(train,steps_per_epoch=900,epochs=50,callbacks=[csv_logger,model_checkpoint],validation_data=validation,validation_steps=200)

def predict(file,model_path,reference,outname,cpu=False):
    if cpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #force cpu
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    model = ChrNet(model_path)
    pickle_predict(file,reference,model,outname)

if __name__ == '__main__':
    run()
    #predict(".dataset/testing_tpm.tsv","./pre_trained/ChrNet.hdf5","./reference/hg19.sorted.bed","ChrNet_prediction",cpu=True)
