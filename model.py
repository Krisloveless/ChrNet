from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model

def ChrNet(pretrained_weights = None):
    input_sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
    # here chr_list is initalized according to hg19.sorted.bed
    # gene size is here; if using another version of reference please change the size of each chromosome
    chr_list = [3203, 2205, 1730, 1364, 1658, 1636, 1431, 1335, 1239, 1267, 1921, 1718, 612, 1099, 1075, 1486, 1873, 645, 2031, 829, 462, 703, 1062, 112]
    multi_inputs = [Input((i,1)) for i in chr_list]
    tmp = [None] * len(input_sequence)
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(4,3,activation="relu",padding="same",name="Conv1_{}".format(input_sequence[k]))(multi_inputs[k])
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(4,3,activation="relu",padding="same",name="Conv2_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(4,3,activation="relu",padding="same",name="Conv3_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
        tmp[k] = Dropout(0.3)(v)
    for k,v in enumerate(tmp):
        tmp[k] = MaxPooling1D(2)(v)
    for k,v in enumerate(tmp):
        tmp[k] = BatchNormalization()(v)
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(8,3,activation="relu",padding="same",name="Conv4_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(8,3,activation="relu",padding="same",name="Conv5_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(8,3,activation="relu",padding="same",name="Conv6_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
        tmp[k] = Dropout(0.3)(v)
    for k,v in enumerate(tmp):
        tmp[k] = MaxPooling1D(2)(v)
    for k,v in enumerate(tmp):
        tmp[k] = BatchNormalization()(v)
    for k,v in enumerate(tmp):
        tmp[k] = Conv1D(16,3,activation="relu",padding="same",name="Conv7_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
         tmp[k] = Conv1D(16,3,activation="relu",padding="same",name="Conv8_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
         tmp[k] = Conv1D(16,3,activation="relu",padding="same",name="Conv9_{}".format(input_sequence[k]))(v)
    for k,v in enumerate(tmp):
        tmp[k] = Dropout(0.3)(v)
    for k,v in enumerate(tmp):
        tmp[k] = MaxPooling1D(2)(v)
    for k,v in enumerate(tmp):
        tmp[k] = BatchNormalization()(v)
    for k,v in enumerate(tmp):
        tmp[k] = Flatten()(v)
    for k,v in enumerate(tmp):
        tmp[k] = Model(inputs=multi_inputs[k],outputs=tmp[k])
    combined = concatenate([i.output for i in tmp])
    combined = Dense(256,activation="relu",name="combined1")(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(256,activation="relu",name="combined2")(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(7,activation="softmax",name="combined3")(combined)
    model = Model(inputs=[i.input for i in tmp],outputs=combined)

    model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights,by_name=True)

    return model

if __name__ == '__main__':
    model = ChrNet()
    model.summary()
    plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
