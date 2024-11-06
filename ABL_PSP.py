import os
import numpy as np
import random
random_seed = 0
print("Selected random seed is : ", random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
import keras # 深都学习框架kears，接口比较简单，是Tensorflow的封装 
import pickle # 模型或数据的保存和加载
import csv
from keras.utils import np_utils # 存放一些不好归类的代码
from keras.models import Sequential # 模型，神经网络
from keras.layers import Layer, Dense, Activation, Conv2D, MaxPooling2D, LayerNormalization, Conv1D,Dropout,Add,Flatten, BatchNormalization,Input,Reshape,MultiHeadAttention
from keras.optimizers import Adam # 优化器
from keras.preprocessing.image import ImageDataGenerator # 数据生成器
import wandb
from tensorflow.keras.layers import Lambda

from PIL import Image # 画图
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
tf.get_logger().setLevel(tf.compat.v1.logging.WARN)
from functools import partial # func(a,b,c) fun1(b,c)
import sys
sys.path.insert(0, './logic/lib/')
sys.path.insert(0, './logic/python/')
from learn_add import *
from learn_add_encoder_decoder import *
import LogicLayer as LL
from keras import optimizers
from models import NN_model # 神经网络
import time
import argparse

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
os.environ["CUDA_VISIBLE_DEVICES"] ='0,1,2,3'
from map_generator import map_generator
seed=0
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# 禁用多线程以确保结果一致性
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
DEBUG = True

# wandb.init(
#     project="ABL",
#     entity="ygzwqzd",save_code=True,
#     # config={
#     # "random_seed": 5,
#     # "num_valid_negative": 100,
#     # "imbalance_algorithm": "logit_adjustment",
#     # "imbalance_ratio": 100,
#     # "base_model":"base_model",
#     # "weight_decay":0,
#     # "valid_iteration":1000,
#     # "epoch":20,
#     # "lr":1e-5,
#     # "batch_size":16,
#     # "tro":2.0,
#     # "scheduler":True,
#     # "augmentation":"synonyms",
#     # "synonyms":5,
#     # "model_path":"seed5.pth"
#     # }
# )

class TransformerEncoderLayer(Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.MultiHeadAttention = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.head_size, dropout=self.dropout
        )
        self.Add_1=Add()
        self.LayerNormalization_1=LayerNormalization()
        self.Conv1D_1=Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.Dropout=Dropout(self.dropout)
        self.Conv1D_2=Conv1D(filters=input_shape[-1], kernel_size=1)
        self.Add_2= Add()
        self.LayerNormalization_2=LayerNormalization()
        # 添加其他需要的层，比如 Conv1D, LayerNormalization等

    def call(self, inputs, training=None):
        x = self.MultiHeadAttention(inputs, inputs)
        # Skip Connection 1
        x = self.Add_1([inputs, x])
        x = self.LayerNormalization_1(x)

        # Feed Forward Part
        x_ff = self.Conv1D_1(x)
        x_ff = self.Dropout(x_ff)
        x_ff = self.Conv1D_2(x_ff)
        # Skip Connection 2
        x_ff = self.Add_2([x, x_ff])
        x_ff = self.LayerNormalization_2(x_ff)
        return x_ff

    def compute_output_shape(self, input_shape):
        return input_shape

def get_img_data(src_path, labels, shape=(28, 28, 1)):
    print("\n** Now getting all images **")
    #image
    X = []
    #label
    Y = []
    h = shape[0]
    w = shape[1]
    d = shape[2]
    #index = [0,1,2,3]
    for (index, label) in enumerate(labels): # [(0,0),(1,1),(2,10),(3,11)]
        label_folder_path = os.path.join(src_path, label)
        for p in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, p)
            if d == 1:
                mode = 'I'
            else:
                mode = 'RGB'
            image = Image.open(image_path).convert(mode).resize((h, w))
            X.append((np.array(image) - 127) * (1 / 128.))
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y) #[000000,11111,222222,33333]

    index = np.array(list(range(len(X))))
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]

    assert (len(X) == len(Y))
    print("Total data size is :", len(X))
    # normalize
    X = X.reshape(-1, h, w, d)
    Y = np_utils.to_categorical(Y, num_classes=len(labels))
    return X, Y

def get_img_data_balance(src_path, labels, shape=(28, 28, 1),num=50):
    print("\n** Now getting all images **")
    #image
    X = []
    #label
    Y = []
    h = shape[0]
    w = shape[1]
    d = shape[2]
    #index = [0,1,2,3]
    for (index, label) in enumerate(labels): # [(0,0),(1,1),(2,10),(3,11)]
        cnt=0
        cur_X=[]
        cur_Y=[]
        label_folder_path = os.path.join(src_path, label)
        for p in os.listdir(label_folder_path):
            cnt+=1
            if cnt>num:
                break
            image_path = os.path.join(label_folder_path, p)
            if d == 1:
                mode = 'I'
            else:
                mode = 'RGB'
            image = Image.open(image_path).convert(mode).resize((h, w))
            cur_X.append((np.array(image) - 127) * (1 / 128.))
            cur_Y.append(index)
        if cnt < num:
            cur_X=cur_X * (num//cnt)+cur_X[:num%cnt]
            cur_Y=cur_Y * (num//cnt)+cur_Y[:num%cnt]
        X.extend(cur_X)
        Y.extend(cur_Y)

    X = np.array(X)
    Y = np.array(Y) #[000000,11111,222222,33333]

    index = np.array(list(range(len(X))))
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]

    assert (len(X) == len(Y))
    print("Total data size is :", len(X))
    # normalize
    X = X.reshape(-1, h, w, d)
    Y = np_utils.to_categorical(Y, num_classes=len(labels))
    return X, Y


def get_nlm_net(labels_num, shape=(28, 28, 1), model_name="LeNet5"):
    assert model_name == "LeNet5"
    #if model_name == "LeNet5":
    #    return NN_model.get_LeNet5_net(labels_num, shape)
    d = shape[2]
    if d == 1:
        return NN_model.get_LeNet5_net(labels_num, shape)
    else:
        return NN_model.get_cifar10_net(labels_num, shape)


def net_model_test(src_path, labels, src_data_name, shape=(28, 28, 1)):
    print("\n** Now use the model to train and test the images **")

    file_name = '%s_correct_model_weights.hdf5' % src_data_name
    if os.path.exists(file_name):
        print("Model file exists, skip model testing step.")
        return
    d = shape[2]
    X, Y = get_img_data(src_path, labels, shape)

    # train:test = 5:1
    X_train = X[:len(X) // 5 * 1]
    y_train = Y[:len(Y) // 5 * 1]
    X_test = X[len(X) // 5 * 1:]
    y_test = Y[len(Y) // 5 * 1:]

    print("Train data size:", len(X_train))
    print("Test data size:", len(X_test))

    model = get_nlm_net(len(labels), shape)
    # Add decay
    opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(optimizer=opt_rms,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Training...')

    if d == 1:
        model.fit(X_train,
                  y_train,
                  epochs=30,
                  batch_size=32,
                  validation_data=(X_test, y_test))
    else:
        #data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),\
          steps_per_epoch=X_train.shape[0]//32,epochs=100,verbose=1,validation_data=(X_test,y_test))
    model.save_weights(file_name)
    print("Model saved to ", file_name)

    print('\nTesting...')
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss: ', loss)
    print('Test accuracy: ', accuracy)


def net_model_pretrain(src_path, labels, src_data_name, shape=(28, 28, 1), pretrain_epochs=10,num=50,balance=False):
    print("\n** Now use autoencoder to pretrain the model **")

    file_name = '%s_pretrain_%d_weights_%d_%s.hdf5' % (src_data_name,pretrain_epochs,num,str(balance))
    h = shape[0]
    w = shape[1]
    d = shape[2]
    if os.path.exists(file_name): #and False:
        print("Pretrain file exists, skip pretrain step.")
        return
    if balance:
        X, _ = get_img_data_balance(src_path, labels, shape, num=num) # labels [0,1,10,11]
    else:
        X, _ = get_img_data(src_path, labels, shape) # labels [0,1,10,11]
    print("There are %d pretrain images" % len(X))

    if d == 1:
        Y = X.copy().reshape(-1, h * w * d)
        model = NN_model.get_LeNet5_autoencoder_net(len(labels), shape)
    else:
        Y = X
        model = NN_model.get_autoencoder_net(len(labels), input_shape=shape)
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    print('Pretraining...')
    model.fit(X, Y, epochs=pretrain_epochs, batch_size=64)
    model.save_weights(file_name)
    print("Model saved to ", file_name)

def net_model_pretrain_sup(src_path, labels, src_data_name, shape=(28, 28, 1), pretrain_epochs=10,num=10,balance=False):
    print("\n** Now use autoencoder to pretrain the model **")

    file_name = '%s_pretrain_%d_weights_sup_%d_%s.hdf5' % (src_data_name,args.pretrain_epochs,num,str(balance))
    h = shape[0]
    w = shape[1]
    d = shape[2]
    if os.path.exists(file_name): #and False:
        print("Pretrain file exists, skip pretrain step.")
        return

    if balance:
        X, Y = get_img_data_balance(src_path, labels, shape, num=num) # labels [0,1,10,11]
    else:
        X, Y = get_img_data(src_path, labels, shape) # labels [0,1,10,11]
    print("There are %d pretrain images" % len(X))

    if d == 1:
        # Y = X.copy().reshape(-1, h * w * d)
        model = NN_model.get_LeNet5_net(len(labels), input_shape=shape)
    # else:
        # Y = X
        # model = NN_model.get_autoencoder_net(len(labels), input_shape=shape)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # X=X[:64]
    # Y=Y[:64]
    print('Y:', Y)
    print('Pretraining...')
    model.fit(X, Y, epochs=pretrain_epochs, batch_size=64)
    model.save_weights(file_name)
    print("Model saved to ", file_name)

def LL_init(pl_file_path):
    print("\n** Initializing prolog **")
    assert os.path.exists(pl_file_path), "%s is not exist" % pl_file_path
    # must initialise prolog engine first!
    LL.init("--quiet --nosignals --stack_limit=10G")
    # consult the background knowledge file
    LL.consult(pl_file_path)
    #test if stack changed
    LL.call("prolog_stack_property(global, limit(X)), writeln(X)")
    print("Prolog has alreadly initialized")


def divide_equation_by_len(equations,equations_str):
    '''
    Divide equations by length
    equations has alreadly been sorted, so just divide it by equation's length
    '''
    equations_by_len = list()
    equations_str_by_len = list()
    start = 0
    for i in range(1, len(equations) + 1):
        #print(len(equations[i]))
        if i == len(equations) or len(equations[i]) != len(equations[start]):
            equations_by_len.append(equations[start:i])
            equations_str_by_len.append(equations_str[start:i])
            start = i
    # print(equations_str)
    return equations_by_len,equations_str_by_len # 5-8 [[[1+0=1],[0+1=1]], [长度是6的式子]]  


def split_equation(equations_by_len, equations_str_by_len,prop_train, prop_val):
    '''
    Split the equations in each length to training and validation data according to the proportion
    '''
    train = []
    train_str=[]
    val = []
    val_str=[]
    all_prop = prop_train + prop_val
    # print(equations_str_by_len)
    for equations, equations_str in zip(equations_by_len,equations_str_by_len):
        # print('type_equations:',type(equations))
        # print('type_equations_0:',type(equations[0]))
        # print('equations_str:',equations_str)
        combined_data = list(zip(equations, equations_str))
        random.shuffle(combined_data)
        equations, equations_str= zip(*combined_data)
        equations=list(equations)
        equations_str=list(equations_str)
        # print('equations_str_af:',equations_str)
        train.append(equations[:len(equations) // all_prop * prop_train])
        val.append(equations[len(equations) // all_prop * prop_train:])
        train_str.append(equations_str[:len(equations_str) // all_prop * prop_train])
        val_str.append(equations_str[len(equations_str) // all_prop * prop_train:])
        #print(len(equations[:len(equations)//all_prop*prop_train]))
        #print(len(equations[len(equations)//all_prop*prop_train:]))
    return train, val,train_str,val_str


def constraint(solution, min_var, max_var):
    '''
    Constrain how many position to abduce
    '''
    x = solution.get_x()
    #print(x)
    return (max_var - x.sum()) * (x.sum() - min_var)




def get_abduced_result(encoder_decoder_model, exs, equations_str,probability,maps, no_change,max_length=8,input_size=4,T=1,budget=10):
    # print('get_abduced_result')
    # print('no_change')
    # print(no_change)
    consist_re = None
    consist_res_max = (-1, [])
    exs_shape = [len(tmpex) for tmpex in exs]
    sol_max=np.array([0] * sum(exs_shape))
    # print('sol_max:',sol_max)
    # print('exs_shape')
    # print(exs_shape)
    # Constrain zoopt modify at least 1 bit and at most 10 bits
    # c = partial(constraint, min_var=1, max_var=10)
    #Try all mappings
    map_cnt=0
    
    for m in maps:
        # map_cnt+=1
        # print('map_cnt:',map_cnt)
        # Check if it can abduce rules without changing any labels
        if no_change:
            # Assuming that each equation is the same length
            # exs [[长为4]，[长为5]，[长为6]]   [15个0]   
            # 0表示不改变
            consist_res = consistent_score_sets(exs, [0] * sum(exs_shape), m)  #consist_res是一个元组包含：0.score= e^(-0)*2+e^(-1)*3 1. 【【0，1，3】 【2，4】】
            # print('consist_res')
            # print(consist_res)  
            if consist_res[0] > consist_res_max[0]:
                consist_res_max = consist_res
        # Find the possible wrong position in symbols and Abduce the right symbol through logic module
        else:
            # Use zoopt to optimize
            # print('pre_encoder_decoder_constraint')
            # print('000')
            st=time.time()
            sol,eval_time = encoder_decoder_constraint(encoder_decoder_model, exs, probability, m,  budget=budget, max_length=max_length,input_size=4,T=T)
            et=time.time()
            # with open('time_test.txt', 'a') as f:
            #     f.write(str(et-st-eval_time)+'\n')
            print(et-st-eval_time)
            # print('666')
            # print('after_encoder_decoder_constraint')
            # opt_var_ids_sets_constraint(exs, m, c)
            # print('sol')
            # print(sol)
            # print('get_x')
            # print(sol.get_x())
            # Get the result
            # print('pre_consistent_score_sets')
            # exs_mapped=[]
            # for i in range(len(exs)):
            #     exs_mapped.append([m[exs[i][j]] for j in range(len(exs[i]))])
            # print('m:',m)
            # print('exs:',exs)
            # print('exs_mapped:',exs_mapped)
            # print('equations_str:',equations_str)
            # print('get_x')
            # print(reform_ids(exs,sol))
            consist_res = consistent_score_sets(exs, [int(i) for i in sol ], m)
            # print('777')
            # print('consist_score')
            # print(consist_res[0])
            # print('after_consistent_score_sets')
            if consist_res[0] > consist_res_max[0]:
                consist_res_max = consist_res
                sol_max=sol

    # consistent_score_sets() returns consist_res=(score, eq_sets). the score is the function value for zoopt, so we only care about the second element when printting result
    max_consist_num = 0
    #Choose the result that has max num of consistent examples
    # print('consist_res_max')
    # print(consist_res_max)
    for re in consist_res_max[1]:
        if len(re.consistent_ex_ids) > max_consist_num:
            max_consist_num = len(re.consistent_ex_ids)
            consist_re = re

    if no_change:
        # print('no_change')
        if max_consist_num == len(exs):
            # print('consistent')
            # print('consist_re')
            # print(consist_re)
            # print("#It can abduce rules without changing any labels")
            # print('check_sol:',np.zeros(len(exs)).shape)
            # print('no_change')
            # print('sol_max:',type(sol_max))
            return consist_re,sol_max,0
        # print('inconsistent')
        return None,None,0
    else:
        # print('change')
        # print('consist_re')
        # print(consist_re)
        # print('sol_max:',type(sol_max))
        return consist_re,sol_max,eval_time


def get_equations_labels(model,encoder_decoder_model,
                         equations,
                         equations_str,
                         labels,
                         abduced_map=None,
                         no_change=False,
                         shape=(28, 28, 1),max_length=8,T=1,budget=10):
    '''
    Get the model's abduced output through abduction
    model: NN model
    equations: equation images
    labels: [0,1,10,11] now  only use len(labels)
    maps: abduced map like [0:'+',1:'=',2:0,3:1] if None, then try all possible mappings
    no_change: if True, it indicates that do not abduce, only get rules from equations
    shape: shape of image
    '''
    # print('get_equations_labels')
    # print('no_chang')
    # print(no_change)
    h = shape[0]
    w = shape[1]
    d = shape[2]
    exs = []
    exs_mapped = []
    probability=[]
    
    # print('len(equations)')
    # print(len(equations))
    for _ in range(len(equations)):
        # print('e')
        # print(len(e))
        # print('e:',e)
        e=equations[_]
        ans=equations_str[_]
        # try:
        prob=model.predict(e.reshape(-1, h, w, d),verbose=0)
        # except:
            # print('异常！')
            # print('e:',e)
            # print('equations:',equations)
            # prob=model.predict(e.reshape(-1, h, w, d),verbose=0)
        _e=np.argmax(prob, axis=1).tolist()
        exs.append(_e)
        # exs_mapped.append([])
        probability.append(prob)
        # print('list')
        # print(np.argmax(model.predict(e.reshape(-1, h, w, d)), axis=1).tolist()) #[2,3,2,3,2,2]
    # if no_change == False:
    #     print("\n\nThis is the model's current label:")
    #     print('exs')
    #     print(exs)
    if abduced_map is None:
        maps = gen_mappings([0, 1, 2, 3], ['+', '=', 0, 1])  # All possible mappings from label to signs(0 1 + =)
        #maps = list(map_generator(['+', '='], 2)) # only consider '+' and '=' mapping
    else:
        maps = [abduced_map]
    # print('e',e)
    # print('prob:',probability)
    # print('map:',maps)
    # 上面得到的是神经网络预测结果，下面要进行反绎
    # Check if it can abduce rules without changing any labels
    consist_re,sol,_= get_abduced_result(encoder_decoder_model,exs,equations_str,probability, maps, True,max_length=max_length,input_size=len(labels),T=T,budget=budget) # 是否可以直接反义出规则

    # print('consist_re')
    # print(consist_re)
    if consist_re is not None:
        sol=reform_ids(equations,sol)
    # print('no_change')
    # print('consist_re:',consist_re)
    # print('sol:',sol)
    if consist_re is None:
        # print('inconsistent without abduce')
        if no_change == True:
            # print('no change and return None')
            return (None, None,None,None,0)
    else:
        # print('consistent without abduce')
        # print('consist_re, consist_re.to_feature().rules')
        # print(consist_re, consist_re.to_feature().rules)
        return (consist_re, consist_re.to_feature().rules.py(),sol,probability,0)
    # print('change and continue to abduce')
    # Find the possible wrong position in symbols and Abduce the right symbol through logic module

    consist_re, sol, eval_time = get_abduced_result(encoder_decoder_model,exs,
                    equations_str, probability, maps, False,max_length=max_length,
                    input_size=len(labels),T=T,budget=budget)
    sol=reform_ids(equations,sol)
    # print('change')
    # print('consist_re:',consist_re)
    # print('sol:',sol)
    if consist_re is None:
        # print('inconsistent with abduce')
        return (None, None, None, None, eval_time)
    # print('consistent with abduce')
    feat = consist_re.to_feature()  # Convert consistent result to add rules my_op, will be used to train the decision MLP
    # print('feat')
    # print(feat)
    rule_set = feat.rules.py()
    # print('rule_set')
    # print(rule_set)
    # if DEBUG:
        # print('****Consistent instance:')
        # print('consistent examples:', end='\t')
        # Max consistent subset's index in original exs list
        # print(consist_re.consistent_ex_ids)
        # print('mapping:', end='\t')
        # Mapping used in abduction
        # print(consist_re.abduced_map)
        # print('abduced examples:', end='\t')
        # Modified label sequence after abduction, will be used to retrain CNN
        # print(consist_re.abduced_exs)
        # print('abduced examples(after mapping):', end='\t')
        # abduced_exs after using mapping
        # print(consist_re.abduced_exs_mapped)

        # print('****Learned feature:')
        # print('rules: ', end='\t')
        # print(rule_set)

    return (consist_re, rule_set, sol, probability, eval_time)


def get_mlp_vector(equation, model, rules, abduced_map, shape=(28, 28, 1)):
    # print('get_mlp_vector')
    h = shape[0]
    w = shape[1]
    d = shape[2]
    model_output = np.argmax(model.predict(equation.reshape(-1, h, w, d),verbose=0), axis=1) #1+0=1 12031
    model_labels = []
    for out in model_output:
        model_labels.append(abduced_map[out]) # 1 10 0 11 1
    #print(model_labels)
    vector = []
    for rule in rules:
        if rule == None:
            vector.append(1)
        else:
        # print('model_labels')
        # print(model_labels)
            ex = LL.PlTerm(model_labels)
            # print('ex')
            # print(ex)
            # print('rule')
            # print(rule)

            f = LL.PlTerm(rule)  # rule也是包含四个符号么？
            # print('f')
            # print(f)
            # print('evalInstFeature')

            if LL.evalInstFeature(ex, f): # 判断是否兼容
                # print(1)
                vector.append(1)
            else:
                # print('ex')
                # print(ex)
                # print('f')
                # print(f)
                # print('consistent?')
                # print(0)
                vector.append(0)
    return vector


def get_mlp_data(equations_true,
                 equations_false,
                 base_model,
                 out_rules,
                 abduced_map,
                 shape=(28, 28, 1)):
    mlp_vectors = []
    mlp_labels = []
    for equation in equations_true:
        # print('mlp_vector')
        # print(get_mlp_vector(equation, base_model, out_rules, abduced_map, shape))
        # print('mlp_label')
        # print(1)
        mlp_vectors.append(
            get_mlp_vector(equation, base_model, out_rules, abduced_map, shape))
        mlp_labels.append(1)
    for equation in equations_false:
        # print('mlp_vector')
        # print(get_mlp_vector(equation, base_model, out_rules, abduced_map, shape))
        # print('mlp_label')
        # print(0)
        mlp_vectors.append(
            get_mlp_vector(equation, base_model, out_rules, abduced_map, shape))
        mlp_labels.append(0)
    mlp_vectors = np.array(mlp_vectors)
    mlp_labels = np.array(mlp_labels)
    return mlp_vectors, mlp_labels


def get_file_data(src_data_file, src_data_str_file, prop_train, prop_val):
    with open(src_data_file, 'rb') as f:
        equations = pickle.load(f)
    with open(src_data_str_file, 'rb') as f:
        equations_str = pickle.load(f)
    # print('equations:',equations_str)
    input_file_true = equations['train:positive']
    input_file_false = equations['train:negative']
    input_file_true_test = equations['test:positive']
    input_file_false_test = equations['test:negative']

    input_file_true_str = equations_str['train:positive']
    input_file_false_str = equations_str['train:negative']
    input_file_true_test_str = equations_str['test:positive']
    input_file_false_test_str = equations_str['test:negative']

    equations_true_by_len,equations_str_true_by_len = divide_equation_by_len(input_file_true,input_file_true_str)
    equations_false_by_len,equations_str_false_by_len = divide_equation_by_len(input_file_false,input_file_false_str)
    equations_true_by_len_test,equations_str_true_by_len_test = divide_equation_by_len(input_file_true_test,input_file_true_test_str)
    equations_false_by_len_test,equations_str_false_by_len_test = divide_equation_by_len(input_file_false_test,input_file_false_test_str)
    #train:validation:test = prop_train:prop_val
    equations_true_by_len_train, equations_true_by_len_validation,equations_str_true_by_len_train, equations_str_true_by_len_validation = split_equation(
        equations_true_by_len, equations_str_true_by_len, prop_train, prop_val)
    equations_false_by_len_train, equations_false_by_len_validation,equations_str_false_by_len_train, equations_str_false_by_len_validation = split_equation(
        equations_false_by_len, equations_str_false_by_len, prop_train, prop_val)

    # print('equations_str_false_by_len_train:',equations_str_false_by_len_train)
    for equations_true in equations_true_by_len:
        print("There are %d true training and validation equations of length %d"
              % (len(equations_true), len(equations_true[0])))
    for equations_false in equations_false_by_len:
        print("There are %d false training and validation equations of length %d"
              % (len(equations_false), len(equations_false[0])))
    for equations_true in equations_true_by_len_test:
        print("There are %d true testing equations of length %d" %
              (len(equations_true), len(equations_true[0])))
    for equations_false in equations_false_by_len_test:
        print("There are %d false testing equations of length %d" %
              (len(equations_false), len(equations_false[0])))

    return (equations_true_by_len_train, equations_true_by_len_validation,
            equations_false_by_len_train, equations_false_by_len_validation,
            equations_true_by_len_test, equations_false_by_len_test,
            equations_str_true_by_len_train, equations_str_true_by_len_validation,
            equations_str_false_by_len_train, equations_str_false_by_len_validation,
            equations_str_true_by_len_test, equations_str_false_by_len_test)


def get_percentage_precision(base_model, select_equations, consist_re, shape):
    h = shape[0]
    w = shape[1]
    d = shape[2]
    consistent_ex_ids = consist_re.consistent_ex_ids
    abduced_map = consist_re.abduced_map

    # if DEBUG:
    #     print("Abduced labels:")
    #     for ex in consist_re.abduced_exs_mapped:
    #         for c in ex:
    #             print(c, end='')
    #         print(' ', end='')
    #     print("\nCurrent model's output:")

    model_labels = []
    for e in select_equations[consistent_ex_ids]:
        hat_y = np.argmax(base_model.predict(e.reshape(-1, h, w, d),verbose=0), axis=1)
        model_labels.append(hat_y)
        info = ""
        for y in hat_y:
            info += str(abduced_map[y])
        # if DEBUG:
        #     print(info, end=' ')
    model_labels = np.concatenate(np.array(model_labels)).flatten()

    abduced_labels = flatten(consist_re.abduced_exs)
    batch_label_model_precision = (model_labels == abduced_labels).sum() / (
        len(model_labels))
    consistent_percentage = len(consistent_ex_ids) / len(select_equations)
    print("\nBatch label model precision:", batch_label_model_precision)
    print("Consistent percentage:", consistent_percentage)

    return consistent_percentage, batch_label_model_precision


def get_rules_from_data(base_model, encoder_decoder_model,equations_true, equations_true_str, labels, abduced_map, shape,
                        LOGIC_OUTPUT_DIM, SAMPLES_PER_RULE,T,log_file,max_length,budget=10): 
    # print('get_rules_from_data')
    out_rules = []
    for i in range(LOGIC_OUTPUT_DIM):
        # print('i:',i)
        # print('i:',i,file=log_file)
        log_file.flush()
        find=False
        while True:
            select_index = np.random.randint(len(equations_true),
                                             size=SAMPLES_PER_RULE)
            select_equations = np.array(equations_true)[select_index]
            select_equations_str = np.array(equations_true_str)[select_index]
            for j in range(SAMPLES_PER_RULE,-1,-1):
                # print('j:',j)
                eq=select_equations[:j]
                _, rule, _, _,_ = get_equations_labels(base_model, encoder_decoder_model,select_equations,
                                               select_equations_str,labels, abduced_map, True, shape,max_length=max_length,T=T,budget=budget)
                if rule != None:
                    find=True
                    break
            if find:
                break
        # print(i,rule)
        out_rules.append(rule)
    return out_rules


def abduce_and_train(base_model, encoder_decoder_model, equations_true, equations_str_true, labels, abduced_map, shape,
                     SELECT_NUM, BATCHSIZE, NN_EPOCHS,max_length=8,T=1,budget=10):
    # print('abduce_and_train')
    h = shape[0]
    w = shape[1]
    d = shape[2]
    #Randomly select several equations
    select_index = np.random.randint(len(equations_true), size=SELECT_NUM)
    select_equations = np.array(equations_true)[select_index]
    select_equations_str = np.array(equations_str_true)[select_index]
    # print('select_equations')
    # print(select_equations)
    # print(len(select_equations))#10
    # print('pre_get_equations_labels')
    consist_re, _ , sol, probability,eval_time = get_equations_labels(base_model,encoder_decoder_model, select_equations, select_equations_str, labels,
                                         abduced_map, False, shape,max_length=max_length,T=T,budget=budget)

    # print('after_get_equations_labels')
    # Can not abduce
    if consist_re is None:
        # print('inconsistent with abduce')
        return (abduced_map,eval_time,0,0)
    # print('consistent with abduce')
    # print('consist_re')
    # print(consist_re)
    # print('consist_re.consistent_ex_ids')
    # print(consist_re.consistent_ex_ids)
    # print('consist_re.abduced_exs')
    # print(consist_re.abduced_exs)
    # print('consist_re.abduced_map')
    # print(consist_re.abduced_map)
    consistent_ex_ids = consist_re.consistent_ex_ids
    equations_labels = consist_re.abduced_exs
    abduced_map = consist_re.abduced_map

    train_pool_X = np.concatenate(select_equations[consistent_ex_ids]).reshape(
        -1, h, w, d)
    # print('train_pool_X')
    # print(train_pool_X.shape)
    train_pool_Y = np_utils.to_categorical(
        flatten(equations_labels),
        num_classes=len(labels))  # Convert the symbol to network output

    train_pool_X_encoder_decoder=[probability[i] for i in consistent_ex_ids] 
    train_pool_Y_encoder_decoder=[sol[i] for i in consistent_ex_ids] 
    # max_length = max(len(sub_sequence) for sub_sequence in train_pool_X_encoder_decoder)
    # print('max_len:',max_length)
    inputs  = []
    for sub_sequence in train_pool_X_encoder_decoder:
        sub_sequence_padded = np.zeros((max_length-len(sub_sequence),len(labels)))
        inputs.append(np.concatenate([sub_sequence_padded,sub_sequence]))
    inputs = np.array(inputs)
    # try:
    #     outputs = np.array(pad_sequences(train_pool_Y_encoder_decoder, padding='pre', dtype='float32'))
    #     # print("sol:",sol)
    # except:
        # print('异常！')
        # print("sol:",sol)
        # print('train_pool_Y_encoder_decoder:',train_pool_Y_encoder_decoder)
    outputs = np.array(pad_sequences(train_pool_Y_encoder_decoder, padding='pre', dtype='float32'))
    # print('outputs:',outputs.shape)
    outputs=np.concatenate([np.zeros((len(outputs),max_length-outputs.shape[1])),outputs],axis=1)
    # print('outputs:',outputs.shape)
    outputs = np_utils.to_categorical(
        flatten(outputs),
        num_classes=2).reshape(len(consistent_ex_ids),max_length,2)
    # print('train_pool_Y')
    # print(train_pool_Y.shape)
    assert (len(inputs) == len(outputs))
    assert (len(train_pool_X) == len(train_pool_Y))
    # print("\nTrain pool size is :", len(train_pool_X))
    # print("Training...")
    #cifar10  data augmentation
    if d > 1:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(train_pool_X)
        base_model.fit_generator(datagen.flow(
            train_pool_X, train_pool_Y, batch_size=train_pool_X.shape[0]),
                                 steps_per_epoch=1,
                                 epochs=NN_EPOCHS)
    else:
        base_model.fit(train_pool_X,
                       train_pool_Y,
                       batch_size=BATCHSIZE,
                       epochs=NN_EPOCHS,
                       verbose=0)
        encoder_decoder_model.fit(inputs,
                       outputs,
                       batch_size=BATCHSIZE,
                       epochs=NN_EPOCHS,
                       verbose=0)

    consistent_percentage, batch_label_model_precision = get_percentage_precision(
        base_model, select_equations, consist_re, shape)
    return abduced_map,eval_time,consistent_percentage,batch_label_model_precision


def validation(base_model, encoder_decoder_model,equations_true, equations_str_true,equations_false, equations_true_val,
               equations_false_val, labels, abduced_map, shape,
               LOGIC_OUTPUT_DIM, SAMPLES_PER_RULE, MLP_BATCHSIZE, MLP_EPOCHS,T=1,max_length=8,budget=10):
    #Generate several rules
    #Get training data and label
    #Train mlp
    #Evaluate mlp
    # print("Now checking if we can go to next course")
    # print('1')
    out_rules = get_rules_from_data(base_model, encoder_decoder_model, equations_true,equations_str_true, labels,
                                    abduced_map, shape, LOGIC_OUTPUT_DIM,
                                    SAMPLES_PER_RULE,T=T,max_length=max_length,budget=budget)
    #print(out_rules)

    #Prepare MLP training data
    # print('2')
    mlp_train_vectors, mlp_train_labels = get_mlp_data(equations_true,
                                                       equations_false,
                                                       base_model, out_rules,
                                                       abduced_map, shape)
    index = np.array(list(range(len(mlp_train_labels))))
    np.random.shuffle(index)
    mlp_train_vectors = mlp_train_vectors[index]
    mlp_train_labels = mlp_train_labels[index]

    # best_accuracy = 0
    #Try three times to find the best mlp
    # print('3')
    # for i in range(1):
        #Train MLP
        # print("Training mlp...")
    mlp_model = NN_model.get_mlp_net(LOGIC_OUTPUT_DIM)
    mlp_model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    # print('4')
    mlp_model.fit(mlp_train_vectors,
                  mlp_train_labels,
                  epochs=MLP_EPOCHS,
                  batch_size=MLP_BATCHSIZE,
                  verbose=0)
    #Prepare MLP validation data
    mlp_val_vectors, mlp_val_labels = get_mlp_data(equations_true_val,
                                                   equations_false_val,
                                                   base_model, out_rules,
                                                   abduced_map, shape)
    #Get MLP validation result
    # print('5')
    result = mlp_model.evaluate(mlp_val_vectors,
                                mlp_val_labels,
                                batch_size=MLP_BATCHSIZE,
                                verbose=0)
    # print("MLP validation result:", result)
    accuracy = result[1]

    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    return accuracy


def get_all_mlp_data(equations_true_by_len_train, equations_false_by_len_train,
                     base_model, out_rules, abduced_map, shape,
                     EQUATION_LEAST_LEN,
                     EQUATION_MAX_LEN):
    mlp_train_vectors = []
    mlp_train_labels = []
    #for each length of test equations
    for equations_type in range(EQUATION_LEAST_LEN - 5, EQUATION_MAX_LEN - 4):
        mlp_train_len_vectors, mlp_train_len_labels = get_mlp_data(
            equations_true_by_len_train[equations_type],
            equations_false_by_len_train[equations_type], base_model,
            out_rules, abduced_map, shape)
        if equations_type == EQUATION_LEAST_LEN - 5:
            mlp_train_vectors = mlp_train_len_vectors.copy()
            mlp_train_labels = mlp_train_len_labels.copy()
        else:
            mlp_train_vectors = np.concatenate((mlp_train_vectors, mlp_train_len_vectors), axis=0)
            mlp_train_labels = np.concatenate((mlp_train_labels, mlp_train_len_labels), axis=0)

    index = np.array(list(range(len(mlp_train_labels))))
    np.random.shuffle(index)
    mlp_train_vectors = mlp_train_vectors[index]
    mlp_train_labels = mlp_train_labels[index]
    return mlp_train_vectors, mlp_train_labels


def test_nn_model(model, src_path, labels, input_shape):
    print("\nNow test the NN model")
    '''
    model = NN_model.get_LeNet5_net(len(labels))
    model.load_weights('mnist_images_nlm_weights_3.hdf5')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    '''
    best_accuracy = 0
    X, Y = get_img_data(src_path, labels, input_shape)
    maps = gen_mappings([0, 1, 2, 3], [0, 1, 2, 3])
    print('\nTesting...')
    # We don't know the map, so we try all maps and get the best accuracy
    for mapping in maps:
        real_Y = []
        for y in Y:
            real_Y.append(mapping[np.argmax(y)])
        Y_cate = np_utils.to_categorical(real_Y, num_classes=len(labels))
        loss, accuracy = model.evaluate(X, Y_cate, verbose=0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    print('Neural network perception accuracy: ', best_accuracy)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU

import tensorflow as tf


# class Seq2SeqModel(Model): 
    # def __init__(self, input_size=4, hidden_size=50, output_size=2):
    #     super(Seq2SeqModel, self).__init__()
        
    #     # 创建双向LSTM层
    #     self.hidden_size=hidden_size
    #     self.encoder = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(hidden_size, return_state=True)
    #     )
        
    #     # 这里output_size需要乘以2，因为双向LSTM的输出维度会加倍
    #     tf.keras.layers.LSTM(output_size, return_sequences=True)
        
    
    # def call(self, input_seq):
    #     # 在这里，输入到encoder和decoder的维度需要保持一致
    #     encoder_output, forward_h, forward_c, backward_h, backward_c = self.encoder(input_seq)
    #     encoder_output_combined = tf.concat([encoder_output[:, :, :self.hidden_size], encoder_output[:, :, self.hidden_size:]], axis=-1)
    #     decoder_output = self.decoder(encoder_output_combined)
    #     return decoder_output

def get_lstm(hidden_size,input_shape):
    model = Sequential()
    model.add(tf.keras.layers.Bidirectional(LSTM(units=hidden_size, input_shape=input_shape, return_sequences=True)))
    model.add(Dense(2, activation='softmax'))
    return model

def get_gru(hidden_size,input_shape):
    model = Sequential()
    model.add(tf.keras.layers.Bidirectional(GRU(units=hidden_size, input_shape=input_shape, return_sequences=True)))
    model.add(Dense(2, activation='softmax'))
    return model

def get_mlp(hidden_size,input_shape):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(hidden_size*input_shape[0],activation='relu'))
    model.add(Dense(2*input_shape[0],activation='relu'))
    model.add(Reshape((input_shape[0],2)))
    model.add(Dense(2, activation='softmax'))
    return model


def get_transformer(hidden_size,input_shape):
    model = Sequential()
    # Input(shape=(MAX_LEN, EMBED_DIM))
    # transformer = Lambda(lambda x: transformer_encoder(x, input_shape[1] ,8, hidden_size,0))
    model.add(TransformerEncoderLayer(input_shape[1] ,8, hidden_size,0))
    model.add(Dense(2, activation='softmax'))
    return model
# class Seq2SeqModel(Model): 
#     def __init__(self, input_size=4, hidden_size=50, output_size=2):
#         super(Seq2SeqModel, self).__init__()
#         self.encoder = LSTM(hidden_size, return_state=True, return_sequences=True)
#         self.decoder = LSTM(output_size, return_state=True, return_sequences=True)
    
#     def call(self, input_seq):
#         encoder_output, state_h, state_c = self.encoder(input_seq)
#         decoder_output = self.decoder(encoder_output)
#         return decoder_output

def encoder_decoder_pretrain(equations_true_by_len_train, base_model, EQUATION_LEAST_LEN, EQUATION_MAX_LEN,
                        input_size=4,hidden_size=64,output_size=2,epochs=10,batch_size=32,shape=(28, 28, 1),src_data_name='',pretrain_epochs=5,args=None):
    
    # encoder_decoder_model=Seq2SeqModel(input_size,hidden_size,output_size)
    if args.snn == 'lstm':
        encoder_decoder_model=get_lstm(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
    elif args.snn == 'gru':
        encoder_decoder_model=get_gru(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
    elif args.snn == 'mlp':
        print('mlp!')
        encoder_decoder_model=get_mlp(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
    elif args.snn == 'transformer':
        encoder_decoder_model=get_transformer(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
    else:
        print(args.snn)
        print('not found')
    list_inputs=[]
    list_outputs=[]
    if os.path.exists('%s_list_outputs_%d_%d.pkl' % (src_data_name,pretrain_epochs,EQUATION_MAX_LEN)) and os.path.exists('%s_list_inputs_%d.pkl' % (src_data_name,pretrain_epochs)):#and False:
        with open('%s_list_inputs_%d_%d.pkl' % (src_data_name,pretrain_epochs,EQUATION_MAX_LEN), 'rb') as file:
            list_inputs=pickle.load(file)
        with open('%s_list_outputs_%d_%d.pkl' % (src_data_name,pretrain_epochs,EQUATION_MAX_LEN), 'rb') as file:
            list_outputs=pickle.load(file)
    else:
        print('Pretraining')
        for i in range(3):# 0,1,2,3,4 
            print('i:', i)
            # print(max(0,i-EQUATION_LEAST_LEN))
            # print(min(i*2+1,EQUATION_MAX_LEN-EQUATION_LEAST_LEN+1))
            min_len=min(EQUATION_MAX_LEN,max(i*4,EQUATION_LEAST_LEN))
            max_len=EQUATION_MAX_LEN
            equations_true = equations_true_by_len_train[min_len - EQUATION_LEAST_LEN : max_len- EQUATION_LEAST_LEN+1]
            equations=[]
            
            for _ in range(len(equations_true)):
                equations.extend(equations_true[_])
        
            exs=[]
            probability=[]
            equations=np.array(equations)
            # print('equations:', len(equations))
            cur_list_inputs=[]
            cur_list_outputs=[]
            for e in equations:
                h=shape[0]
                w=shape[1]
                d=shape[2]
                probability=base_model.predict(e.reshape(-1, h, w, d),verbose=0)
                vector=np.zeros(probability.shape[0])
                wrong_idxs = random.sample(range(probability.shape[0]), i)
                vector[wrong_idxs]=1
                for idx in wrong_idxs:
                    max_idx=np.argmax(probability[idx])
                    while np.argmax(probability[idx])==max_idx:
                        np.random.shuffle(probability[idx])
                list_inputs.append(probability)
                list_outputs.append(vector)
                cur_list_inputs.append(probability)
                cur_list_outputs.append(vector)
            # max_length=EQUATION_MAX_LEN
            # # max_length = max(len(sub_sequence) for sub_sequence in cur_list_inputs)
            # inputs  = []
            # for sub_sequence in cur_list_inputs:
            #     sub_sequence_padded = np.zeros((max_length-len(sub_sequence),input_size))
            #     inputs.append(np.concatenate([sub_sequence_padded,sub_sequence]))
            # inputs = np.array(inputs)
            # print(inputs.shape)
            # # inputs = [pad_sequences(sub_sequence, padding='pre', dtype='float32', maxlen=max_length) for sub_sequence in list_inputs]
            # outputs = np.array(pad_sequences(cur_list_outputs, padding='pre', dtype='float32'))
            # print('outputs:',outputs.shape)
            # outputs=np.concatenate([np.zeros((len(outputs),max_length-outputs.shape[1])),outputs],axis=1)
            # print('outputs:',outputs.shape)
            # print('inputs:',inputs.shape)
            # # print(len(inputs))
            # # print(len(outputs))
            # outputs = np_utils.to_categorical(flatten(outputs),num_classes=2).reshape(len(cur_list_outputs),max_length,2) 
            # print('pre_fit')
            # test_loss, test_acc = encoder_decoder_model.evaluate(inputs[len(inputs)//2:], outputs[len(inputs)//2:])
            # print('test_loss:',test_loss)
            # print('test_acc:',test_acc)
            # encoder_decoder_model.fit(inputs[:len(inputs)//2],
            #                           outputs[:len(inputs)//2],
            #                           epochs=epochs,
            #                           batch_size=batch_size,
            #                           verbose=0)
            # test_loss, test_acc = encoder_decoder_model.evaluate(inputs[len(inputs)//2:], outputs[len(inputs)//2:])
            # print('after_fit')
            # print('test_loss:',test_loss)
            # print('test_acc:',test_acc)
        with open('%s_list_inputs_%d_%d.pkl'%(src_data_name,pretrain_epochs,EQUATION_MAX_LEN), 'wb') as file:
            pickle.dump(list_inputs, file)
        with open('%s_list_outputs_%d_%d.pkl'%(src_data_name,pretrain_epochs,EQUATION_MAX_LEN), 'wb') as file:
            pickle.dump(list_outputs, file)
    encoder_decoder_model.compile(loss='binary_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])
    # print('len:',len(list_inputs))
    # print('inputs:', list_inputs[0].shape)
    # print('outputs:', list_outputs[0].shape)
    max_length = EQUATION_MAX_LEN
    # max_length = max(len(sub_sequence) for sub_sequence in list_inputs)
    # print('max_len:',max_length)
    # print(list_inputs[0])
    print(pad_sequences(list_inputs[0], padding='pre', dtype='float32', maxlen=max_length))
    inputs  = []
    for sub_sequence in list_inputs:
        sub_sequence_padded = np.zeros((max_length-len(sub_sequence),input_size))
        inputs.append(np.concatenate([sub_sequence_padded,sub_sequence]))
    inputs = np.array(inputs)
    print(inputs.shape)
    # inputs = [pad_sequences(sub_sequence, padding='pre', dtype='float32', maxlen=max_length) for sub_sequence in list_inputs]
    outputs = pad_sequences(list_outputs, padding='pre', dtype='float32')
    # print(len(inputs))
    # print(len(outputs))
    outputs = np_utils.to_categorical(flatten(outputs),num_classes=2).reshape(len(list_outputs),max_length,2)
    # print(outputs.shape)
    encoder_decoder_model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    encoder_decoder_model.fit(inputs,
                  outputs,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0)

    # encoder_decoder_model.fit(inputs,
    #               outputs,
    #               epochs=epochs,
    #               batch_size=batch_size,
    #               verbose=0)
    return encoder_decoder_model

def nlm_main_func(labels, src_data_name, src_data_file, src_data_str_file, pl_file_path, shape, src_path, args):
    LL_init(pl_file_path)
    # text_file=open(args.log_file+'.txt', "w", encoding="utf-8")
    text_file=open('testext6_'+args.src_data_name+'_'+str(args.pretrain)+'_'+str(args.pretrain_encoder)+
        '_'+str(args.sup)+'_'+str(args.balance)+'_'+str(args.num)+'_'+str(args.pretrain_epochs)+'_'
        +str(args.epoch)+'_'+str(args.T)+'_'+str(args.SELECT_NUM)+'_'+str(args.SAMPLES_PER_RULE)+'_'
        +str(args.EQUATION_MAX_LEN)+'_'+str(args.MLP_EPOCHS)+'_'+str(args.budget)+'_'+str(args.NN_EPOCHS)
        +'_'+str(args.hidden_size)+'_'+args.snn+'_'+str(args.PROP_TRAIN)+'_'+str(args.PROP_VALIDATION)+'_'+args.hint+'.txt', "w", encoding="utf-8")
    csv_file=open('testext6_'+args.src_data_name+'_'+str(args.pretrain)+'_'+str(args.pretrain_encoder)
        +'_'+str(args.sup)+'_'+str(args.balance)+'_'+str(args.num)+'_'+str(args.pretrain_epochs)+'_'
        +str(args.epoch)+'_'+str(args.T)+'_'+str(args.SELECT_NUM)+'_'+str(args.SAMPLES_PER_RULE)+'_'
        +str(args.EQUATION_MAX_LEN)+'_'+str(args.MLP_EPOCHS)+'_'+str(args.budget)+'_'+str(args.NN_EPOCHS)
        +'_'+str(args.hidden_size)+'_'+args.snn+'_'+str(args.PROP_TRAIN)+'_'+str(args.PROP_VALIDATION)+'_'+args.hint+'.csv', "w", encoding="utf-8")
    r = csv.DictWriter(csv_file,['epoch','result','num_rules'])
    h = shape[0]
    w = shape[1]
    d = shape[2]
    abduced_map = None

    #LOGIC_OUTPUT_DIM = 50 # The mlp vector has 50 dimensions
    LOGIC_OUTPUT_DIM = args.LOGIC_OUTPUT_DIM
    #EQUATION_MAX_LEN = 8 # Only learn the equations of length 5-8
    EQUATION_LEAST_LEN = args.EQUATION_LEAST_LEN
    EQUATION_MAX_LEN = args.EQUATION_MAX_LEN
    EQUATION_LEN_CNT = EQUATION_MAX_LEN - EQUATION_LEAST_LEN + 1  #equations index is 0-4
    #SELECT_NUM = 10 # Select 10 equations to abduce rules
    SELECT_NUM = args.SELECT_NUM
    #
    ## Proportion of train and validation = 3:1
    #PROP_TRAIN = 3
    PROP_TRAIN = args.PROP_TRAIN
    #PROP_VALIDATION = 1
    PROP_VALIDATION = args.PROP_VALIDATION
    #
    #CONSISTENT_PERCENTAGE_THRESHOLD = 0.9
    CONSISTENT_PERCENTAGE_THRESHOLD = args.CONSISTENT_PERCENTAGE_THRESHOLD
    #BATCH_LABEL_MODEL_PRECISION_THRESHOLD = 0.9 #If consistent percentage is higher than 0.9 and model precision higher than 0.9, then the condition is satisfied
    BATCH_LABEL_MODEL_PRECISION_THRESHOLD = args.BATCH_LABEL_MODEL_PRECISION_THRESHOLD
    #CONDITION_CNT_THRESHOLD = 5       #If the condition has been satisfied 5 times, the start validation
    CONDITION_CNT_THRESHOLD = args.CONDITION_CNT_THRESHOLD
    #NEXT_COURSE_ACC_THRESHOLD = 0.86  #If the validation accuracy of a course higher than the threshold, then go to next course
    NEXT_COURSE_ACC_THRESHOLD = args.NEXT_COURSE_ACC_THRESHOLD

    #SAMPLES_PER_RULE = 3 # Use 3 samples to abduce a rule when training mlp
    SAMPLES_PER_RULE = args.SAMPLES_PER_RULE
    #NN_BATCHSIZE = 32    # Batch size of neural network
    NN_BATCHSIZE = args.NN_BATCHSIZE
    #NN_EPOCHS = 10       # Epochs of neural network
    NN_EPOCHS = args.NN_EPOCHS
    #MLP_BATCHSIZE = 128  # Batch size of mlp
    MLP_BATCHSIZE = args.MLP_BATCHSIZE
    #MLP_EPOCHS = 60      # Epochs of mlp
    MLP_EPOCHS = args.MLP_EPOCHS

    # Get NN model and compile
    base_model = get_nlm_net(len(labels), shape)
    if args.pretrain:
        if args.sup:
            if d == 1:
                t_model = NN_model.get_LeNet5_net(len(labels), shape)
            else:
                t_model = NN_model.get_cifar10_net(len(labels), shape)
            t_model.load_weights('%s_pretrain_%d_weights_sup_%d_%s.hdf5' % (src_data_name,args.pretrain_epochs,args.num,str(args.balance)))
        else:
            if d == 1:
                t_model = NN_model.get_LeNet5_autoencoder_net(len(labels), shape)
            else:
                t_model = NN_model.get_cifar10_net(len(labels), shape)
            t_model.load_weights('%s_pretrain_%d_weights_%d_%s.hdf5' % (src_data_name,args.pretrain_epochs,args.num,str(args.balance)))
    
        for i in range(len(base_model.layers)):
            base_model.layers[i].set_weights(t_model.layers[i].get_weights())
    opt_rms = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
    base_model.compile(optimizer=opt_rms, loss='categorical_crossentropy', metrics=['accuracy'])

    # Get file data
    equations_true_by_len_train,equations_true_by_len_validation,equations_false_by_len_train,\
    equations_false_by_len_validation,equations_true_by_len_test,equations_false_by_len_test,\
    equations_str_true_by_len_train,equations_str_true_by_len_validation,equations_str_false_by_len_train,\
    equations_str_false_by_len_validation,equations_str_true_by_len_test,equations_str_false_by_len_test = get_file_data(src_data_file, src_data_str_file,PROP_TRAIN, PROP_VALIDATION)
    # print('type:',type(equations_true_by_len_train[0]))
    equations_true=equations_true_by_len_train[:EQUATION_MAX_LEN - 4]
    equations_str_true=equations_str_true_by_len_train[:EQUATION_MAX_LEN - 4]
    equations=[]
    
    for _ in range(len(equations_true)):
        equations.extend(equations_true[_])
    equations=np.array(equations)
    exs=[]
    maps = gen_mappings([0, 1, 2, 3], ['+', '=', 0, 1])
    for e in equations:
        prob=base_model.predict(e.reshape(-1, h, w, d),verbose=0)
        exs.append(np.argmax(prob, axis=1).tolist())
    max_map_consist_num=-1
    max_map=None
    for mapping in maps: 
        map_consist_num=0
        for ex in exs:
            # print(ex)
            # print([mapping[ex[i]] for i in range(len(ex))])
            con_res=consistent_score_mapped([ex],[0] * len(ex),mapping)
            if con_res:
                map_consist_num+=1
        if max_map_consist_num<map_consist_num:
            max_map_consist_num=map_consist_num
            max_map=mapping
        # print('mapping:',mapping)
        # print('map_consist_num:',map_consist_num)
    print('max_map_consist_num:',max_map_consist_num,file=text_file)
    print('max_map:',max_map,file=text_file)
    print('max_map_consist_num:',max_map_consist_num)
    print('max_map:',max_map)
    print('recall:',max_map_consist_num/len(exs))
    if args.pretrain_encoder:
        file_name = '%s_encoder_decoder_model_weights_%d_%s_%d_%s_%s_%d_%d_%s_%d_%d*4.hdf5' % (src_data_name,args.pretrain_epochs,str(args.sup),args.num,str(args.pretrain),str(args.balance),args.EQUATION_MAX_LEN,args.hidden_size,args.snn,args.PROP_TRAIN,args.PROP_VALIDATION)

        if os.path.exists(file_name) is not True: #or True:
            print('pretrain')
            pretrained_encoder_decoder=encoder_decoder_pretrain(equations_true_by_len_train[:EQUATION_MAX_LEN - 4], base_model, EQUATION_LEAST_LEN, EQUATION_MAX_LEN, input_size=4,hidden_size=args.hidden_size,
                output_size=2, epochs=NN_EPOCHS, batch_size=NN_BATCHSIZE,shape=shape,src_data_name=src_data_name,args=args)
            pretrained_encoder_decoder.save_weights(file_name)
        else:
            print('pretrained')
            if args.snn == 'lstm':
                pretrained_encoder_decoder=get_lstm(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            elif args.snn == 'gru':
                pretrained_encoder_decoder=get_gru(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            elif args.snn == 'mlp':
                pretrained_encoder_decoder=get_mlp(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            elif args.snn == 'transformer':
                pretrained_encoder_decoder=get_transformer(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            pretrained_encoder_decoder.call(tf.zeros((1,EQUATION_MAX_LEN,4)))
            pretrained_encoder_decoder.load_weights(file_name)
    # for i in range(len(pretrained_encoder_decoder.layers)):
    #     pretrained_encoder_decoder.layers[i].set_weights(t_encoder_decoder_model.layers[i].get_weights())
    else:
        if args.snn == 'lstm':
            pretrained_encoder_decoder=get_lstm(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
        elif args.snn == 'gru':
            pretrained_encoder_decoder=get_gru(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
        elif args.snn == 'mlp':
            encoder_decoder_model=get_mlp(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
        elif args.snn == 'transformer':
            encoder_decoder_model=get_transformer(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
    pretrained_encoder_decoder.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    #Seq2SeqModel(input_size=4,hidden_size=50,output_size=2).load_weights(file_name)
    total_time_list=[]
    start_time=time.time()
    # Start training / for each length of equations
    equations_true = []
    equations_false = []
    equations_true_val = []
    equations_false_val = []
    equations_true_test = []
    equations_false_test = []
    equations_str_true = []
    equations_str_false = []
    equations_str_true_val = []
    equations_str_false_val = []
    equations_str_true_test = []
    equations_str_false_test = []
    for equations_type in range(EQUATION_LEAST_LEN - 5, EQUATION_MAX_LEN - 4): # 5 0,1,2
        print('equations_type:',equations_type)
        print('len:',len(equations_true_by_len_train[equations_type][0]))
        print('equations_type:',equations_type,file=text_file)
        print('len:',len(equations_true_by_len_train[equations_type][0]),file=text_file)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print("LENGTH: ", 5 + equations_type, " to ", 5 + equations_type + 1,file=text_file)
        # print("LENGTH: ", 5 + equations_type, " to ", 5 + equations_type + 1)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # equations_true = equations_true_by_len_train[equations_type]
        # # print('type:',type(equations_true))
        # # print('len_equations_true:', len(equations_true))
        # # print('len_equations_true_0:', equations_true[0].shape)
        # equations_false = equations_false_by_len_train[equations_type]
        # equations_true_val = equations_true_by_len_validation[equations_type]
        # equations_false_val = equations_false_by_len_validation[equations_type]
        # equations_str_true = equations_str_true_by_len_train[equations_type]
        # equations_str_false = equations_str_false_by_len_train[equations_type]
        # equations_str_true_val = equations_str_true_by_len_validation[equations_type]
        # equations_str_false_val = equations_str_false_by_len_validation[equations_type]
        equations_true.extend(equations_true_by_len_train[equations_type])
        equations_false.extend(equations_false_by_len_train[equations_type])
        equations_true_val.extend(equations_true_by_len_validation[equations_type])
        equations_false_val.extend(equations_false_by_len_validation[equations_type])
        equations_true_test.extend(equations_true_by_len_test[equations_type])
        equations_false_test.extend(equations_false_by_len_test[equations_type])
        equations_str_true.extend(equations_str_true_by_len_train[equations_type])
        equations_str_false.extend(equations_str_false_by_len_train[equations_type])
        equations_str_true_val.extend(equations_str_true_by_len_validation[equations_type])
        equations_str_false_val.extend(equations_str_false_by_len_validation[equations_type])
        equations_str_true_test.extend(equations_str_true_by_len_test[equations_type])
        equations_str_false_test.extend(equations_str_false_by_len_test[equations_type])
        # 每次两种长度的公式
        #the times that the condition of beginning to train MLP is continuously satisfied
        # print('equations_true[0]')
        # print(equations_true[0])
        # print('equations_false[0]')
        # print(equations_false[0])
        #print(len(equations_true))#450
        #print(len(equations_true[0]))#5
        #print(len(equations_true[0][0]))#28
        #print(len(equations_true[0][0][0]))#28
    # random_numbers = random.sample(range(len(equations_true_val)), 100)
    # equations_true_val=[equations_true_val[idx] for idx in random_numbers]
    # equations_str_true_val=[equations_str_true_val[idx] for idx in random_numbers]
    # random_numbers = random.sample(range(len(equations_false_val)), 100)
    # equations_false_val=[equations_false_val[idx] for idx in random_numbers]
    # equations_str_false_val=[equations_str_false_val[idx] for idx in random_numbers]
    random_numbers = random.sample(range(len(equations_true_test)), 300)
    equations_true_test=[equations_true_test[idx] for idx in random_numbers]
    equations_str_true_test=[equations_str_true_test[idx] for idx in random_numbers]
    random_numbers = random.sample(range(len(equations_false_test)), 300)
    equations_false_test=[equations_false_test[idx] for idx in random_numbers]
    equations_str_false_test=[equations_str_false_test[idx] for idx in random_numbers]
    print('equations_true_val:',len(equations_true_val))
    print('equations_false_val:',len(equations_false_val))
    print('equations_true_test:',len(equations_true_test))
    print('equations_false_test:',len(equations_false_test))
    abduced_map = max_map
    condition_cnt = 0
    total_time=0
    total_time_s=0
    sum_batch=0
    sum_con=0
    flag=False
    while True:
        total_time+=1
        print('total_time:',total_time,file=text_file)
        print('total_time:',total_time)
        # if equations_type < 1:
            
        s_time=time.time()
        # Abduce and train NN # 预测结果到符号的映射，不固定map是因为没有使用图像标签
        abduced_map ,eval_time,consistent_percentage,batch_label_model_precision= abduce_and_train(
            base_model, pretrained_encoder_decoder, equations_true, equations_str_true, labels, abduced_map, shape, #      labels【0,1,10,11】
            SELECT_NUM, NN_BATCHSIZE, NN_EPOCHS,max_length=EQUATION_MAX_LEN,T=args.T,budget=args.budget)

        sum_batch+=batch_label_model_precision
        sum_con+=consistent_percentage

        print('batch_label_model_precision:',batch_label_model_precision)
        print('consistent_percentage:',consistent_percentage)

        print('batch_label_model_precision:',batch_label_model_precision,file=text_file)
        print('consistent_percentage:',consistent_percentage,file=text_file)
        e_time=time.time()
        print('abduce_time:',e_time - s_time)
        print('eval_time:',eval_time)
        print('abduce_time:',e_time - s_time,file=text_file)
        print('eval_time:',eval_time,file=text_file)
        total_time_s += e_time - s_time -eval_time 
        text_file.flush()

        if total_time % args.epoch==0:
            print('final_total_time:',total_time,file=text_file)
            print('final_total_time:',total_time)
            total_time_list.append(total_time)
            print('total_time:',total_time_list,file=text_file)
            print('total_time:',total_time_list)
            # end_time=time.time()
            print('total_time_sec:',total_time_s,file=text_file)
            print('total_time_sec:',total_time_s)
            text_file.flush()
            out_rules=[]
            for i in range((len(equations_true)+args.SAMPLES_PER_RULE-1)//args.SAMPLES_PER_RULE):
                select_equations=np.array(equations_true[i*args.SAMPLES_PER_RULE:min(i*args.SAMPLES_PER_RULE+args.SAMPLES_PER_RULE,len(equations_true))])
                select_equations_str=equations_str_true[i*args.SAMPLES_PER_RULE:min(i*args.SAMPLES_PER_RULE+args.SAMPLES_PER_RULE,len(equations_true))]
                _, rule, _, _,_ = get_equations_labels(base_model, pretrained_encoder_decoder,select_equations,
                                               select_equations_str,labels, abduced_map, True, shape,max_length=EQUATION_MAX_LEN,T=args.T,budget=args.budget)
                if rule:
                    out_rules.append(rule)
                    #if len(rule)>=100:
                    #    break
            #Get mlp training data
            print('len_rule:',len(out_rules))
            print('len_rule:',len(out_rules),file=text_file)
            text_file.flush()
            dic = {}
            if len(out_rules):
                mlp_train_vectors, mlp_train_labels = get_all_mlp_data(
                    equations_true_by_len_train, equations_false_by_len_train, base_model,
                    out_rules, abduced_map, shape, EQUATION_LEAST_LEN, EQUATION_MAX_LEN)
                
                mlp_model = NN_model.get_mlp_net(len(out_rules))
                mlp_model.compile(loss='binary_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                mlp_model.fit(mlp_train_vectors,
                              mlp_train_labels,
                              epochs=MLP_EPOCHS,
                              batch_size=MLP_BATCHSIZE,
                              verbose=0)
                # mlp_val_len_vectors, mlp_val_len_labels = get_mlp_data(
                #     equations_true_val, equations_false_val, base_model, out_rules,
                #     abduced_map, shape)
                # # result = mlp_model.evaluate(mlp_test_len_vectors,
                # #                             mlp_test_len_labels,
                # #                             batch_size=MLP_BATCHSIZE,
                # #                             verbose=0)
                # result_val = mlp_model.evaluate(mlp_val_len_vectors,
                #             mlp_val_len_labels,
                #             batch_size=MLP_BATCHSIZE,
                #             verbose=0)

                mlp_test_len_vectors, mlp_test_len_labels = get_mlp_data(
                    equations_true_test, equations_false_test, base_model, out_rules,
                    abduced_map, shape)
                result = mlp_model.evaluate(mlp_test_len_vectors,
                                            mlp_test_len_labels,
                                            batch_size=MLP_BATCHSIZE,
                                            verbose=0)

                print('test_acc:',result[1])
                print('test_acc:',result[1],file=text_file)
                # print('val_acc:',result_val[1])
                # print('val_acc:',result_val[1],file=text_file)
                dic['epoch']=total_time
                dic['result']=result[1]
                # dic['result_val']=result_val[1]
                dic['num_rules']=len(out_rules)
            else:
                # es=[]
                # pred=[]
                # ans=[]
                # num_true=0
                # num_false=0
                # for _ in range(len(equations_true_val)):
                #     # print('e')
                #     # print(len(e))
                #     # print('e:',e)
                #     e=equations_true_val[_]
                # # ans=equations_str_true_val[_]
                # # try:
                #     prob=base_model.predict(e.reshape(-1, h, w, d),verbose=0)
                #     # except:
                #         # print('异常！')
                #         # print('e:',e)
                #         # print('equations:',equations)
                #         # prob=model.predict(e.reshape(-1, h, w, d),verbose=0)
                #     _e=np.argmax(prob, axis=1).tolist()
                #     es.append(_e)
                #     ans.append(1)
                #     con_res=consistent_score_mapped([_e],[0] * len(_e),abduced_map)
                #     if con_res:
                #         pred.append(1)
                #         num_true+=1
                #     else:
                #         pred.append(0)
                # print('recall:',num_true/len(equations_true_val))
                # print('recall:',num_true/len(equations_true_val),file=text_file)

                # for _ in range(len(equations_false_val)):
                #     # print('e')
                #     # print(len(e))
                #     # print('e:',e)
                #     e=equations_false_val[_]
                #     # ans=equations_str_false_val[_]
                #     # try:
                #     prob=base_model.predict(e.reshape(-1, h, w, d),verbose=0)
                # # except:
                #     # print('异常！')
                #     # print('e:',e)
                #     # print('equations:',equations)
                #     # prob=model.predict(e.reshape(-1, h, w, d),verbose=0)
                #     _e=np.argmax(prob, axis=1).tolist()
                #     es.append(_e)
                #     ans.append(0)
                #     con_res=consistent_score_mapped([_e],[0] * len(_e),abduced_map)
                #     if con_res:
                #         pred.append(1)
                #     else:
                #         pred.append(0)
                #         num_false+=1
                # print('neg_recall:',num_false/len(equations_false_val),file=text_file)
                # print('val_acc:',(num_false+num_true)/len(pred),file=text_file)
                # print('neg_recall:',num_false/len(equations_false_val))
                # print('val_acc:',(num_false+num_true)/len(pred))
                # result_val=(num_false+num_true)/len(pred)

                es=[]
                pred=[]
                ans=[]
                num_true=0
                num_false=0
                for _ in range(len(equations_true_test)):
                    # print('e')
                    # print(len(e))
                    # print('e:',e)
                    e=equations_true_test[_]
                # ans=equations_str_true_val[_]
                # try:
                    prob=base_model.predict(e.reshape(-1, h, w, d),verbose=0)
                    # except:
                        # print('异常！')
                        # print('e:',e)
                        # print('equations:',equations)
                        # prob=model.predict(e.reshape(-1, h, w, d),verbose=0)
                    _e=np.argmax(prob, axis=1).tolist()
                    es.append(_e)
                    ans.append(1)
                    con_res=consistent_score_mapped([_e],[0] * len(_e),abduced_map)
                    if con_res:
                        pred.append(1)
                        num_true+=1
                    else:
                        pred.append(0)
                print('recall:',num_true/len(equations_true_test))
                print('recall:',num_true/len(equations_true_test),file=text_file)

                for _ in range(len(equations_false_test)):
                    # print('e')
                    # print(len(e))
                    # print('e:',e)
                    e=equations_false_test[_]
                    # ans=equations_str_false_val[_]
                    # try:
                    prob=base_model.predict(e.reshape(-1, h, w, d),verbose=0)
                # except:
                    # print('异常！')
                    # print('e:',e)
                    # print('equations:',equations)
                    # prob=model.predict(e.reshape(-1, h, w, d),verbose=0)
                    _e=np.argmax(prob, axis=1).tolist()
                    es.append(_e)
                    ans.append(0)
                    con_res=consistent_score_mapped([_e],[0] * len(_e),abduced_map)
                    if con_res:
                        pred.append(1)
                    else:
                        pred.append(0)
                        num_false+=1
                print('neg_recall:',num_false/len(equations_false_test),file=text_file)
                print('test_acc:',(num_false+num_true)/len(pred),file=text_file)
                print('neg_recall:',num_false/len(equations_false_test))
                print('test_acc:',(num_false+num_true)/len(pred))
                result=(num_false+num_true)/len(pred)
                dic['epoch']=total_time
                dic['result']=(num_false+num_true)/len(pred)
                # dic['result_val']=result_val
                dic['num_rules']=0
            print('sum_batch:',sum_batch/10)
            print('sum_batch:',sum_batch/10,file=text_file)
            print('sum_con:',sum_con/10)
            print('sum_con:',sum_con/10,file=text_file)
            # if sum_batch/10 >= BATCH_LABEL_MODEL_PRECISION_THRESHOLD and sum_con /10>= CONSISTENT_PERCENTAGE_THRESHOLD:
            #     condition_cnt += 1
            # else:
            #     condition_cnt = 0
            sum_batch=0
            sum_con=0
            # print('condition_cnt:',condition_cnt)
            print('consistent_percentage:',consistent_percentage)
            print('batch_label_model_precision:',batch_label_model_precision)
            print('condition_cnt:',condition_cnt,file=text_file)
            print('consistent_percentage:',consistent_percentage,file=text_file)
            print('batch_label_model_precision:',batch_label_model_precision,file=text_file)
            # if condition_cnt >= 10:

            # # text_file.flush()
            # # decide next course or restart
            # # Save model and go to next course
            #     if dic['result_val'] <= NEXT_COURSE_ACC_THRESHOLD and flag is False:
            #         print('result_val:',result_val)
            #         print('result_val:',result_val,file=text_file)
            #     #     base_model.save_weights('%s_nlm_weights_%d_%d_%s_%d_%s.hdf5' %
            #     #                             (src_data_name, equations_type,args.pretrain_epochs,str(args.sup),args.num))
            #     #     pretrained_encoder_decoder.save_weights('%s_encoder_decoder_weights_%d_%d_%s_%d.hdf5' %
            #     #                             (src_data_name, equations_type,args.pretrain_epochs,str(args.sup),args.num))
            #     # else:
            #         # #Restart current course: reload model
            #         # if equations_type == EQUATION_LEAST_LEN - 5:
            #         #     for i in range(len(base_model.layers)):
            #         #         base_model.layers[i].set_weights(
            #         #             t_model.layers[i].get_weights())
            #         #     pretrained_encoder_decoder.load_weights(file_name)
            #         # else:
            #         if args.pretrain:
            #             if args.sup:
            #                 if d == 1:
            #                     t_model = NN_model.get_LeNet5_net(len(labels), shape)
            #                 else:
            #                     t_model = NN_model.get_cifar10_net(len(labels), shape)
            #                 t_model.load_weights('%s_pretrain_%d_weights_sup_%d_%s.hdf5' % (src_data_name,args.pretrain_epochs,args.num,str(args.balance)))
            #             else:
            #                 if d == 1:
            #                     t_model = NN_model.get_LeNet5_autoencoder_net(len(labels), shape)
            #                 else:
            #                     t_model = NN_model.get_cifar10_net(len(labels), shape)
            #                 t_model.load_weights('%s_pretrain_%d_weights_%d_%s.hdf5' % (src_data_name,args.pretrain_epochs,args.num,str(args.balance)))
                    
            #             for i in range(len(base_model.layers)):
            #                 base_model.layers[i].set_weights(t_model.layers[i].get_weights())
            #         if args.pretrain_encoder:
            #             file_name = '%s_encoder_decoder_model_weights_%d_%s_%d_%s_%s_%d_%d_%s_%s.hdf5' % (src_data_name,args.pretrain_epochs,str(args.sup),args.num,str(args.pretrain),str(args.balance),args.EQUATION_MAX_LEN,args.hidden_size,args.snn,args.hint)

            #             if os.path.exists(file_name) is not True: #or True:
            #                 print('pretrain')
            #                 pretrained_encoder_decoder=encoder_decoder_pretrain(equations_true_by_len_train[:EQUATION_MAX_LEN - 4], base_model, EQUATION_LEAST_LEN, EQUATION_MAX_LEN, input_size=4,hidden_size=args.hidden_size,
            #                     output_size=2, epochs=NN_EPOCHS, batch_size=NN_BATCHSIZE,shape=shape,src_data_name=src_data_name,args=args)
            #                 pretrained_encoder_decoder.save_weights(file_name)
            #             else:
            #                 print('pretrained')
            #                 if args.snn == 'lstm':
            #                     pretrained_encoder_decoder=get_lstm(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #                 elif args.snn == 'gru':
            #                     pretrained_encoder_decoder=get_gru(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #                 elif args.snn == 'mlp':
            #                     pretrained_encoder_decoder=get_mlp(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #                 elif args.snn == 'transformer':
            #                     pretrained_encoder_decoder=get_transformer(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #                 pretrained_encoder_decoder.call(tf.zeros((1,EQUATION_MAX_LEN,4)))
            #                 pretrained_encoder_decoder.load_weights(file_name)
            #         # for i in range(len(pretrained_encoder_decoder.layers)):
            #         #     pretrained_encoder_decoder.layers[i].set_weights(t_encoder_decoder_model.layers[i].get_weights())
            #         else:
            #             if args.snn == 'lstm':
            #                 pretrained_encoder_decoder=get_lstm(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #             elif args.snn == 'gru':
            #                 pretrained_encoder_decoder=get_gru(hidden_size=args.hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #             elif args.snn == 'mlp':
            #                 pretrained_encoder_decoder=get_mlp(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #             elif args.snn == 'transformer':
            #                 pretrained_encoder_decoder=get_transformer(hidden_size=hidden_size,input_shape=(EQUATION_MAX_LEN,4))
            #         # base_model.load_weights(
            #         #     '%s_nlm_weights_%d_%d.hdf5' %
            #         #     (src_data_name, equations_type - 1,args.pretrain_epochs))
            #         # pretrained_encoder_decoder.load_weights(
            #         #     '%s_encoder_decoder_weights_%d.hdf5' %
            #         #     (src_data_name, equations_type - 1,args.pretrain_epochs))
            #         pretrained_encoder_decoder.compile(loss='binary_crossentropy',
            #           optimizer='rmsprop',
            #           metrics=['accuracy'])
            #         print("Failed! Reload model.")
            #         print("Failed! Reload model.",file=text_file)
            #         condition_cnt = 0
            #     else:
            #         flag=True
                # exs.append(_e)
                # exs_mapped.append([])
                # probability.append(prob)
        # if consistent_percentage == 0:
        #     continue

        #Test if we can use mlp to evaluate
        #The condition is: consistent_percentage >= CONSISTENT_PERCENTAGE_THRESHOLD and batch_label_model_precision>=BATCH_LABEL_MODEL_PRECISION_THRESHOLD
        #The condition has been satisfied continuously five times
        # if condition_cnt >= CONDITION_CNT_THRESHOLD:


            # for equations_type, (_equations_true, _equations_false) in enumerate(
            #         zip(equations_true_by_len_test, equations_false_by_len_test)):
            #     #for each length of test equations
            #     # mlp_test_len_vectors, mlp_test_len_labels = get_mlp_data(
            #     #     _equations_true, _equations_false, base_model, out_rules,
            #     #     abduced_map, shape)
            #     # result = mlp_model.evaluate(mlp_test_len_vectors,
            #     #                             mlp_test_len_labels,
            #     #                             batch_size=MLP_BATCHSIZE,
            #     #                             verbose=0)
            #     _len=len(mlp_test_len_vectors)
            #     sum_res+=result[1]*_len
            #     sum_len+=_len
            #     results.append(result[1])
            #     lens.append(_len)
            #     print("The result of testing length %d equations is:" %
            #           (equations_type + 5),file=text_file)
            #     print(result,file=text_file)
            #     print("The result of testing length %d equations is:" %
            #           (equations_type + 5))
            #     print(result)
            #     text_file.flush()
            # print('total_avg:',sum_res/sum_len)
            # print('total_avg:',sum_res/sum_len,file=text_file)
            r.writerow(dic)
            csv_file.flush()
            text_file.flush()
        
        if total_time==500:
            break
        # best_accuracy = 0
        # X, Y = get_img_data(src_path, labels, shape)
        # # maps = gen_mappings([0, 1, 2, 3], [0, 1, 2, 3])
        # print('\nTesting...',file=text_file)
        # print('\nTesting...')
        # # We don't know the map, so we try all maps and get the best accuracy
        # # for mapping in maps:
        # real_Y = []
        # for y in Y:
        #     real_Y.append(np.argmax(y))
        # Y_cate = np_utils.to_categorical(real_Y, num_classes=len(labels))
        # loss, accuracy = base_model.evaluate(X, Y_cate, verbose=0)
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        # print('Neural network perception accuracy: ', best_accuracy,file=text_file)
        # print('Neural network perception accuracy: ', best_accuracy)
    return base_model


def arg_init():
    parser = argparse.ArgumentParser()
    #LOGIC_OUTPUT_DIM = 50 #The mlp vector has 50 dimensions
    parser.add_argument(
        '--LOD',
        dest="LOGIC_OUTPUT_DIM",
        metavar="LOGIC_OUTPUT_DIM",
        type=int,
        default=50,
        help='The last mlp feature vector dimensions, default is 50')

    #EQUATION_LEAST_LEN = 5 #Only learn the equations of length 5-8
    parser.add_argument(
        '--ELL',
        dest="EQUATION_LEAST_LEN",
        metavar='EQUATION_LEAST_LEN',
        type=int,
        default=5,
        help='Equation least (minimum) length for training, default is 5')

    #EQUATION_MAX_LEN = 8 #Only learn the equations of length 5-8
    parser.add_argument(
        '--EML',
        dest="EQUATION_MAX_LEN",
        metavar='EQUATION_MAX_LEN',
        type=int,
        default=10,
        help='Equation max length for training, default is 8')

    #SELECT_NUM = 10 #Select 10 equations to abduce rules
    parser.add_argument(
        '--SN',
        dest="SELECT_NUM",
        metavar='SELECT_NUM',
        type=int,
        default=3,
        help=
        'Every time pick SELECT_NUM equations to abduce rules, default is 10')

    # Proportion of train and validation = 3:1
    #PROP_TRAIN = 3
    parser.add_argument(
        '--PT',
        dest="PROP_TRAIN",
        metavar='PROP_TRAIN',
        type=int,
        default=3,
        help='Proportion of train and validation rate, default PROP_TRAIN is 3'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=64,
    )
    #PROP_VALIDATION = 1
    parser.add_argument(
        '--PV',
        dest="PROP_VALIDATION",
        metavar='PROP_VALIDATION',
        type=int,
        default=1,
        help=
        'Proportion of train and validation rate, default PROP_VALIDATION is 1'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="mnist")

    #CONSISTENT_PERCENTAGE_THRESHOLD = 0.9
    parser.add_argument('--CCT', dest="CONSISTENT_PERCENTAGE_THRESHOLD", metavar='CONSISTENT_PERCENTAGE_THRESHOLD', type=float, default=0.9, \
      help='Consistent percentage threshold, which decision whether training goes to next stage, default is 0.9')

    #BATCH_LABEL_MODEL_PRECISION_THRESHOLD = 0.9 #If consistent percentage is higher than 0.9 and model precision higher than 0.9, then the condition is satisfied
    parser.add_argument('--BLMPT', dest="BATCH_LABEL_MODEL_PRECISION_THRESHOLD", metavar='BATCH_LABEL_MODEL_PRECISION_THRESHOLD', type=float, default=0.9, \
      help='If consistent percentage is higher than BATCH_LABEL_MODEL_PRECISION_THRESHOLD and model precision higher than BATCH_LABEL_MODEL_PRECISION_THRESHOLD, then the condition is satisfied, default is 0.9')
    #CONDITION_CNT_THRESHOLD = 5       #If the condition has been satisfied 5 times, the start validation
    parser.add_argument('--CPT', dest="CONDITION_CNT_THRESHOLD", metavar='CONDITION_CNT_THRESHOLD', type=int, default=5, \
      help='If the condition has been satisfied CONSISTENT_PERCENTAGE_THRESHOLD times, the start validation, default is 5')
    #NEXT_COURSE_ACC_THRESHOLD = 0.86  #If the validation accuracy of a course higher than the threshold, then go to next course
    parser.add_argument('--NCAT', dest="NEXT_COURSE_ACC_THRESHOLD", metavar='NEXT_COURSE_ACC_THRESHOLD', type=float, default=0.75, \
      help='If the validation accuracy of a course higher than the threshold, then go to next course, default is 0.86')

    #SAMPLES_PER_RULE = 3 # Use 3 samples to abduce a rule when training mlp
    parser.add_argument(
        '--SPR',
        dest="SAMPLES_PER_RULE",
        metavar='SAMPLES_PER_RULE',
        type=int,
        default=3,
        help=
        'Use SAMPLES_PER_RULE samples to abduce a rule when training mlp, default is 3'
    )
    #NN_BATCHSIZE = 32    # Batch size of neural network
    parser.add_argument('--NB',
                        dest="NN_BATCHSIZE",
                        metavar='NN_BATCHSIZE',
                        type=int,
                        default=32,
                        help='Batch size of neural network, default is 32')
    #NN_EPOCHS = 10       # Epochs of neural network
    parser.add_argument('--NE',
                        dest="NN_EPOCHS",
                        metavar='NN_EPOCHS',
                        type=int,
                        default=10,
                        help='Epochs of neural network, default is 10')
    #MLP_BATCHSIZE = 128  # Batch size of mlp
    parser.add_argument('--MB',
                        dest="MLP_BATCHSIZE",
                        metavar='MLP_BATCHSIZE',
                        type=int,
                        default=128,
                        help='Batch size of mlp, default is 128')
    #MLP_EPOCHS = 60      # Epochs of mlp
    parser.add_argument('--ME',
                        dest="MLP_EPOCHS",
                        metavar='MLP_EPOCHS',
                        type=int,
                        default=10,
                        help='MLP_EPOCHS, default is 60')

    parser.add_argument('--src_dir',
                        metavar='dataset dir',
                        type=str,
                        default="../dataset",
                        help="Where store the dataset")
    parser.add_argument('--src_data_name',
                        type=str,
                        default="_images",
                        help="Dataset name")
    parser.add_argument('--height', type=int, default=28, help='Img height')
    parser.add_argument('--weight', type=int, default=28, help='Img weight')
    parser.add_argument('--channel',
                        type=int,
                        default=1,
                        help='Img channel num')
    parser.add_argument('--num',
                        type=int,
                        default=5)
    parser.add_argument('--epoch',
                        type=int,
                        default=10)
    parser.add_argument('--T',
                        type=float,
                        default=1)
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Pretrain_epochs, default is 10')
    #parser.add_argument('--pl_file_path', type=str, default="logic/prolog/learn_add.pl", help="Which prolog file will be used")
    parser.add_argument(
        '--src_data_file',
        type=str,
        default="_equation_data_train_len_26_test_len_26_sys_2_.pk",
        help="This file is generated by equation_generator.py")
    parser.add_argument(
        '--src_data_str_file',
        type=str,
        default="_equation_data_train_len_26_test_len_26_sys_2_str.pk",
        help="This file is generated by equation_generator.py")
    parser.add_argument(
        '--hint',
        type=str,
        default="")
    parser.add_argument(
        '--budget',
        type=int,
        default=10)
    parser.add_argument(
        '--sup',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--pretrain',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--pretrain_encoder',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--balance',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=''
    )
    parser.add_argument(
        '--device',
        type=str,
        default='2'
    )

    parser.add_argument(
        '--snn',
        type=str,
        default='lstm'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # here the "labels" are just dir names for storing handwritten images of the 4 symbols
    labels = ['0', '1', '10', '11'] # 可能出现的标签 # 两个数据集：简单的例子：二进制加减法 0 1 10：+ 11：=
    args = arg_init()
    src_dir = args.src_dir
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.src_data_name = args.dataset+args.src_data_name
    input_shape = (args.height, args.weight, args.channel) # 图像的形状
    pl_file_path = "logic/prolog/learn_add.pl" # 规则
    #pl_file_path = args.pl_file_path
    args.src_data_file = args.dataset+args.src_data_file
    args.src_data_str_file = args.dataset+args.src_data_str_file
    src_path = os.path.join(src_dir, args.src_data_name)
    '''
    seed=0
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    '''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    # config.gpu_options.visible_device_list='1,2,3'
    # tf.config.experimental.set_visible_devices([gpus[3]],'GPU')
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    # Unsupervised pre-train for the CNN to help it distinguish between different symbols
    if args.pretrain:
        if args.sup:
            net_model_pretrain_sup(src_path=src_path,
                               labels=labels,
                               src_data_name=args.src_data_name,
                               shape=input_shape,
                               pretrain_epochs=args.pretrain_epochs,num=args.num,balance=args.balance)
        else:
            net_model_pretrain(src_path=src_path,
                               labels=labels,
                               src_data_name=args.src_data_name,
                               shape=input_shape,
                               pretrain_epochs=args.pretrain_epochs,num=args.num,balance=args.balance)
    # Abductive Learing main function
    model = nlm_main_func(labels=labels,
                          src_data_name=args.src_data_name,
                          src_data_file=args.src_data_file,
                          src_data_str_file=args.src_data_str_file,
                          pl_file_path=pl_file_path,
                          shape=input_shape,src_path=src_path,
                          args=args)

    # test_nn_model(model, src_path, labels, input_shape)
