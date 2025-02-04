# -*- coding: utf-8 -*-
"""CRF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LK2MPDYVskcVKbHsnG5bJKjNurclAuH8
"""

import numpy as np
import nltk
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow
import numpy
from collections import Counter
# !pip install git+https://www.github.com/keras-team/keras-contrib.git
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000
from keras.utils.data_utils import get_file
from zipfile import ZipFile
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import *
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import conll2000
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow
import numpy
import keras
from keras import Input
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.datasets import conll2000
from keras.utils.data_utils import get_file
import copy

 
train_file = "train.txt"
test_file = "test.txt"
output_train_file = "train_features.txt"

def filex(name):
  train_sent_words=[]

  with open(name,'r') as f:
      lines = f.readlines()
      sent = []
      for line in lines:
          line = line.rstrip()
          if line.rstrip() == "":
              train_sent_words.append(sent)
              sent = []
          else:
              line = line.split(" ")
              word = []
              seps = line[2].split("-")
              if len(seps) == 2:
                  word.append(line[0]);word.append(line[1]);word.append(seps[0]);word.append(seps[1])
              else:
                  word.append(line[0]);word.append(line[1]);word.append(seps[0]);word.append("")
              sent.append(word)
  return train_sent_words
             
            
# Word2vec
train_sent_words = filex(train_file)
test_sent_words = filex(test_file)
sents = [list(zip(*sent))[0] for sent in train_sent_words]
sents2 = [list(zip(*sent))[0] for sent in test_sent_words]

model = Word2Vec(sents+sents2, min_count=1)
words = list(model.wv.vocab)
model.save('model.bin')
new_model = Word2Vec.load('model.bin')


# POS

pos_tags = [list(zip(*sent))[1] for sent in train_sent_words]
pos_tags2 = [list(zip(*sent))[1] for sent in test_sent_words]

flat_list = set([item for sublist in pos_tags for item in sublist])
unique_tag_list = list(flat_list)
number_of_tags = len(unique_tag_list)

def create_one_hot_POS(pos):
    x = [0]*number_of_tags
    x[unique_tag_list.index(pos)] = 1
    return x

# Chunk labels 

chunk_tags = ['B','I','O']
chunk_labels = [list(zip(*sent))[2] for sent in train_sent_words]
chunk_labels2 = [list(zip(*sent))[2] for sent in test_sent_words]
def create_one_hot_CL(cl):
    x = [0]*3
    x[chunk_tags.index(cl)] = 1
    return chunk_tags.index(cl)

# Morphological features

lis_noun_suff = ['acy','al','ance','ence','dom','er','or','ism','ist','ity','ty','ment','ness','ship','tion','sion']
lis_verb_suff = ['ate','en','ify','fy','ize','ise','ed','ing','in','ted']
lis_adv_suff = ['ly','ward','wise']
lis_adj_suff = ['able','ible','al','esque','ful','ic','ical','ious','ous','ish','ive','less','y']

stemmer = PorterStemmer() 

def give_morphological_features(word):
    ret = [0]*10
    temp = 0
    temp = ord(word[0])
    temp2 = -5
    temp3 = -5
    if(len(word)>2):
        temp2 = ord(word[-2])
        temp3 = ord(word[-3])
    flag = 0
    flag2 = 0
    verb_flag = 0
    len_flag = 0
    poss_flag = 0
    #Suffix
    bi_suf = word[-2:]
    tri_suf = word[-3:]
    four_suf = word[-4:]
    stem = stemmer.stem(word)
    if((bi_suf in lis_adj_suff or tri_suf in lis_adj_suff or four_suf in lis_adj_suff) and stem!=word):
        ret[5] = 1
    elif((bi_suf in lis_noun_suff or tri_suf in lis_noun_suff or four_suf in lis_noun_suff) and stem!=word):
        ret[6] = 1
    elif((bi_suf in lis_adv_suff or tri_suf in lis_adv_suff or four_suf in lis_adv_suff) and stem!=word):
        ret[7] = 1 
        flag2 = 1
    elif((bi_suf in lis_verb_suff or tri_suf in lis_verb_suff or four_suf in lis_verb_suff) and stem!=word):
        ret[8] = 1
        verb_flag = 1
    #Check Hyphen
    for jj in range(len(word)-1):
        if(ord(word[jj])==45 and ord(word[jj+1])!=45):
            ret[4] = 1 
            flag = 1
            break
    #Check Number
    if(temp<=57 and temp >47):
        ret[0] = 1
    #Check First Upper Cap 
    if(temp<=90 and temp>=65):
        ret[1] = 1 
    #Check LOwer Cap  
    if(len(word)<=3):
        ret[2] = 1 
        len_flag = 1
    if(temp2==39 or temp3==39): 
        ret[9] = 1
        poss_flag = 1
    if((temp==46 or temp==96 or temp==95 or temp ==94 or temp==58 or temp ==59 or temp ==33 or temp==34  or temp ==44)):
        ret[3] = 1  
    return ret

morph_features = list(map(lambda x: list(map(lambda y:give_morphological_features(y), x)),sents))
morph_features2 = list(map(lambda x: list(map(lambda y:give_morphological_features(y), x)),sents2))

from keras.preprocessing.sequence import pad_sequences
total_feat = []
for i,sent in enumerate(sents):
  feat1 = list(map(lambda x,y: list(new_model[x])+ create_one_hot_POS(y),sent,pos_tags[i]))
  total_feat.append(list(map(lambda x:x[0]+x[1],zip(feat1,morph_features[i]))))
total_feat2 = []
for i,sent in enumerate(sents2):
  feat1 = list(map(lambda x,y: list(new_model[x])+ create_one_hot_POS(y),sent,pos_tags2[i]))
  total_feat2.append(list(map(lambda x:x[0]+x[1],zip(feat1,morph_features2[i]))))  
maxlen = max(len(s) for s in total_feat)  
total_feat = pad_sequences(total_feat, maxlen = 78,value = [0.0]*154,padding = 'post',dtype = 'float32')
total_feat2 = pad_sequences(total_feat2, maxlen = 78,value = [0.0]*154,padding = 'post',dtype = 'float32')

y_train = []
for i in range(len(chunk_labels)):
  lab = list(map(lambda x:chunk_tags.index(x),chunk_labels[i]))
  y_train.append(np.asarray(lab))
y_test = []
for i in range(len(chunk_labels2)):
  lab = list(map(lambda x:chunk_tags.index(x),chunk_labels2[i]))
  y_test.append(np.asarray(lab))
y_train = pad_sequences(y_train, maxlen = 78,value = 3,padding = 'post',dtype = 'float32')
y_test = pad_sequences(y_test, maxlen = 78,value = 3,padding = 'post',dtype = 'float32')

total_feat = np.asarray(total_feat)
final = np.zeros((8936,78,154*5))
for i in range(len(total_feat)):
  for j in range(78):
    final[i][j][:154] = total_feat[i][j]  
    if(j<77):      
      final[i][j][154*2:154*3] = total_feat[i][j+1]
    if(j>0):      
      final[i][j][154:154*2] = total_feat[i][j-1]
      final[i][j][154*5:154*5+1] = y_train[i][j-1]
    if(j<76):      
      final[i][j][154*4:154*5] = total_feat[i][j+2]
    if(j>1):      
      final[i][j][154*3:154*4] = total_feat[i][j-2]  

total_feat2 = np.asarray(total_feat2)
final2 = np.zeros((len(total_feat2),78,154*5))
for i in range(len(total_feat2)):
  for j in range(78):
    final2[i][j][:154] = total_feat2[i][j]  
    if(j<77):      
      final2[i][j][154*2:154*3] = total_feat2[i][j+1]
    if(j>0):      
      final2[i][j][154:154*2] = total_feat2[i][j-1]
    if(j<76):      
      final2[i][j][154*4:154*5] = total_feat2[i][j+2]
    if(j>1):      
      final2[i][j][154*3:154*4] = total_feat2[i][j-2]



#CRF Train
crf = CRF(4, sparse_target=True)
inputs = Input(shape=(78,770, ))
outputs = crf(inputs)
model = keras.Model(inputs, outputs)
model.summary()
model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
total_feat = np.asarray(total_feat)
y_train = np.asarray(y_train)
y_train = y_train.squeeze()
y_train = np.expand_dims(y_train, axis=2)
x = final[:7700]
valx = final[7700:]
y = y_train[:7700,:,:]
valy = y_train[7700:]
model.fit(x, y, epochs=15,validation_data=[valx,valy])

total_feat2 = np.asarray(total_feat2)
y_test = np.asarray(y_test)
y_test = np.expand_dims(y_test, axis=2)
test_y_pred = model.predict(final2)
test_y_true = y_test
chunk_tags = ["B","I","O"]
ss=np.argmax(test_y_pred,axis=2)
coun = 0
coun2=0
valy = valy.squeeze()
for i in range(len(y_test)):
  for j in range(78):
    if(y_test[i][j]<2):
      coun2 = coun2+1
      if(y_test[i][j]==ss[i][j]):
        coun = coun+1

from sklearn.metrics import confusion_matrix
import sklearn
import itertools
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(12,12))   
ss = ss.flatten()
test_y_true  = test_y_true.flatten()
mat = confusion_matrix(test_y_true, ss, labels=[0,1,2,3])
labels=['B','I','O','Pad']
ax.matshow(mat, cmap=plt.cm.Blues)
lis = [-1,0,1,2,3]
ax.set_xticks(lis)
ax.set_yticks(lis)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
for i in range(4):
  for j in range(4):
    c = mat[j,i]
    ax.text(i, j, str(c), va='bottom', ha='center')
    ax.text(i,j,str(round((c/mat.sum()*100),2))+"%",va='top', ha='center')
plt.xlabel('Predicted')
ax.xaxis.set_label_position('top')
plt.ylabel('True')
per_Pos = np.zeros((2))    
for i in range(2):
  per_Pos[i] = mat[i][i]/np.sum(mat[i])
print("per_pos",per_Pos)

# import sys
# import tensorflow
# import keras
# print(sys.version)
# print(numpy.__version__)
# print(tensorflow.__version__)
# print(keras.__version__)