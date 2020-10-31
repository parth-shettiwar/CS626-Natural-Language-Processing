import numpy as np
import nltk
from nltk.stem.porter import *
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.corpus import conll2000
import copy

train_file = "train.txt"
test_file = "test.txt"
output_train_file = "train_features.txt"

train_sent_words=[]

with open(train_file,'r') as f:
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
            
# Word2vec

sents = [list(zip(*sent))[0] for sent in train_sent_words]

model = Word2Vec(sents, min_count=1)
words = list(model.wv.vocab)
model.save('model.bin')
new_model = Word2Vec.load('model.bin')


# POS

pos_tags = [list(zip(*sent))[1] for sent in train_sent_words]
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
def create_one_hot_CL(cl):
    x = [0]*3
    x[chunk_tags.index(cl)] = 1
    return x

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
    if(temp<=57 and temp >=47):
        ret[0] = 1
    #Check First Upper Cap 
    if(temp<=90 and temp>=65):
        ret[1] = 1 
    #Check LOwer Cap  
    if(len(word)<=3):
        ret[2] = 1 
        len_flag = 1
    if(temp2==39 or temp3==39): 
      # print(word)
        ret[9] = 1
        poss_flag = 1
    if((temp==46 or temp==96 or temp==95 or temp ==94 or temp==58 or temp ==59 or temp ==33 or temp==34  or temp ==44)):
        ret[3] = 1  
    return ret

morph_features = list(map(lambda x: list(map(lambda y:give_morphological_features(y), x)),sents))
# print(morph_features[0])

with open(output_train_file,'w') as f:

    for i,sent in enumerate(sents):
        # Word2vec
        # print(len(chunk_labels[i]),len(sent),len(pos_tags[i]))
        feat1 = list(map(lambda x,y,z: list(new_model[x])+ create_one_hot_POS(y)+create_one_hot_CL(z),sent,pos_tags[i],chunk_labels[i]))
        feat2 = feat1.copy(); feat2.insert(0,[0]*len(feat1[0]))
        feat3 = feat2.copy(); feat3.insert(0,[0]*len(feat1[0]))
        total_feat = list(map(lambda x:x[0]+x[1],zip(list(map(lambda x: x[0]+x[1]+x[2],zip(feat1,feat2,feat3))),morph_features[i])))
        for i,word in enumerate(sent):
            f.write(word + " " + " ".join(map(str,total_feat[i]))+'\n')
        f.write('\n')
