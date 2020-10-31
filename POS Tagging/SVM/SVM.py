import numpy as np
import nltk
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.stem.porter import *
 
stemmer = PorterStemmer() 

data = nltk.corpus.brown.tagged_sents(tagset='universal')


x_train = []
y_train = []
y_train_lab = []
dict_lab = {'VERB':0,'NOUN':1,'PRON':2,'ADJ':3,'ADV':4,'DET':5,'ADP':6,'PRT':7,'NUM':8,'CONJ':9,'X':10,'.':11}
lis_noun_suff = ['acy','al','ance','ence','dom','er','or','ism','ist','ity','ty','ment','ness','ship','tion','sion']
lis_verb_suff = ['ate','en','ify','fy','ize','ise','ed','ing','in','ted']
lis_adv_suff = ['ly','ward','wise']
lis_adj_suff = ['able','ible','al','esque','ful','ic','ical','ious','ous','ish','ive','less','y']
lis_pref = []
#Data Retrieval
for i in range(len(data)):
  x_train.append([0] *len(data[i]))
  for j in range(len(data[i])):
    x_train[i][j] = data[i][j][0]
for i in range(len(data)):
  y_train.append([0] *len(data[i]))
  y_train_lab.append([0] *len(data[i]))
  for j in range(len(data[i])):
    y_train[i][j] = data[i][j][1]
    y_train_lab[i][j] = dict_lab[y_train[i][j]]
print(y_train_lab[1])    
set1 = []
set2 = []
set3 = []
set4 = []
set5 = []
print(x_train[0])
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

print(len(x_train))
model = Word2Vec(x_train, min_count=1)
print(model)
words = list(model.wv.vocab)
print(len(words))
print(model['sentence'])
model.save('model.bin')
new_model = Word2Vec.load('model.bin')
print(new_model)

print(new_model['hello'])

y_onehot = []
print(y_train_lab[5000])
for i in range(len(y_train_lab)):
  k = []
  for j in range(len(y_train_lab[i])):
    k.append([0]*12) 
  for j in range(len(y_train_lab[i])):
    k[j][y_train_lab[i][j]] = 1
  y_onehot.append(k)
  
print((y_onehot[1][4]))

#Data Preprocessing
case_vec = []  
for i in range(len(x_train)):
  kk = np.zeros([len(x_train[i]),10])
  word_vec = np.zeros([len(x_train[i]),110])
  for j in range(len(x_train[i])):
    word = x_train[i][j]
    temp = 0
    temp = ord(word[0])
    temp2 = -5
    temp3 = -5
    if(len(word)>2):
      temp2 = ord(word[len(word)-2])
      temp3 = ord(word[len(word)-3])
    flag = 0
    flag2 = 0
    verb_flag = 0
    len_flag = 0
    poss_flag = 0
    #Suffix
    bi_suf = word[len(word)-2:len(word)]
    tri_suf = word[len(word)-3:len(word)]
    four_suf = word[len(word)-4:len(word)]
    stem = stemmer.stem(word)
    if((bi_suf in lis_adj_suff or tri_suf in lis_adj_suff or four_suf in lis_adj_suff) and stem!=word):
      kk[j][5] = 1
      # print("adj ",word)
    elif((bi_suf in lis_noun_suff or tri_suf in lis_noun_suff or four_suf in lis_noun_suff) and stem!=word):
      kk[j][6] = 1
      # print("noun ",word)
    elif((bi_suf in lis_adv_suff or tri_suf in lis_adv_suff or four_suf in lis_adv_suff) and stem!=word):
      kk[j][7] = 1  
      flag2 = 1
      # print("adverb ",word)   
    elif((bi_suf in lis_verb_suff or tri_suf in lis_verb_suff or four_suf in lis_verb_suff) and stem!=word):
      kk[j][8] = 1
      verb_flag = 1
      # print("verb ",word)

    #Check Hyphen
    for jj in range(len(word)-1):
      if(ord(word[jj])==45 and ord(word[jj+1])!=45):
        kk[j][4] = 1 
        flag = 1
        # print(word)
        break
    #Check Number
    if(temp<=57 and temp >=47):
      kk[j][0] = 1
    #Check Full Cap   
    # elif(temp2<=90 and temp2>=65):
      # kk[j][1] = 1
    #Check First Upper Cap 
    if(temp<=90 and temp>=65):
      kk[j][1] = 1 
    #Check LOwer Cap  
    if(len(word)<=3):
      kk[j][2] = 1 
      len_flag = 1
    if(temp2==39 or temp3==39): 
      # print(word)
      kk[j][9] = 1
      poss_flag = 1
    if((temp==46 or temp==96 or temp==95 or temp ==94 or temp==58 or temp ==59 or temp ==33 or temp==34  or temp ==44)):
      kk[j][3] = 1  
 
    word_vec[j][:100] = new_model[word] 
    word_vec[j][100:110] = kk[j]


  case_vec.append(word_vec)

print(case_vec[0][0])

print(len(case_vec[0][0]))

case_vec2 = []
 for i in range(len(case_vec)):
   word_vec2 = np.zeros([len(case_vec[i]),330])
   for j in range(len(case_vec[i])):
     word_vec2[j][:110] =  case_vec[i][j]      
     if(j>0 and j!=len(case_vec[i])-1):
      word_vec2[j][110:220] =  case_vec[i][j-1]
     if(j>=0 and j!=len(case_vec[i])-1):
      word_vec2[j][220:330] =  case_vec[i][j+1]
   case_vec2.append(word_vec2)    
print(case_vec2[0][0])

flatList = [ item for elem in case_vec2 for item in elem]
flatList2 = [ item for elem in y_train_lab for item in elem]
flatList3 = [ item for elem in x_train for item in elem]
print(flatList3[0])
print(len(flatList[0]))
print(flatList2[0])

from tempfile import TemporaryFile

with open('f1.npy', 'wb') as f1:
    np.save(f1, flatList)
with open('f2.npy', 'wb') as f2:
    np.save(f2, flatList2)
with open('f3.npy', 'wb') as f3:
    np.save(f3, flatList3)        
with open('f1.npy', 'rb') as f4:
    a = np.load(f4)
print(a.shape)

import numpy as np
with open('f1.npy', 'rb') as f4:
  a = np.load(f4) 
with open('f2.npy', 'rb') as f5:
  b = np.load(f5)

def test(zz, X):
  probs = np.dot(X, zz)
  pred = probs.argmax(axis=1)
  return pred

#SVM Algo
import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from numpy import linalg as LA
def train(x_train,y_train):
  alpha = np.zeros((12,x_train.shape[0]),dtype=np.float64)
  weights = np.zeros((12,x_train.shape[1]),dtype=np.float64)
  threshold = 0.00001
  v_max = -1000000
  iter = 0
  v_avg = 0
  while((v_avg>threshold or (iter==0)) and  iter<20):
    vsum = 0
    for i in range(x_train.shape[0]):
      margin = np.ones((12))
      margin[y_train[i]] = 0
      C_matrix = np.zeros((12))
      C_matrix[y_train[i]] = 1

      grad = np.dot(x_train[i],weights.T) + margin
    
      vv = grad.max()-grad[alpha[:,i]<C_matrix].min()
   
      
      vsum = vsum + vv

      if(vv>0.00000001):
        x_norm = LA.norm(x_train[i])
  
        bet = x_norm*(C_matrix - alpha[:,i]) + (grad)/(x_norm)
        Low = 0
        Up = np.max(bet)
        s= np.inf
        maxi = 0
        while((np.abs(s/x_norm)>0.0001 or s>=0) and maxi<=1000):
          theta = (Low + Up)/2
          s =  np.sum(np.maximum(bet - theta,0)) - x_norm 
          maxi = maxi + 1
          if s <= 0:
            Up = theta
          else:
            Low = theta
        
          
        Del = C_matrix - alpha[:,i] - (np.maximum(bet - theta,0))/x_norm
        
        
        alpha[:,i] = alpha[:,i] + Del
        weights = weights + np.dot(Del.reshape(12,1),x_train[i].reshape(1,x_train.shape[1]))
      
        
    if(iter ==0):
      vinit = vsum  
    print(vinit)
    print(vsum)  
    v_avg = vsum/vinit  
    print(v_avg)    
    print("iter",iter)
    iter=iter+1
  return weights  
if __name__ == '__main__':

    print(a.shape)
    print(b.shape)

    from sklearn.model_selection import KFold
    kfold = KFold(5, True, 1)
    fin_pred = []
    test_lab = []
    for train1, test1 in kfold.split(a,b):
      print(a[train1].shape)
      print(type(a[test1]))
      weigh = train(a[train1],b[train1])
      fin_pred.append(test(weigh.T, a[test1]))
      test_lab.append(b[test1])
    print(d[:6])

#Testing

from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,12))
Acc = np.zeros((5))
for i in range(5):
  for j in range(len(test_lab[i])):
    if(test_lab[i][j]==fin_pred[i][j]):
      Acc[i] = Acc[i]+1
  Acc[i] = Acc[i]/len(test_lab[i])
print("5 fold Acc",Acc) 
print("Avg Accuracy",np.sum(Acc)/5)     
y_pred = np.array(list(itertools.chain.from_iterable(fin_pred)))
y_tr = np.array(list(itertools.chain.from_iterable(test_lab)))
  
mat = confusion_matrix(y_tr, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11])

labels=['VERB','NOUN','PRON','ADJ','ADV','DET','ADP','PRT','NUM','CONJ','X','END']
ax.matshow(mat, cmap=plt.cm.Blues)
lis = [-1,0,1,2,3,4,5,6,7,8,9,10,11]
ax.set_xticks(lis)
ax.set_yticks(lis)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
for i in range(12):
  for j in range(12):
    c = mat[j,i]
    ax.text(i, j, str(c), va='bottom', ha='center')
    ax.text(i,j,str(round((c/mat.sum()*100),2))+"%",va='top', ha='center')
plt.xlabel('Predicted')
plt.ylabel('True')
per_Pos = np.zeros((12))    
for i in range(12):
  per_Pos[i] = mat[i][i]/np.sum(mat[i])
print("per_pos",per_Pos)
