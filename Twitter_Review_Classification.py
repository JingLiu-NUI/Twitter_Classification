#!/usr/bin/env python
# coding: utf-8

# In this task you will develop a system to detect irony in text. We will use the data from the SemEval-2018 task on irony detection. You should use the file `SemEval2018-T3-train-taskA.txt` from Blackboard it consists of examples as follows:

# ```csv
# Tweet index     Label   Tweet text
# 1       1       Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion  http://t.co/fej2v3OUBR
# 2       1       @mrdahl87 We are rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)
# 3       1       Hey there! Nice to see you Minnesota/ND Winter Weather 
# 4       0       3 episodes left I'm dying over here
# ```
# 

# # Task 1 (5 Marks)
# 
# Read all the data and find the size of vocabulary of the dataset (ignoring case) and the number of positive and negative examples.

# In[ ]:


from google.colab import drive
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download("all")
drive.mount('/content/drive')
get_ipython().system('Is "/content/drive/My Drive/Colab Notebooks"')
buf = open("/content/drive/My Drive/Colab Notebooks/SemEval2018-T3-train-taskA.txt",encoding="ISO-8859-1")
def dataset_split():
  label=[]
  label0=[]
  label1=[]
  corpus=[]
  vocab=[]
  original_data=[]
  total_sentence=[]
  temp_original=[]
  lists=buf.read().split('\n')
  for line in range(len(lists)):
    if line!=0:
      list_of_doc=lists[line].split('\t')
      temp_original.append(list_of_doc)
      label.append(list_of_doc[1])  
      if(list_of_doc[1]=='0'):
        label0.append(list_of_doc[1])
      else:
        label1.append(list_of_doc[1])
      corpus.append(list_of_doc[2])
  print("negative label size:", len(label0))
  print("positive label size:", len(label1))

  for sen in corpus:
    ss=[]
    sentence_list=nltk.word_tokenize(sen)
    for vol in sentence_list:
      vocab.append(vol.lower())
      ss.append(vol.lower())
    total_sentence.append(ss)
  for i in range(len(total_sentence)):
    original_data.append((int(temp_original[i][0]),int(temp_original[i][1]),total_sentence[i]))
  vocabulary=set(vocab)
  return total_sentence,label,vocabulary,original_data


# # Task 2 (20 Marks)
# 
# Develop a classifier using the Naive Bayes model to predict if an example is ironic. The model should convert each Tweet into a bag-of-words and calculate
# 
# $p(\text{Ironic}|w_1,\ldots,w_n) \propto \prod_{i=1,\ldots,n} p(w_i \in \text{tweet}| \text{Ironic}) p(\text{Ironic})$
# 
# $p(\text{NotIronic}|w_1,\ldots,w_n) \propto \prod_{i=1,\ldots,n} p(w_i \in \text{tweet}| \text{NotIronic}) p(\text{NotIronic})$
# 
# You should use add-alpha smoothing to calculate probabilities

# In[ ]:


#this is my bag of word, firtly get training setence and label and vocabulery of dataset,
#then using counter function to caculate the frequency of each word in each sentence,
#then change each sentence length become vacabulary and if the world in sentence existing in vacabulary
#then change the location of that word in vector to the frequency otherwise append 0,then get a new
#training matrix.
#I did this with MiaoLi
from numpy import*
from collections import Counter
def load_dataset():
  train_list=[]
  total_sentence,label,vocabulary,original_data=dataset_split()
  for sentence in total_sentence:
      bow=[]
      frequencies_words=Counter(sentence)
      for voc in vocabulary:    
          if voc in sentence:
              bow.append(frequencies_words[voc])
          else:
              bow.append(0) 
      train_list.append(bow)
  train_matrix=array(train_list)
  return train_matrix,label,original_data
#the bayes function is firstly using the matrix of train and lebal to training the model,
#here I use the ones to initialize the matrix of p(wi∈tweet|Ironic),on the matrix, caculate each conlums for the words and 
#caculate all worlds which belong to label positive and negative respectively, using the probability of each attribute and then divide by their 
#total sum of each label, and here can get the p(w1,w2..|Positive), the initail total number is 1 and the ones() funcion all used for avoiding the  
#probability are 0.
def train(trainMatrix,labelMatrix):
    label00=[]
    label11=[]
    numTrainRow = trainMatrix.shape[0]
    numTrainCol=trainMatrix.shape[1]
    for i in labelMatrix:
      if i=="0":
        label00.append(i)
      else:
        label11.append(i)
    num_train_label1=len(label11)
    p0Num = ones(numTrainCol)
    p1Num = ones(numTrainCol)
    p1label=num_train_label1/numTrainRow
    totalNum_1=2.0
    totalNum_0=2.0
    k=0.5
    for i in range(numTrainRow):
        if labelMatrix[i]=='1':
          p1Num+=trainMatrix[i]
          #number of total features for label=1
          totalNum_1+=sum(trainMatrix[i])
        else:
          p0Num+=trainMatrix[i]
          totalNum_0+=sum(trainMatrix[i])
    #p(w1|C0)
    vec0=log(p0Num/totalNum_0)
    #p(w1,w2|C1)
    vec1=log(p1Num/totalNum_1)
    return vec0,vec1,p1label
#after traing a model, here using the the test data to testing, the vec are each test veactor,
#it contains 0,1 and the if its 0 which means the probability are 0, if there is non-0,then get the 
#relevent probabilty of this world based on the training probability, using(W1...n|Positive) and (W1..n|Negative)
#mutiply the probablity of positive or negative to get which one is bigger, Lastly get the label for bigger probabilty. 
def NB_testing(data,vec0,vec1,p1label):
    prediction=[]
    for vec in data:
      p_pos=sum(vec*vec1)+log(p1label)
      p_neg=sum(vec*vec0)+log((1-p1label))
      if p_pos>p_neg:
        prediction.append('1')
      else:
        prediction.append('0')
        
    return prediction
 


# # Task 3 (15 Marks)
# 
# Divide the data into a training and test set and justify your split.
# 
# Choose a suitable evaluation metric and implement it. Explain why you chose this evaluation metric.
# 
# Evaluate the method in Task 2 according to this metric.

# In[ ]:


#this is for anncuracy, accuracy is a evaluation metric of the classfication, I only 
# caculate the correct numebrs of corract divided by the total numbers of labels
#the reason why I use accuracy is because the classification are clear to evalate how many 
#the origthm predict correct and how many the predict wrong, its useful to evaluate the classification.
dataset,label,original_dataset=load_dataset()
train_size=4/5*len(dataset)
train_size=int(train_size)
trainX=dataset[:train_size]
trainY=label[:train_size]
testX=dataset[train_size:]
testY=label[train_size:]
vec0,vec1,p1label=train(trainX,trainY)
def accuracy_evaluation(prediction,testY):
    correctNum=0
    accuracy=0.0
    for i in range(len(testY)):
          if str(testY[i])==str(prediction[i]):
            correctNum+=1
    accuracy=correctNum/len(testY)
    return accuracy
  
prediction=NB_testing(testX,vec0,vec1,p1label)
accuracy_ba=accuracy_evaluation(prediction,testY)
print("accuracy for bayis",accuracy_ba)
def original_data():
    original_train=original_dataset[:train_size]
    original_test=original_dataset[train_size:]
    return original_train,original_test
  


# # Task 4 (20 Marks)
# 
# Run the following code to generate a model from your training set. The training set should be in a variable  called `train` and is assumed to be of the form:
# 
# ```
# [(1, 1, ['sweet', 'united', 'nations', 'video', '.', 'just', 'in', 'time', 'for', 'christmas', '.', '#', 'imagine', '#', 'noreligion', 'http', ':', '//t.co/fej2v3oubr']), 
#  (2, 1, ['@', 'mrdahl87', 'we', 'are', 'rumored', 'to', 'have', 'talked', 'to', 'erv', "'s", 'agent', '...', 'and', 'the', 'angels', 'asked', 'about', 'ed', 'escobar', '...', 'that', "'s", 'hardly', 'nothing', ';', ')']), 
#  (3, 1, ['hey', 'there', '!', 'nice', 'to', 'see', 'you', 'minnesota/nd', 'winter', 'weather']), 
#  (4, 0, ['3', 'episodes', 'left', 'i', "'m", 'dying', 'over', 'here']), 
#  ...
# ]
#  ```
# 
# 

# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import optimizers

## These values should be set from Task 3
train, test = original_data()

def make_dictionary(train, test):
    dictionary = {}
    for d in train+test:
        for w in d[2]:
            if w not in dictionary:
                dictionary[w] = len(dictionary)
    return dictionary

class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.current_sent = 0
        self.skip_step = skip_step
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, 2))
        while True:
            for i in range(self.batch_size):
                # Choose a sentence and position with at lest num_steps more words
                while self.current_idx + self.num_steps >= len(self.data[self.current_sent][2]):
                    self.current_idx = self.current_idx % len(self.data[self.current_sent][2])
                    self.current_sent += 1
                    if self.current_sent >= len(self.data):
                        self.current_sent = 0
                # The rows of x are set to values like [1,2,3,4,5]
                x[i, :] = [self.vocabulary[w] for w in self.data[self.current_sent][2][self.current_idx:self.current_idx + self.num_steps]]
                # The rows of y are set to values like [[1,0],[1,0],[1,0],[1,0],[1,0]]
                y[i, :, :] = [[self.data[self.current_sent][1], 1-self.data[self.current_sent][1]]] * self.num_steps
                self.current_idx +=self.skip_step
            yield x, y

# Hyperparameters for model
vocabulary = make_dictionary(train, test)
num_steps =5
batch_size = 20
num_epochs = 50 # Reduce this if the model is taking too long to train (or increase for performance)
hidden_size = 50 # Increase this to improve perfomance (or increase for performance)
use_dropout=True

# Create batches for RNN
train_data_generator = KerasBatchGenerator(train, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(test, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

# A double stacked LSTM with dropout and n hidden layers
model = Sequential()
model.add(Embedding(len(vocabulary), hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
  model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

#categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])

# Train the model
model.fit_generator(train_data_generator.generate(), len(train)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(test)//(batch_size*num_steps))

# Save the model
model.save("final_model3.hdf5")


# Now consider the following code:

# In[ ]:


#Here for my logic is from the LSTM we feed 5 element each time for the input, and output are real value and 
#wrong value probability, so here for the output of test, I split and feed the sentece with 5 words each time,
#using a cont to caculate hoe many input for each sentence len(sen)/5, after prediction then combine the output probabilty
#of each sencence and then sum all probaibilty of left and right on the y[correct,wrong], then divedeved by the numbers of 
#how many output for a sentence and lastly compare the average value with the 0.5, if the left greater than 0.5, then its 
#1 and otherwise its 0.  then get the newest label and input to accuracy evalution to get the accuracy, here I got the accuracy of this
#LSTM is 0.5542328042328042
model = load_model("final_model3.hdf5")
real=[]
# x = np.zeros((1,num_steps))
# x[0,:] = [vocabulary["this"],vocabulary["is"],vocabulary["an"],vocabulary["easy"],vocabulary["test"]]
# print(x)
# print(model.predict(x))
test_data=[]
skip_step=5
num_steps=5
x=[]
y=[]
nums=[]
test2=[]
predictions=[]
for aa in range(len(test)):
  if len(test[aa][2])>num_steps:
      test2.append(test[aa])
prediction=np.zeros((len(test2),num_steps,2))
for current_sent in range(len(test2)):
    index=0
    #num=int(len(test2[current_sent][2])/num_steps)
    num=int(len(test2[current_sent][2]))-(num_steps-1)
    nums.append(num)
    for i in range(num):
          if num!=0:
            x.append([vocabulary[w] for w in test2[current_sent][2][index:index + num_steps]])
            y=[test2[current_sent][1], 1-test2[current_sent][1]] * num_steps
#             index+=1                
xarr=np.array(x)
prediction=model.predict(xarr) 
pre=[]

another_pred=np.zeros((len(test2),num_steps,2))
i=0
for num in nums:
  sum=0
  for x in range(num):
      for j in range(prediction.shape[1]):
              sum+=prediction[i][j][0]
      i=i+1        
  if num!=0:
    ave=sum/(num*num_steps)
  else:
    ave=1
  if ave>0.5:
     pre.append(1)
  else:
     pre.append(0)
label_test=[]
for i in test2:
  label_test.append(i[1])
accuracy_LSTM=accuracy_evaluation(pre,label_test)

print("accuracy for LSTM:",accuracy_LSTM)


# Using the code above write a function that can predict the label using the LSTM model above and compare it with the evaluation performed in Task 3

# # Task 5 (40 Marks)
# 
# Suggest an improvement to either the system developed in Task 2 or 4 and show that it improves according to your evaluation metric.
# 
# Please note this task is marked according to: demonstration of knowledge from the lecutures (10), originality and appropriateness of solution (10), completeness of description (10), technical correctness (5) and improvement in evaluation metric (5).

# In[ ]:


#for the improvement, here I change the skip step and number epoch and convert the
#current_idx to 0, firstly, I think the reason why the accuracy is lower because the 
#training set instance are small, if I change the skip step to 1, then here I will get 
#for example, there is 1 sentence which have 10 words, for the orighnal way, there are only 2 input
#instance, but if change it to 1, there will have 6 training instance can be trainning the dataset,
#then bigger training dataset let the LSTM learn more thing and for the current_idx, its same, if every times
#when start a new sentence, then start from the first word will better than start with self.current_idx % len(self.data[self.current_sent][2])
#for the epoch I also choose lots of different number then I found the 60 is best for my nural network.
#here when I changed the parameter, the testing accuracy from 0.55 to 0.61 or 0.6243386243386243

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import optimizers

## These values should be set from Task 3
train, test = original_data()

from nltk.corpus import stopwords
# def data_without_stopword(text):
#   text_new=[]
#   for sen in text:
#     fitered1=[w for w in sen[2] if(w not in stopwords.words('english'))]
#     fitered=[w for w in fitered1 if(w.isalpha())]
#     text_new.append((sen[0],sen[1],fitered1))
#   return text_new

# train=data_without_stopword(train)
# test=data_without_stopword(test)

def make_dictionary(train, test):
    dictionary = {}
    for d in train+test:
        for w in d[2]:
            if w not in dictionary:
                dictionary[w] = len(dictionary)
    return dictionary

class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.current_sent = 0
        self.skip_step = skip_step
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, 2))
        while True:
            for i in range(self.batch_size):
                # Choose a sentence and position with at lest num_steps more words
                while self.current_idx + self.num_steps >= len(self.data[self.current_sent][2]):
                    self.current_idx = 0
                    self.current_sent += 1
                    if self.current_sent >= len(self.data):
                        self.current_sent = 0
                # The rows of x are set to values like [1,2,3,4,5]
                x[i, :] = [self.vocabulary[w] for w in self.data[self.current_sent][2][self.current_idx:self.current_idx + self.num_steps]]
                # The rows of y are set to values like [[1,0],[1,0],[1,0],[1,0],[1,0]]
                y[i, :, :] = [[self.data[self.current_sent][1], 1-self.data[self.current_sent][1]]] * self.num_steps
                self.current_idx +=1
            yield x, y

# Hyperparameters for model
vocabulary = make_dictionary(train, test)
num_steps =5
batch_size = 20
num_epochs = 60 # Reduce this if the model is taking too long to train (or increase for performance)
hidden_size = 50 # Increase this to improve perfomance (or increase for performance)
use_dropout=True

# Create batches for RNN
train_data_generator = KerasBatchGenerator(train, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(test, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

# A double stacked LSTM with dropout and n hidden layers
model = Sequential()
model.add(Embedding(len(vocabulary), hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
  model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
#categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])
# Train the model
model.fit_generator(train_data_generator.generate(), len(train)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(test)//(batch_size*num_steps))
# Save the model
model.save("final_model4.hdf5")


# In[ ]:



#accuracy for LSTM: 0.6243386243386243
model = load_model("final_model4.hdf5")
real=[]
# x = np.zeros((1,num_steps))
# x[0,:] = [vocabulary["this"],vocabulary["is"],vocabulary["an"],vocabulary["easy"],vocabulary["test"]]
# print(x)
# print(model.predict(x))
test_data=[]
skip_step=5
num_steps=5
x=[]
y=[]
nums=[]
test2=[]
predictions=[]
for aa in range(len(test)):
  if len(test[aa][2])>num_steps:
      test2.append(test[aa])
prediction=np.zeros((len(test2),num_steps,2))
for current_sent in range(len(test2)):
    index=0
    #num=int(len(test2[current_sent][2])/num_steps)
    num=int(len(test2[current_sent][2]))-(num_steps-1)
    nums.append(num)
    for i in range(num):
          if num!=0:
            x.append([vocabulary[w] for w in test2[current_sent][2][index:index + num_steps]])
            y=[test2[current_sent][1], 1-test2[current_sent][1]] * num_steps
#             index+=1                
xarr=np.array(x)
prediction=model.predict(xarr) 
pre=[]
# for i in range(prediction.shape[0]):
#   for j in range(prediction.shape[1]):
#     print(prediction[i][j])
#     print(y[i][j])

another_pred=np.zeros((len(test2),num_steps,2))
i=0
for num in nums:
  sum=0
  first_po=0.0
  second_po=0.0
  for x in range(num):
      for j in range(prediction.shape[1]):
              sum+=prediction[i][j][0]
      i=i+1        
  if num!=0:
    ave=sum/(num*num_steps)
  else:
    ave=1
  if ave>0.5:
     pre.append(1)
  else:
     pre.append(0)
label_test=[]
for i in test2:
  label_test.append(i[1])
accuracy_LSTM=accuracy_evaluation(pre,label_test)

print("accuracy for LSTM:",accuracy_LSTM)
              


# In[ ]:


import matplotlib.pyplot as plt
#from the confusionmatricx we can see how many should be positive and we predict to positive and negitive
#how many should be negitive and we predict it to negative
def  confusion_matricx(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for x in range(len(y_true)):
        if y_true[x] == 0 and y_pred[x] == 0:
            TN += 1
        if y_true[x] == 0 and y_pred[x] == 1:
            FP += 1
        if y_true[x] == 1 and y_pred[x] == 1:
            TP += 1
        if y_true[x] == 1 and y_pred[x] == 0:
            FN += 1
    cm = [[TN, FP],[FN, TP]]
    cm=np.array(cm)
    return cm
comfuse = confusion_matricx(label_test,pre)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(comfuse, cmap=plt.cm.Blues, alpha=0.3)
for i in range(comfuse.shape[0]):
    for j in range(comfuse.shape[1]):
        ax.text(x=j, y=i, s=comfuse[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()


# In[ ]:


#here I also use the precision score1,
#The proportion of the predicted results that meets the actual value can be understood as the case of no "false positives".
#get the True negitive, true positive, flase positive and false negitive and check the probability for true positive in positive case.

def precision_score1(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for x in range(len(y_true)):
        if y_true[x] == 0 and y_pred[x] == 0:
            TN += 1
        if y_true[x] == 0 and y_pred[x] == 1:
            FP += 1
        if y_true[x] == 1 and y_pred[x] == 1:
            TP += 1
        if y_true[x] == 1 and y_pred[x] == 0:
            FN += 1
        x += 1
    p=TP/float((TP+FP))
    return p
  
print('Precision: %.3f' % precision_score1(y_true=label_test, y_pred=pre))
# recall score are : 
#The ratio of the correct number of classifications to the number of all “should” be correctly 
#classified (conforming to the target label) can be understood as the absence of “missing” in the recall rate.
def recall_score(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for x in range(len(y_true)):
        if y_true[x] == 0 and y_pred[x] == 0:
            TN += 1
        if y_true[x] == 0 and y_pred[x] == 1:
            FP += 1
        if y_true[x] == 1 and y_pred[x] == 1:
            TP += 1
        if y_true[x] == 1 and y_pred[x] == 0:
            FN += 1
        x += 1
    p=TP/float((TP+FN))
    return p 
print('Recall: %.3f' % recall_score(y_true=label_test, y_pred=pre))
#the F1 value is the harmonic mean of the precision rate and the recall rate
def F1_score(y_true, y_pred):
    P=precision_score1(y_true, y_pred)
    R=recall_score(y_true, y_pred)
    F=2*P*R/float(P+R)
    return F
print('F1: %.3f' % F1_score(y_true=label_test, y_pred=pre))

