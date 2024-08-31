#!/usr/bin/env python
# coding: utf-8

# # TASK - 1:

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer


# In[2]:


train = pd.read_csv("D://M_AI//AI Capstone//Ecommerce//train_data.csv")
test_val = pd.read_csv("D://M_AI//AI Capstone//Ecommerce//test_data_hidden.csv")
test = pd.read_csv("D://M_AI//AI Capstone//Ecommerce//test_data.csv")
train.shape, test_val.shape, test.shape


# In[3]:


train.head()


# In[4]:


#Checking for null values
train.isnull().sum()


# In[5]:


#using bfill method to remove the null values
train.fillna(method='bfill',axis=0, inplace = True)
train.isnull().sum()


# In[6]:


#Checking the sentiment value counts
train['sentiment'].value_counts()


# In[7]:


#Checking the name value counts
pd.DataFrame(train.name.value_counts())


# In[8]:


#Checkin value counts for each class
cols = ['name','brand','categories','primaryCategories','reviews.date','reviews.text','reviews.title','sentiment']
for i in cols:
    p = pd.DataFrame(train[i].value_counts())
    print(p)


# In[9]:


#define a function to clean the sentence
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

ps = PorterStemmer()

def clean_text(text):
    wt = word_tokenize(text)
    text = [word for word in text if word not in string.punctuation]
    text = ''.join(text)
    text = [ps.stem(word) for word in text.split() if word.lower() not in stopwords.words('english')]
    return text


# In[10]:


#Cleaning the Train Dataset

#Use Tfidf vectorizer to convert into features
reviews_train = TfidfVectorizer(analyzer = clean_text).fit(train['reviews.text'])
print(len(reviews_train.vocabulary_))
reviews_train_transform = reviews_train.transform(train['reviews.text'])


# In[11]:


# Storing the reviews using Tfidf Transformer
train_tfidf_trans = TfidfTransformer().fit(reviews_train_transform)
reviews_train_tfidf = train_tfidf_trans.transform(reviews_train_transform)
print(reviews_train_tfidf.shape)

words_train = reviews_train.get_feature_names()
print(words_train)


# In[12]:


#Cleaning the Test_val data Set

#Use Tfidf vectorizer to convert into features
reviews_testval = TfidfVectorizer(analyzer = clean_text).fit(test_val['reviews.text'])
print(len(reviews_testval.vocabulary_))
reviews_testval_transform = reviews_testval.transform(test_val['reviews.text'])

# Storing the reviews using Tfidf Transformer
testval_tfidf_trans = TfidfTransformer().fit(reviews_testval_transform)
reviews_testval_tfidf = testval_tfidf_trans.transform(reviews_testval_transform)
print(reviews_testval_tfidf.shape)


# In[13]:


#Cleaning the Test data Set

#Use Tfidf vectorizer to convert into features
reviews_test = TfidfVectorizer(analyzer = clean_text).fit(test['reviews.text'])
print(len(reviews_test.vocabulary_))
reviews_test_transform = reviews_test.transform(test['reviews.text'])

# Storing the reviews using Tfidf Transformer
test_tfidf_trans = TfidfTransformer().fit(reviews_test_transform)
reviews_test_tfidf = test_tfidf_trans.transform(reviews_test_transform)
print(reviews_test_tfidf.shape)


# In[14]:


from wordcloud import WordCloud, STOPWORDS 

#all_text = ' '.join([text for text in train['reviews.text']])
pos_text = ' '.join([text for text in train['reviews.text'][train['sentiment']=='Positive']])
neg_text = ' '.join([text for text in train['reviews.text'][train['sentiment']=='Negative']])
neu_text = ' '.join([text for text in train['reviews.text'][train['sentiment']=='Neutral']])

##########################

wordcloud = WordCloud(width=4000, height=1200, random_state=21, max_font_size=180).generate(pos_text)
plt.figure(figsize=(20,25))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(' POSITIVE REVIEWS')
plt.show()

##########################

wordcloud = WordCloud(width=4000, height=1200, random_state=21,max_font_size=180).generate(neu_text)
plt.figure(figsize=(20,25))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('NEUTRAL REVIEWS')
plt.show()

##########################

wordcloud = WordCloud(width=4000, height=1200, random_state=21,max_font_size=180).generate(neg_text)
plt.figure(figsize=(20,25))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(' NEGATIVE REVIEWS')
plt.show()


# In[15]:


# Split the data for classification
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( reviews_train_tfidf,train.sentiment,train_size=0.8,random_state = 123 )


# In[16]:


# Run Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

review_model = MultinomialNB().fit(train_X,train_y)
pred = review_model.predict(test_X)

# Plotting confusion matrix
cm = plot_confusion_matrix(review_model,test_X,test_y, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# Getting the classification report
print("Naive Bayes Classification Report:\n\n",classification_report(test_y, pred))


# # TASK - 2:

# In[17]:


#As we can see from the above confusion matrix most of negative and neutral reviews are classified as postive
# So this is a class imbalance problem


# In[18]:


#UnderSampling using Random Under Sampler from imbalanced learn

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state = 0)
#cc.fit(train_X,train_y)
X_resampled, y_resampled = rus.fit_resample(reviews_train_tfidf,train.sentiment)

print(X_resampled.shape)
print(y_resampled.shape)

X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,train_size = 0.8, random_state = 123)


# In[19]:


#Using Random Forest Classifier for Under-Sampled Data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rfc = RandomForestClassifier()
a = rfc.fit(X_train,y_train)

pred1 = rfc.predict(X_test)

# Plotting confusion matrix
cm1 = plot_confusion_matrix(rfc,X_test,y_test, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#AUC-ROC Score
print("The AUC-ROC Score is :", roc_auc_score(y_test,rfc.predict_proba(X_test),multi_class='ovr'))

# Getting the classification report
print("Random Forest Under Sampling Classification Report:\n\n",classification_report(y_test, pred1))


# In[20]:


#Using XGBoost Classifier for Under-Sampled Data


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb = XGBClassifier()
b = xgb.fit(X_train,y_train)

pred2 = xgb.predict(X_test)

# Plotting confusion matrix
cm2 = plot_confusion_matrix(xgb,X_test,y_test, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#AUC-ROC Score
print("The AUC-ROC Score is :", roc_auc_score(y_test,xgb.predict_proba(X_test),multi_class='ovr'))

# Getting the classification report
print("XG Boost Under Sampling Classification Report:\n\n",classification_report(y_test, pred2))


# In[21]:


#OverSampling using Random Over Sampler from imbalanced learn
from imblearn.over_sampling import RandomOverSampler


rus = RandomOverSampler(random_state = 0)
#cc.fit(train_X,train_y)
X_resampled1, y_resampled1 = rus.fit_resample(reviews_train_tfidf,train.sentiment)

print(X_resampled1.shape)
print(y_resampled1.shape)

X1_train, X1_test, y1_train, y1_test = train_test_split(X_resampled1,y_resampled1,train_size = 0.8, random_state = 123)


# In[22]:


#Using Random Forest Classifier for Over-Sampled Data

rfc1 = RandomForestClassifier()
c = rfc1.fit(X1_train,y1_train)

pred3 = rfc1.predict(X1_test)

# Plotting confusion matrix
cm3 = plot_confusion_matrix(rfc1,X1_test,y1_test, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#AUC-ROC Score
print("The AUC-ROC Score is :", roc_auc_score(y1_test,rfc.predict_proba(X1_test),multi_class='ovr'))

# Getting the classification report
print("Random Forest Over Sampling Classification Report:\n\n",classification_report(y1_test, pred3))


# In[23]:


#Using XGBoost Classifier for Over-Sampled Data

xgb1 = XGBClassifier()
d = xgb1.fit(X1_train,y1_train)

pred4 = xgb1.predict(X1_test)

# Plotting confusion matrix
cm4 = plot_confusion_matrix(xgb1,X1_test,y1_test, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#AUC-ROC Score
print("The AUC-ROC Score is :", roc_auc_score(y1_test,xgb1.predict_proba(X1_test),multi_class='ovr'))

# Getting the classification report
print("XGBoost Over Sampling Classification Report:\n\n",classification_report(y1_test, pred4))


# In[24]:


#Accuracy Scores for the above used models
from sklearn.metrics import accuracy_score
score1 = accuracy_score(y_test,pred1)
score2 = accuracy_score(y_test,pred2)
score3 = accuracy_score(y1_test,pred3)
score4 = accuracy_score(y1_test,pred4)
print("Accuracy Scores:\n Random Forest Under Sampling: {}\n XGBoost Under Sampling: {}\n Random Forest Over Sampling: {}\n XGBoost Over Sampling: {}".format(score1,score2,score3,score4))


# # TASK 3:

# In[25]:


## APPLYING LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['sentiment']= le.fit_transform(train['sentiment'])
train.head()


# In[26]:


from keras.layers import Input,Conv2D,Dense,Conv2DTranspose,LeakyReLU,Reshape
from keras.layers import Flatten,BatchNormalization,MaxPool2D,Dropout, Embedding,LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
import tensorflow as tf



# In[151]:


#Initialize Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
reviews = train['reviews.text'].tolist()
sentiment = train['sentiment'].tolist()

#Let tokenizer look at all the text
a = tokenizer.fit_on_texts(words_train)

#Vocablury
len(tokenizer.word_index)
#print(a)


# In[152]:


#Convert text into numbers
features = tokenizer.texts_to_matrix(reviews, mode='tfidf')
print(features.shape)
print(features)


# In[153]:


#Initialize model, reshape & normalize data
model3 = tf.keras.models.Sequential()

#normalize data
model3.add(tf.keras.layers.BatchNormalization(input_shape=(5000,)))

#Add Dense Layers
model3.add(tf.keras.layers.Dense(200))
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.BatchNormalization())

model3.add(tf.keras.layers.Dense(100))
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.BatchNormalization())

model3.add(tf.keras.layers.Dense(60))
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.BatchNormalization())

model3.add(tf.keras.layers.Dense(30))
model3.add(tf.keras.layers.LeakyReLU())
model3.add(tf.keras.layers.BatchNormalization())


# In[154]:


#Output layer
model3.add(tf.keras.layers.Dense(1, activation='softmax'))


# In[155]:


#Compile the model
adam_op = tf.keras.optimizers.Adam(lr=0.001, decay=0.001) #momentum=0.9, nesterov=True)
model3.compile(optimizer= adam_op, loss='categorical_crossentropy', metrics=['accuracy'])


# In[156]:


#X3_train, X3_test, y3_train, y3_test = train_test_split(features, train['sentiment'], test_size=0.2, random_state=42)


# In[33]:


#Convert output labels to multiple values
#y3_train = tf.keras.utils.to_categorical(y3_train)
#y3_test = tf.keras.utils.to_categorical(y3_test)


# In[159]:


#Train the model
history3 = model3.fit(features,train['sentiment'], epochs=30,batch_size=16)         
          


# # TASK - 4:

# In[40]:


#Build the tokenizer
top_words = 10000
t = Tokenizer(num_words=top_words) # num_words -> Vocablury size
t.fit_on_texts(reviews)


# In[41]:


#Clean text
import re, string

def clean_str(string):
  """
  String cleaning before vectorization
  """
  try:    
    string = re.sub(r'^https?:\/\/<>.*[\r\n]*', '', string, flags=re.MULTILINE)
    string = re.sub(r"[^A-Za-z]", " ", string)         
    words = string.strip().lower().split()    
    words = [w for w in words if len(w)>=1]
    return " ".join(words)
  except:
    return ""


# In[42]:


train['clean_review'] = train['reviews.text'].apply(clean_str)
train.head()


# In[43]:


#Get the word index for each of the word in the review
X2_train, X2_test, y2_train, y2_test = train_test_split(train['clean_review'], train['sentiment'], test_size=0.2, random_state=42)
X2_train = t.texts_to_sequences(X2_train.tolist())
X2_test = t.texts_to_sequences(X2_test.tolist())


# In[44]:


#Pad Sequences - Important
from tensorflow.python.keras.preprocessing import sequence

max_review_length = 1000
X2_train = sequence.pad_sequences(X2_train,maxlen=max_review_length,padding='post')
X2_test = sequence.pad_sequences(X2_test, maxlen=max_review_length, padding='post')


# In[45]:


#List to hold all words in each review
documents = []

#Iterate over each review
for doc in train['clean_review']:
    documents.append(doc.split(' '))

print(len(documents))


# In[46]:


import gensim
#Build the model
word2vec = gensim.models.Word2Vec(documents, #Word list
                               min_count=10, #Ignore all words with total frequency lower than this                           
                               workers=4, #Number of CPU Cores
                               size=50,  #Embedding size
                               window=5, #Maximum Distance between current and predicted word
                               iter=10   #Number of iterations over the text corpus
                              )  


# In[47]:


embedding_vector_length = word2vec.wv.syn0.shape[1]
embedding_vector_length


# In[48]:


embedding_matrix = np.zeros((top_words + 1, embedding_vector_length))
embedding_matrix.shape


# In[49]:


for word, i in sorted(t.word_index.items(),key=lambda x:x[1]):
    if i > top_words:
        break
    if word in word2vec.wv.vocab:
        embedding_vector = word2vec.wv[word]
        embedding_matrix[i] = embedding_vector


# In[50]:


model = Sequential()
model.add(Embedding(top_words + 1,
                    embedding_vector_length,
                    input_length=max_review_length,
                   weights=[embedding_matrix],
                   trainable=False)
         )
#Add Layer with 100 LSTM Memory Units

model.add(LSTM(100))
#model.add(LSTM(100))
model.add(Dense(1,activation='softmax'))


# In[51]:


opt = Adam(lr=0.01, decay=0.001, beta_1=0.5)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])


# In[52]:


model.summary()


# In[53]:


history = model.fit(X2_train,y2_train,
          epochs=5,
          batch_size=32,          
          validation_data=(X2_test, y2_test))


# In[55]:


from keras.layers import GRU

model1 = Sequential()
model1.add(Embedding(top_words + 1,
                    embedding_vector_length,
                    input_length=max_review_length,
                   weights=[embedding_matrix],
                   trainable=False)
         )
#Add a layer with 100 GRU units
model1.add(GRU(100))
model1.add(Dense(1,activation='softmax'))


# In[115]:


from keras.optimizers import SGD
opt1 = SGD(lr=0.01, decay=0.001) #beta_1=0.5)
model1.compile(optimizer=opt1,loss='categorical_crossentropy',metrics=['accuracy'])


# In[116]:


model1.summary()


# In[117]:


history1 = model1.fit(X2_train,y2_train,
          epochs=5,
          batch_size=32,          
          validation_data=(X2_test, y2_test))


# In[ ]:


#Comparing the accuracies of ML and Neural net models


# In[149]:


print("Accuracy Scoresof ML Models:\n Random Forest Under Sampling: {}\n XGBoost Under Sampling: {}\n Random Forest Over Sampling: {}\n XGBoost Over Sampling: {}".format(score1,score2,score3,score4))


# In[169]:


lstm_a = max(history.history['accuracy'])
gru_a = max(history1.history['accuracy'])
dnn_a = max(history3.history['accuracy'])


# In[170]:


print("Accuracy Scores of Neural Nets: \n DNN: {}\n LSTM: {}\n GRU: {}".format(dnn_a,lstm_a,gru_a))


# In[118]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[144]:


#Experiment with three hyperparameters in the model:

#Number of units in the first dense layer
#Dropout rate in the dropout layer
#Optimizer

from tensorboard.plugins.hparams import api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd','adam']))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([ 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


# In[145]:


def train_test_model(hparams):
    model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
     tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
    model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

    model.fit(X2_train, y2_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(X2_test, y2_test)
    return accuracy


# In[146]:


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# In[147]:


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
             hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1


# In[148]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/hparam_tuning')


# In[ ]:




