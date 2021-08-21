#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


reviews = pd.read_table("arxivdata/arxivdata.tsv",error_bad_lines=False,low_memory=False)
reviews = reviews.sample(frac=1)
pro_cat = reviews["product_category"].unique()[:]
reviews = reviews.loc[reviews['product_category'].isin(pro_cat)]
reviews = reviews.sample(frac=1)


# In[3]:


import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.models import Sequential,Model
import pyprind
from keras_self_attention import SeqSelfAttention
from gensim.models import Word2Vec
import string
import pickle

# In[4]:


DIM_SIZE = 300
w2v_model = Word2Vec.load("amazon_data/model.en")
_BOS, _EOS = "_BOS "," _EOS"


# In[5]:


def remove_punct(sentence):
    #sentence = re.sub(stop, " ", sentence) 
    return sentence.lower().translate(str.maketrans('', '', string.punctuation))

def create_dictionary(reviews):

    input_texts = [] 
    target_texts = []
    prod_category = []
    # idx2word
    input_title_vocab = set()
    target_review_vocab = set()
    #
    for index,row in reviews.iterrows():
        attributes = [row["product_title"],row["review_headline"],row["review_body"]]
        if not isinstance(row["product_title"],str): 
            continue
        if not isinstance(row["review_headline"],str): 
            continue
        if not isinstance(row["review_body"],str): 
            continue   
        input_text, target_text = remove_punct(row["product_title"]),        remove_punct(" ".join([row["review_headline"],row["review_body"]]))
        target_text = _BOS + target_text + _EOS
        input_texts.append(input_text)
        target_texts.append(target_text)
        prod_category.append(row["product_category"])
        for word in input_text.split():
            if word not in input_title_vocab:
                input_title_vocab.add(word)
        for word in target_text.split():
            if word not in target_review_vocab:
                target_review_vocab.add(word)
    
    return input_texts, target_texts, input_title_vocab, target_review_vocab, prod_category

input_texts, target_texts, input_title_vocab, target_review_vocab, prod_category = create_dictionary(reviews)


# In[6]:


vocab = sorted(list(input_title_vocab.union(target_review_vocab)))
target_review_vocab = sorted(list(target_review_vocab))
# the number of sample vocablary
encoder_vocab_size = len(input_title_vocab)
decoder_vocab_size = len(target_review_vocab)
vocab_size = len(vocab)
# Define max length of encoder / decoder
#max_encoder_seq_length = max([len(text.split()) for text in input_texts])
#max_decoder_seq_length = max([len(text.split()) for text in target_texts])
input_di_text,target_di_text = {},{}
text_index = 0
max_seq_length = 0
for text in input_texts:
    if len(text.split()) <= 100:
        input_di_text[text_index] = text
        text_index += 1
        max_seq_length = max(max_seq_length,len(text.split()))
for text in target_texts:
    if len(text.split()) <= 100:
        target_di_text[text_index] = text
        text_index += 1
        max_seq_length = max(max_seq_length,len(text.split()))
    

num_prod = len(reviews['product_category'].unique())
inverse_input_vocab = dict(
    [(word, id) for id, word in enumerate(input_title_vocab)])
inverse_target_vocab = dict(
    [(word, id) for id, word in enumerate(target_review_vocab)])
inverse_vocab = dict(
    [(word, id) for id, word in enumerate(vocab)])
inverse_prod_category = dict(
    [(word, id) for id, word in enumerate(reviews['product_category'].unique())])

pickle.dump(inverse_input_vocab,open("inverse_input_vocab.pkl","wb"))
pickle.dump(inverse_target_vocab,open("inverse_target_vocab.pkl","wb"))
pickle.dump(inverse_vocab,open("inverse_vocab.pkl","wb"))
pickle.dump(inverse_prod_category,open("inverse_prod_category.pkl","wb"))


# In[9]:


input_texts = list(input_di_text.values())
target_texts = list(target_di_text.values())
def batch_generator(batch_size):
    while True:
        encoder_input_data = np.zeros(
            (batch_size, max_seq_length, DIM_SIZE),
            dtype='float32')
        decoder_input_data = np.zeros(
            (batch_size, max_seq_length, DIM_SIZE),
            dtype='float32')
        decoder_target_data = np.zeros(
            (batch_size, max_seq_length, DIM_SIZE),
            dtype='float32')
        prod_category_data = np.zeros(
            (batch_size, 1, num_prod),
            dtype='float32')
        n_batches_for_epoch = len(input_di_text)//batch_size
        for i in range(n_batches_for_epoch):
            for pair_text_idx, (input_text, target_text,prod_cat) in enumerate(zip(input_texts[batch_size*i:batch_size*(i+1)], target_texts[batch_size*i:batch_size*(i+1)],prod_category[batch_size*i:batch_size*(i+1)])):
                for timestep, word in enumerate(input_text.split()):
                    if word in w2v_model.wv:
                        encoder_input_data[pair_text_idx, timestep] = w2v_model.wv[word]
                # decoder_target_data is ahead of decoder_input_data by one timestep
                for timestep, word in enumerate(target_text.split()):
                    if word in w2v_model.wv:
                        decoder_input_data[pair_text_idx, timestep] = w2v_model.wv[word]
                    if timestep > 0:
                        # decoder_target_data will be ahead by one timestep（LSTM)
                        if word in w2v_model.wv:
                            decoder_target_data[pair_text_idx, timestep - 1] = w2v_model.wv[word]
                prod_category_data[pair_text_idx,0, inverse_prod_category[prod_cat]] = 1.
            yield([encoder_input_data, decoder_input_data],[decoder_target_data,prod_category_data])


# In[ ]:
from keras.callbacks import TensorBoard, ModelCheckpoint


encoder_inputs = Input(shape=(None, DIM_SIZE))
encoder_bilstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(encoder_inputs)
encoder_self_attention_output = SeqSelfAttention(attention_activation='sigmoid')(encoder_bilstm)

decoder_inputs = Input(shape=(None, DIM_SIZE))
decoder_bilstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(decoder_inputs)
decoder_self_attention_output = SeqSelfAttention(attention_activation='sigmoid')(decoder_bilstm)
#decoder_flatten = Flatten()(decoder_self_attention_output)
decoder_dense = Dense(num_prod,activation="softmax")(decoder_self_attention_output)
filepath="amazon-43-weights-improvement-shuffle-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1)
model = Model(inputs=[encoder_inputs,decoder_inputs],output=[decoder_inputs,decoder_dense])
model.summary()
from keras.optimizers import Adam
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=opt, metrics=["kullback_leibler_divergence","categorical_accuracy"],loss=["binary_crossentropy","categorical_crossentropy"]) # Set up model
model.fit_generator(generator=batch_generator(32),
                    nb_epoch=40,
                    shuffle=True,
                    samples_per_epoch=len(input_di_text),
                    callbacks=[checkpoint,TensorBoard(log_dir='/tmp/autoencoder')],
                    verbose=True)


# In[ ]:





# In[ ]:




