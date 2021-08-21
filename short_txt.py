#!/usr/bin/env pytho/n
# coding: utf-8

# In[70]:


import pandas as pd


# In[71]:


reviews = pd.read_table("amazon_data/sampled_reviews.tsv",error_bad_lines=False,low_memory=False)
#reviews = reviews.sample(frac=0.25).reset_index(drop=True)
#reviews = reviews.groupby('product_category').apply(lambda x: x.sample(n=50000)).reset_index(drop = True)


# In[72]:


num_prod = len(reviews['product_category'].unique())


# In[73]:


num_prod


# In[74]:


from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import string
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec
import logging
import pickle 

# In[96]:


_BOS = "_BOS "
_EOS = " _EOS"
DIM_SIZE = 300
w2v_model = Word2Vec.load("amazon_data/model.en")


# In[84]:


def remove_punct(sentence):
    #sentence = re.sub(stop, " ", sentence) 
    return sentence.lower().translate(str.maketrans('', '', string.punctuation))

def create_dictionary(data_path):
    reviews = pd.read_table(data_path,error_bad_lines=False,low_memory=False)
    reviews = reviews.groupby('product_category').apply(lambda x: x.sample(n=10000,replace=True)).reset_index(drop = True)
    reviews = reviews.sample(frac=1).reset_index(drop=True)
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
    pickle.dump(input_title_vocab,open("input_title.pkl","wb"))
    pickle.dump(target_review_vocab,open("target_review.pkl","wb"))
    return input_texts, target_texts, input_title_vocab, target_review_vocab, prod_category

input_texts, target_texts, input_title_vocab, target_review_vocab, prod_category = create_dictionary("amazon_data/sampled_reviews.tsv")


# In[85]:


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


# In[86]:


from keras.callbacks import TensorBoard, ModelCheckpoint


# In[63]:

"""
encoder_input_data = np.zeros(
    (len(input_di_text), max_seq_length, DIM_SIZE),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(target_di_text), max_seq_length, DIM_SIZE),
   dtype='float32')
decoder_target_data = np.zeros(
    (len(target_di_text), max_seq_length, DIM_SIZE),
    dtype='float32')
prod_category_data = np.zeros(
    (len(prod_category), 1, num_prod),
    dtype='float32')

for pair_text_idx, (input_text, target_text,prod_cat) in enumerate(zip(input_texts, target_texts,prod_category)):
    for timestep, word in enumerate(input_text.split()):
        encoder_input_data[pair_text_idx, timestep, inverse_vocab[word]] = 1.
    # decoder_target_data is ahead of decoder_input_data by one timestep
    for timestep, word in enumerate(target_text.split()):
        decoder_input_data[pair_text_idx, timestep, inverse_vocab[word]] = 1.
        if timestep > 0:
            # decoder_target_data will be ahead by one timestep（LSTM)
            # decoder_target_data will not include the start character.
            decoder_target_data[pair_text_idx, timestep - 1, inverse_vocab[word]] = 1.
    prod_category_data[pair_text_idx,0, inverse_prod_category[prod_cat]] = 1.
"""

# In[95]:


NUM_HIDDEN_UNITS = 128 # NUM_HIDDEN_LAYERS
BATCH_SIZE = 32
NUM_EPOCHS = 4


# In[88]:

import tensorflow as tf
class Attention(tf.keras.Model):

    def __init__(self, units):
        # initialize initial values
        super(Attention, self).__init__()
        # architecture of nn
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        prob_score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(prob_score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


# In[97]:


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
                        decoder_target_data[pair_text_idx, timestep - 1] = w2v_model.wv[word]
                prod_category_data[pair_text_idx,0, inverse_prod_category[prod_cat]] = 1.
            yield([encoder_input_data, decoder_input_data],[decoder_target_data,prod_category_data])

from keras import backend as K
def weighted_categorical_crossentropy():
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = np.array([1]*len(reviews["product_category"].unique()))/len(reviews["product_category"].unique())
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss
"""
Encoder Architecture
"""

encoder_inputs = Input(shape=(None, DIM_SIZE))
encoder_lstm = LSTM(units=NUM_HIDDEN_UNITS,return_state=True)
# x-axis: time-step lstm
encoder_outputs,state_h,state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.

"""
Decoder Architecture
"""
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_inputs = Input(shape=(None, DIM_SIZE))
decoder_lstm = LSTM(units=NUM_HIDDEN_UNITS)
# x-axis: time-step lstm
decoder_outputs,decoder_state_h,decoder_state_c= decoder_lstm(decoder_inputs,initial_state=encoder_states,return_state=True) # Set up the decoder, using `encoder_states` as initial state.
decoder_softmax_layer = Dense(DIM_SIZE, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

classification_dense = Dense(100, activation = 'relu')(decoder_outputs)
classification_outputs = Dense(num_prod,activation = 'relu')(classification_dense)
"""
Encoder-Decoder Architecture
"""
# Define the model that will turn, `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
filepath="weights-improvement-shuffle-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1)
model = Model([encoder_inputs, decoder_inputs], [decoder_outputs,classification_outputs])
print(model.summary())
model.compile(optimizer="rmsprop", metrics=["accuracy"], loss=["categorical_crossentropy",weighted_categorical_crossentropy()]) # Set up model


model.fit_generator(generator=batch_generator(BATCH_SIZE),
                    nb_epoch=10,
                    shuffle=True,
                    samples_per_epoch=len(input_di_text),
                    callbacks=[checkpoint,TensorBoard(log_dir='/tmp/autoencoder')],
                    verbose=True)

# model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data,prod_category_data],
#           batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2) # Run training
model.save("seq2seq_translate_model.txt") # Save model






