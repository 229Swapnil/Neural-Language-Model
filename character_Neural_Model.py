import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Embedding
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras import metrics
import random
import nltk
from nltk.corpus import gutenberg
import collections as coll
import numpy as np
from pickle import load
from pickle import dump

def get_data_1(data, maxlen):
  
    chars = sorted(list(set(data)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    step = 1
    n_samples = 0
    for i in range(0, len(data) - maxlen, step):
        n_samples += 1
    
    x = np.zeros((n_samples, maxlen))
    y = np.zeros((n_samples))

    for i in range(n_samples):
        for j in range(maxlen):
            x[i,j] = char_indices[data[i+j]]
        y[i] = char_indices[data[i+maxlen]]
    return x,y,char_indices,indices_char

def get_test(data,maxlen,char_indices):
    test_lower = []
    for i in range(len(data)):
        if data[i] in char_indices.keys():
            test_lower += data[i]
     
    step = 4
    n_samples = 0
    for i in range(0, len(data) - maxlen, step):
        n_samples += 1
        
    x = np.zeros((n_samples, maxlen))
    y = np.zeros((n_samples))

    for i in range(n_samples):
        for j in range(maxlen):
            x[i,j] = char_indices[data[i+j]]
        y[i] = char_indices[data[i+maxlen]]
    return x,y,n_samples

def get_perp(model,x_test,y_test,N):
    perp = 1
    for i in range(len(y_test)):
        preds = model.predict(x_test[i,:].reshape(1,-1), verbose=0)[0]
        perp = perp*((1/preds[int(y_test[i])])**N)
    return perp

def get_model_1(x,y,Vocab_size,maxlen):
    model = Sequential()
    model.add(Embedding(Vocab_size, 50, input_length=maxlen))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(Vocab_size))
    model.add(Activation("softmax"))
    print(model.summary())
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',  metrics=['accuracy'])
    return model

def sample(preds, temperature=1.0):
    # Sample an index (pasted code).
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def get_sentence(model, train_data, indices_char, char_indices, maxlen):
    start_index = random.randint(0, len(train_data) - maxlen - 1)
    sentence = train_data[start_index: start_index + maxlen]

    generated = ''

    for i in range(100):        
        x_pred = np.zeros((1,maxlen))
        for j in range(len(sentence)):
            x_pred[0,j] = char_indices[sentence[j]]

    # Predict the next term.
        preds = model.predict(x_pred, verbose=0)[0]

    # Get next index and then get the next value from the index.
        next_index = sample(preds, 1)
        next_char = indices_char[next_index]

    # Update our sentence.
        sentence += next_char
        sentence = sentence[1:]
    
        generated += next_char
    return generated

data = gutenberg.raw('austen-emma.txt') + gutenberg.raw('austen-persuasion.txt') + gutenberg.raw('austen-sense.txt')

data = data.lower()

train_data = data[0:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):len(data)]

maxlen = 50

x,y,char_indices,indices_char = get_data_1(train_data,maxlen)
x_test, y_test,N = get_test(test_data,maxlen,char_indices)
model = get_model_1(x,y,len(char_indices.keys()),maxlen)

model.fit(x, y, batch_size=512, epochs=10)

perp = get_perp(model,x_test,y_test,1/N)

print(perp)

sent = get_sentence(model, train_data, indices_char, char_indices, maxlen)
print(sent)

model.save('Model2_char.h5')
dump(indices_char,open('indices_char.pkl','wb'))
dump(char_indices,open('char_indices.pkl','wb'))
