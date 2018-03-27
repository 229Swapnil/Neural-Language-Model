import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Embedding
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras import metrics
from keras.preprocessing.text import Tokenizer
import random
import nltk
from nltk.corpus import gutenberg
import collections as coll
import numpy as np
from pickle import load
from pickle import dump



def get_split(data):
    ######## Text Preprocessing
    sent_list = []
    for i in range(len(data)):
        sent = data[i]
        sent = [word.lower() for word in sent]
        sent = [word for word in sent if word.isalpha()==True]
        sent_list.append(sent)
    train_sent = sent_list[0:int(0.9*len(sent_list))]
    test_sent = sent_list[int(0.9*len(sent_list)):len(sent_list)]
    return train_sent, test_sent



def get_data_1(train_sents, maxlen):
    word_list = []
    for i in range(len(train_sents)):
        for words in train_sents[i]:
            word_list.append(words)
    
    sequence=[]
    stride=1
    #applying windowing for sequence genration

    for i in range(0,len(word_list)-maxlen,stride):
        line=word_list[i:i+maxlen]
        sequence.append(line)
    
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(sequence)
    seq=tokenizer.texts_to_sequences(sequence)
    vocab_len=len(tokenizer.word_index.items())+1
    
    seq=np.array(seq)
    x_train=seq[:,:-1]
    y_train=np.zeros((x_train.shape[0],x_train.shape[1],1))
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            y_train[i,j,0]=seq[i,j+1]
        
    return x_train,y_train,vocab_len,tokenizer


def get_test_data_1(test_sents,maxlen,tokenizer):    
    vocab = []
    for words,_ in tokenizer.word_index.items():
        vocab.append(words)
    
    word_list = []
    for i in range(len(test_sents)):
        text = [word for word in test_sents[i] if word in vocab]
        word_list += text
        
    sequence=[]
    stride=1
    #applying windowing for sequence genration

    for i in range(0,len(word_list)-maxlen,stride):
        line=word_list[i:i+maxlen]
        sequence.append(' '.join(line))
        
    seq=tokenizer.texts_to_sequences(sequence)
    seq=np.array(seq)
    x_test=seq[:,:-1]
    y_test=np.zeros((x_test.shape[0],x_test.shape[1],1))
    for i in range(x_test.shape[0]):
        for j in range(x_test.shape[1]):
            y_test[i,j,0]=seq[i,j+1]
        
    return x_test,y_test,len(sequence)  



def get_model_1(x,y,Vocab_size,maxlen):
    model = Sequential()
    model.add(Embedding(Vocab_size, 30)) #input_length=maxlen))
    model.add(LSTM(60, return_sequences=True))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(Vocab_size))
    model.add(Activation("softmax"))
    print(model.summary())
    optimizer = RMSprop(lr=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')
    return model



############ Finding Perplexity
def get_perplexity(model, x_test,y_test,N):
    perp = 1
    for i in range(x_test.shape[0]):
        preds = model.predict(x_test[i,:], verbose=0)
        if preds[-1,0,int(y_test[i,49,0])]>0.0001
        perp = perp*((1/preds[-1,0,int(y_test[i,49,0])])**N)
    return perp



def sample(preds, temperature=1.0):
    # Sample an index (pasted code).
    preds = np.asarray(preds).astype("float64")
    preds = preds/np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



def get_sent(model, tokenizer,start_word, n_tokens):
    generated=['the']    
    for i in range(n_tokens):
        sent = start_word
        x_preds=tokenizer.texts_to_sequences([sent])[0]
        preds=model.predict(x_preds,verbose=0)
        word_index=sample(preds[-1,0,:])
        for word, index in tokenizer.word_index.items():
            if index==word_index:
                output=word
                break
        generated.append(output)
        sent+=output
    return ' '.join(generated)



################ Main ###############
data = list(gutenberg.sents('austen-emma.txt')) + list(gutenberg.sents('austen-persuasion.txt')) + list(gutenberg.sents('austen-sense.txt')) 

maxlen = 51

train_sent,test_sent = get_split(data)
x,y,Vocab_size,tokenizer = get_data_1(train_sent,maxlen)
model = get_model_1(x,y,Vocab_size,maxlen)



model.fit(x, y, batch_size=50, epochs=30)



from pickle import dump
model.save('Model.h5')
dump(tokenizer,open('Tokenizer.pkl','wb'))



x_test,y_test,N = get_test_data_1(test_sent,maxlen,tokenizer)



perp = get_perplexity(model, x_test,y_test,1/N)



print(perp)


sent = get_sent(model,tokenizer,'the',50)
print(sent)

