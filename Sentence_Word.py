import warnings
warnings.filterwarnings('ignore')

from pickle import load
from pickle import dump
import random
import nltk
import numpy as np
from tensorflow.python.keras.models import load_model


model = load_model('model_word_final.h5')
tokenizer = load(open('Tokenizer_final.pkl','rb'))



def sample(preds, temperature=1.0):
    # Sample an index (pasted code).
    preds = np.asarray(preds).astype("float64")
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



def get_sent(model, tokenizer,start_word, n_tokens):
    generated=[start_word]
    sent = start_word
    for i in range(n_tokens):
        x_preds=tokenizer.texts_to_sequences([sent])[0]
        preds=model.predict(x_preds,verbose=0)
        word_index=sample(preds[-1,0,:])
        for word, index in tokenizer.word_index.items():
            if index==word_index:
                output=word
                break
        generated.append(output)
        sent = sent + ' ' + output
    return ' '.join(generated)



sent = get_sent(model,tokenizer,'the',12)
print(sent)

