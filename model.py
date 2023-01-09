import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Embedding, SpatialDropout1D, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import re
from keras.preprocessing.text import Tokenizer
#from keras.utils.data_utils import pad_sequences

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

import pickle

df = pd.read_csv('sentiment-tweets.csv')

dataset =  df[['text','airline_sentiment']]

dataset.rename(columns = {'airline_sentiment':'sentiment'}, inplace = True)

dataset['text'] = dataset['text'].apply(lambda x: x.lower())
dataset['text'] = dataset['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

for idx,row in dataset.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(dataset['text'].values)
X = tokenizer.texts_to_sequences(dataset['text'].values)
X = pad_sequences(X)

Y = pd.get_dummies(dataset['sentiment']).values
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
model.add(Dropout(0.5))

model.add( LSTM( lstm_out, dropout=0.2, recurrent_dropout=0.2 ) )


model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
nb_epochs = 10

history = model.fit( X_train, y_train, epochs = nb_epochs, batch_size = batch_size, verbose = 2)



twt = ['I am happy to see you']

print(twt)

def makePrediction(twt):
    #vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)

    sentiment = model.predict(twt, batch_size=32, verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        return "negative"
    elif (np.argmax(sentiment) == 1):
        return "neutre"
    else:
        return "positive"

print(makePrediction(twt) )



class SentimentAnalysis():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, twt):
        #vectorizing the tweet by the pre-fitted tokenizer instance
        twt = self.tokenizer.texts_to_sequences(twt)
        #padding the tweet to have exactly the same shape as `embedding_2` input
        twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)
        sentiment = self.model.predict(twt, batch_size=32, verbose = 2)[0]
        if(np.argmax(sentiment) == 0):
            return "negative"
        elif (np.argmax(sentiment) == 1):
            return "neutre"
        else:
             return "positive"

myModel = SentimentAnalysis(model, tokenizer)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(myModel, file)


print(myModel.predict(twt))


