'''

@author: mgdak

'''

#TODO: refactor into classes with Factory design model and interfaces

import argparse
import sys
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2VecKeyedVectors
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, Embedding, Lambda, Conv1D, MaxPooling1D
from keras.optimizers import Adam, RMSprop
from keras import callbacks, regularizers
from keras import backend as K
import tensorflow as tf

MAX_WORDS_NO = 100 #based on histogram data
WORD2VEC_NO_OF_FEATURES = 300 #number of features of a Word2Vec model
FILTER_SIZES = [3, 5]
NUM_FILTERS = [128, 256]


def initTokenizers():
    # Load stop-words
    stop_words = set(stopwords.words('english'))
    
    # Initialize tokenizer
    # It's also possible to try with a stemmer or to mix a stemmer and a lemmatizer
    tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    return lemmatizer, tokenizer, stop_words

'''
Reads the input data from DB and Tokenizes & Lemmitizes sentences
'''
def prepareDataSet(lemmatizer, tokenizer, stop_words, x_dataset, missedWords, word2index, w2v_model):
    X = np.zeros(shape=(len(x_dataset), MAX_WORDS_NO*2), dtype=int)
    
    word2indexId = 1
    idx = 0
    for question in x_dataset['question_text']:
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(question) if t.lower() not in stop_words]
        for jdx in range(len(tokens[:MAX_WORDS_NO-1])):
            word = tokens[jdx]
            if word in w2v_model:
                if word not in word2index:
                    word2index[word] = word2indexId
                    word2indexId += 1
                X[idx, jdx] = word2index[word] 
            else:
                X[idx, jdx] = 0
                missedWords.append(word)
                    
    idx = 0
    for answers in x_dataset['answer_text']:
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(answers) if t.lower() not in stop_words]
        
        for jdx in range(len(tokens[:MAX_WORDS_NO-1])):
            word = tokens[jdx]
            if word in w2v_model:
                if word not in word2index:
                    word2index[word] = word2indexId
                    word2indexId += 1
                X[idx, MAX_WORDS_NO + jdx] = word2index[word] 
            else:
                X[idx, MAX_WORDS_NO + jdx] = 0
                missedWords.append(word)

    return X


def tokenizeLemmatizeDataSet(X_train, X_test, word2vecURI):
    lemmatizer, tokenizer, stop_words = initTokenizers()
    
    #load Word2Vec model
    w2v_model = Word2VecKeyedVectors.load_word2vec_format(word2vecURI, binary=False)
    print("vocab_size = %s", len(w2v_model.vocab))
    
    #determine number of features for each word in the model
    WORD2VEC_NO_OF_FEATURES = w2v_model['dog'].shape[0]

    print("num_features = ", WORD2VEC_NO_OF_FEATURES)
    print("len(X_train) = ", len(X_train))
    print("len(X_test) = ", len(X_test))
    
    #create the list to get the all words which we are missing in the Word2Vec model
    missedWords = []
    word2index = {}
    
    X_train = prepareDataSet(lemmatizer, tokenizer, stop_words, X_train, missedWords, word2index, w2v_model)
    X_test = prepareDataSet(lemmatizer, tokenizer, stop_words, X_test, missedWords, word2index, w2v_model)
    
    print("Number of used words = ", len(set(word2index)))
    print("Number of words missing = ", len(set(missedWords)))
    
    return X_train, X_test, w2v_model, word2index


'''
 plots the distribution of senteces lengths after lemmitazation in order to decide on 
 maximum number of words. For the data provided it turned out that 300 covers 98% population
'''
def plotXHist(X_train, X_test):
    X = X_train + X_test
    values = []
    for val in X:
        lenght = len(val)
        values.append(lenght)
    
    plt.hist(values, bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    plt.show()

'''
Serializes the trained model
'''
def serializeModel(model, fileName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(fileName + ".h5")
    print("Saved model to disk")
    

'''
This is a workaround for a known bug in gensim get_keras_embedding
'''
def createKerasEmbeddingLayer(w2v_model, word2index, trainable):
    vocab_len = len(word2index) + 1
    emb_matrix = np.zeros((vocab_len, WORD2VEC_NO_OF_FEATURES))
    
    for word, index in word2index.items():
        emb_matrix[index, :] = w2v_model[word]

    embedding_layer = Embedding(vocab_len, 
                                WORD2VEC_NO_OF_FEATURES, 
                                trainable=trainable, 
                                mask_zero=True, 
                                input_shape=(MAX_WORDS_NO*2, ),
                                embeddings_regularizer = regularizers.l2(1e-7))
    
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

'''
Builds simple LSTM model woth Keras Embedding lazer
'''
def createSimpleLSTMWithEmbeddingModel(w2v_model, word2index, trainable, learning_rate, lr_decay):
    model = Sequential()
    model.add(createKerasEmbeddingLayer(w2v_model, word2index, trainable))
    model.add(LSTM(128, 
                   dropout=0.3, 
                   recurrent_dropout=0.3, 
                   return_sequences=False, 
                   kernel_regularizer = regularizers.l2(1e-7),
                   bias_regularizer = regularizers.l2(1e-7),
                   activity_regularizer = regularizers.l2(1e-7)))
    model.add(Dense(1))
    
    rms = Adam(decay=lr_decay, lr=learning_rate)
    model.compile(loss=rmsle, optimizer=rms, metrics=['accuracy'])
    
    return model


def crateTrainEvaluateLSTMModel(y_train, y_test, X_train, X_test, savedModelName, noOfEpochs, w2v_model, word2index, trainable, learning_rate, lr_decay):
    
    model = createSimpleLSTMWithEmbeddingModel(w2v_model, word2index, trainable, learning_rate, lr_decay)

    # Train model
    print('Train...')
    history = model.fit(X_train, 
                        y_train, 
                        batch_size=32, 
                        epochs=noOfEpochs, 
                        validation_data=(X_test, y_test),
                        callbacks=[createEarlyStopping()])


    #Plot and save model
    plotAndSaveModel(savedModelName, model, history)

'''
Builds simple CNN model using Conv1D layers
'''

def createEarlyStopping():
    return callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, mode='auto')


def createCNNModel(w2v_model, word2index, trainable, learning_rate, lr_decay):
    model = Sequential()
    
    model.add(createKerasEmbeddingLayer(w2v_model, word2index, trainable))
    #workaround for known bug in Keras https://github.com/keras-team/keras/issues/4978
    model.add(Lambda(lambda x: x, output_shape=lambda s:s))
    model.add(Dropout(0.4))
    
    model.add(Conv1D(NUM_FILTERS[0], 
                     FILTER_SIZES[0], 
                     padding='valid', 
                     activation='relu', 
                     strides=1,
                     kernel_regularizer = regularizers.l2(1e-6),
                     bias_regularizer = regularizers.l2(1e-6),
                     activity_regularizer = regularizers.l2(1e-6)))
    model.add(MaxPooling1D(2,strides=1, padding='valid'))

    model.add(Conv1D(NUM_FILTERS[1], 
                     FILTER_SIZES[1], 
                     padding='valid', 
                     activation='relu', 
                     strides=1,
                     kernel_regularizer = regularizers.l2(1e-6),
                     bias_regularizer = regularizers.l2(1e-6),
                     activity_regularizer = regularizers.l2(1e-6)))
    model.add(MaxPooling1D(4,strides=1, padding='valid'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dropout(0.4))
    
    model.add(Dense(units=1))
    
    # this creates a model
    rms = RMSprop(lr=learning_rate, decay=lr_decay)
    model.compile(loss=rmsle, optimizer=rms, metrics=['accuracy'])
    
    return model


'''
Evaluates given model based on validation data set passed
Plots the accuracy graph
'''
def plotAndSaveModel(savedModelName, model, history):
    model.summary()    

    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.title('Performance of model LSTM')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs No')
    plt.savefig(savedModelName + '_initialModel_plot.png')
    serializeModel(model, savedModelName + "_initialModel")


def crateTrainEvaluateCNNModel(Y_train, Y_test, X_train_vectorized, X_test_vectorized, savedModelName, noOfEpochs, w2v_model, word2index, trainable, learning_rate, lr_decay):
    
    model = createCNNModel(w2v_model, word2index, trainable, learning_rate, lr_decay)
    
    # Train model
    print('Train...')
    history = model.fit(X_train_vectorized, 
                        Y_train, 
                        batch_size=32, 
                        epochs=noOfEpochs, 
                        validation_data=(X_test_vectorized, Y_test),
                        callbacks=[createEarlyStopping()])
    
    # Plot and save model
    plotAndSaveModel(savedModelName, model, history)

    
#TODO: refactor needed    
def vectorizeInput(X_train, w2v_model, empty_word, missedWords, networkModel, word2index):
    #Keras Embedding layer requires id of a word2vec not the embedding 
    X_train_vectorized = np.zeros(shape=(len(X_train), MAX_WORDS_NO*2), dtype=int)
    
    word2indexId = 1
    
    for idx, document in enumerate(X_train):
        for jdx, word in enumerate(document):
            if word in w2v_model:
                if word not in word2index:
                    word2index[word] = word2indexId
                    word2indexId += 1
                X_train_vectorized[idx, jdx] = word2index[word] 
            else:
                X_train_vectorized[idx, jdx] = 0
                missedWords.append(word)
                    
    return X_train_vectorized


'''
Based on imported word2vec embeddings vectorizes the input
'''
def loadWord2VecAndVectorizeInputs(X_train, X_test, word2vecURI):
    
    #load Word2Vec model
    w2v_model = Word2VecKeyedVectors.load_word2vec_format(word2vecURI, binary=False)
    print("vocab_size = %s", len(w2v_model.vocab))
    
    #determine number of features for each word in the model
    WORD2VEC_NO_OF_FEATURES = w2v_model['dog'].shape[0]

    print("num_features = ", WORD2VEC_NO_OF_FEATURES)
    print("len(X_train) = ", len(X_train))
    print("len(X_test) = ", len(X_test))
    
    #define the missing word vector
    empty_word = np.zeros(WORD2VEC_NO_OF_FEATURES, dtype=float)
    
    #create the list to get the all words which we are missing in the Word2Vec model
    missedWords = []
    word2index = {}
    
    #vectorize each input
    X_train_vectorized = vectorizeInput(X_train, w2v_model, empty_word, missedWords, word2index)
    X_test_vectorized = vectorizeInput(X_test, w2v_model, empty_word, missedWords, word2index)

    print("Number of used words = ", len(set(word2index)))
    print("Number of words missing = ", len(set(missedWords)))
    
    return X_train_vectorized, X_test_vectorized, w2v_model, word2index


def load_data(filename):
    data = pd.read_csv(filename, sep="\t", index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    # Select the columns (feel free to select more)
    X = data.loc[:, ['question_text', 'answer_text']]
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return X, None
    return X, y


def rmsle(y, y0):
    return K.sqrt(K.mean(K.square(tf.log1p(y) - tf.log1p(y0))))


def main(args):
    #import pydevd;pydevd.settrace();
    pprint(args)

    X, y = load_data(args.trainDataSetURI)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #plotXHist(X_train['question_text'], X_test['answer_text'])     # use this to define MAX_WORDS_NO which will account for most of the data
                
    X_train, X_test, w2v_model, word2index = tokenizeLemmatizeDataSet(X_train, X_test, args.word2vecmodel)

    #vectorize the sentences using Word2Vec from fastText pointed by args.word2vecmodel
    #X_train_vectorized, X_test_vectorized, w2v_model, word2index = loadWord2VecAndVectorizeInputs(X_train, X_test, args.word2vecmodel)
    
    if args.networkModel=='CNN':
        crateTrainEvaluateCNNModel(y_train, y_test, X_train, X_test, args.savedModelName, args.no_of_epochs, w2v_model, word2index, args.trainable, args.learning_rate, args.lr_decay)
    else:
        crateTrainEvaluateLSTMModel(y_train, y_test, X_train, X_test, args.savedModelName, args.no_of_epochs, w2v_model, word2index, args.trainable, args.learning_rate, args.lr_decay)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trainable', 
        help='Whether Keras Embedding layer should be trained',
        type=str2bool, 
        nargs='?',
        const=True, 
        default=False)
        
    parser.add_argument('--networkModel', type=str,  choices=['LSTM', 'CNN'],
        help='LSTM - simple LSTM model using 3d input, CNN - simple CNN'
        , default='LSTM')

    parser.add_argument('--trainDataSetURI', type=str,
        help='URI pointing to train data set'
        , default='C:\\Users\\michal.gdak\\Desktop\\py\\kaggledays\\train\\train.csv')
    
    parser.add_argument('--testDataSetURI', type=str,
        help='URI pointing to test data set'
        , default='C:\\Users\\michal.gdak\\Desktop\\py\\kaggledays\\test\\test.csv')
    
    parser.add_argument('--word2vecmodel', type=str,
        help='URI pointing to word 2 vec model to be used'
        , default='C:\\Users\\michal.gdak\\Desktop\\py\\imdb\\crawl-300d-2M.vec')
    
    parser.add_argument('--savedModelName', type=str,
        help='File name for the train model persistance'
        , default='LSTM')
    
    parser.add_argument('--no_of_epochs', type=int,
        help='Number of epochs to run.', default=150)
    
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.003)
    
    parser.add_argument('--lr_decay', type=float,
        help='Learning rate decay', default=1e-4)
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))