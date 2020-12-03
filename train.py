# bert
import sys
sys.path.append('modules')
from keras_bert import load_trained_model_from_checkpoint

import os
pretrained_path = os.path.expanduser('~/model/bert-wiki-ja/')
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'model.ckpt-1400000')
sentencepiece_path = os.path.join(pretrained_path, 'wiki-ja.model')

dataset_path = os.path.expanduser('~/data/livedoor-text/')
#dataset_path = os.path.expanduser('~/data/small/')

model_path = os.path.join(pretrained_path, 'model_livedoor.h5')

EPOCHS=100
BATCH_SIZE=16

bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False, trainable=False)
bert.summary()

# sentencepiece
import sentencepiece as spm
from keras import utils
import logging
import numpy as np

maxlen = 512
bert_dim = 768

sp = spm.SentencePieceProcessor()
sp.Load(sentencepiece_path)

# SentencePieceで学習済みモデルのBertから特徴ベクトルを抽出。
def _get_vector(feature):
    common_seg_input = np.zeros((1, maxlen), dtype = np.float32)
    indices = np.zeros((1, maxlen), dtype = np.float32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        try:
            indices[0, t] = sp.piece_to_id(token)
        except:
            logging.warn(f'{token} is unknown.')
            indices[0, t] = sp.piece_to_id('<unk>')
    vector =  bert.predict([indices, common_seg_input])[0]
    hoge = bert.predict([indices, common_seg_input]) #check

    return vector

# read data
import glob, re
def _load_data_livedoor(dir, max_len):
    text = []
    label2id = {}
    count = 0
    labels = []
    label_names = []
    dir += "**/" # read sub directories on livedoor
    for label in glob.glob(dir):
        # get labelname
        match = re.match(r'.+\/(.*)\/', label)
        if match:
            labelname = match.group(1)
            label2id[labelname] = count
            label_names.append(labelname)
            count += 1
        else:
            print("error: dataset directory was not found")
            exit()
        
        # read files
        for file in glob.glob(label + "/*.txt"):
            with open(file, 'r', encoding='utf-8') as fh:
                sentences = fh.readlines()
                tmp = '\n'.join(sentences[2:]) #冒頭2行（URL, タイムスタンプ）を除外。
                if len(tmp) > (max_len-2): # [CLS], [SEP]
                    tmp = tmp[:max_len-2]
                text.append(tmp)
                labels.append(label2id[labelname])
    labels = utils.to_categorical(labels, len(label2id))
    return text, labels, label2id, label_names

def _text_to_vectors(text):
    print("_text_to_vectors(): begin.")
    vectors = []
    count = 0
    for sentences in text:
        vectors.append(_get_vector(sentences))
        count += 1
        if count % 1000 == 0:
            print("_text_to_vectors(): " + str(count) + " done...")
    print("_text_to_vectors(): " + str(count) + " done.")
    return np.array(vectors)


import time
start = time.time()
text, labels, label2id, label_names = _load_data_livedoor(dataset_path, maxlen)
print(label2id)
X = _text_to_vectors(text)
print("_text_to_vectors(): time = ", round(time.time() - start, 2), "sec")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
print("len(x_train), len(y_train): ", len(x_train), len(y_train))


# ready model for classification
from keras.layers import Dense, LSTM, Bidirectional
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
input_layer = Input(x_train[0].shape)
x1 = Bidirectional(LSTM(356))(input_layer)
output_layer = Dense(len(label2id), activation='softmax')(x1)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['mae', 'mse', 'acc'])
model.summary()

# fine-tune for classification
start = time.time()
print("model.fit(): begin")
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks = [
              EarlyStopping(patience=5, monitor='val_acc', mode='max'),
              ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
          ],
          validation_split=0.2, shuffle=True
    )
print("model.fit(): time = ", round(time.time() - start, 2), "sec")
#model.save(model_path)

# visualize history
import matplotlib.pyplot as plt
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('logs/training-validation-accuracy.png')
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('logs/training-validation-loss.png')

# predict
start = time.time()
print("model.predict(): begin")
predicts = model.predict(x_test, verbose=True).argmax(axis=1)
print("model.predict(): time = ", round(time.time() - start, 2), "sec")

# accuracy
test_labels = y_test.argmax(axis=1)
print("ACC(test): ", np.sum(test_labels == predicts) / len(y_test))

# report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_labels, predicts, target_names=label_names))
