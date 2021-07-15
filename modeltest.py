import h5py
import numpy as np
from sklearn import preprocessing

# f = h5py.File('path/filename.h5','r') #打开h5文件
f = h5py.File('C:/Users/kaka5/OneDrive/Desktop/spr_model.h5','r')
f.keys() #可以查看所有的主键
print([key for key in f.keys()])

import tensorflow as tf

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('C:/Users/kaka5/OneDrive/Desktop/spr_model.h5')

# Show the model architecture
new_model.summary()


import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def is_valid(file_path):
    ''' returns True if a regular files. False for hidden files.
    Also, True is a known user with a name, False if anon.
    '''
    file_name = tf.strings.split(file_path, '/')[-1]
    if tf.strings.substr(file_name, 0, 1) == tf.constant(b'.'):
        return False
    sc = tf.strings.split(file_path, '/')[-3]
    speaker = tf.strings.split(sc, '-')[0]
    return not tf.strings.substr(speaker, 0, 9) == tf.constant(b'anonymous')

list_ds = tf.data.Dataset.list_files(str("C:/Users/kaka5/OneDrive/Desktop/voxforge/"'*/wav/*.wav'))
list_ds = list_ds.filter(is_valid)
for f in list_ds.take(3):
  print(f.numpy())
  
def extract_speaker(file_path):
    ''' extract speaker name from the file path '''
    sc = tf.strings.split(file_path, '/')[-3]
    return tf.strings.split(sc, '-')[0]

def wav2mfcc(file_path, max_pad_len=196):
    ''' convert wav file to mfcc matrix with truncation and padding '''
    wave, sample_rate = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sample_rate)
    mfcc = mfcc[:, :max_pad_len]
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def extract_mfcc(file_path):
    ''' returns 3D tensor of the mfcc coding from the wav file '''
    file_name = bytes.decode(file_path.numpy())
    mfcc = tf.convert_to_tensor(wav2mfcc(file_name))
    mfcc = tf.expand_dims(mfcc, 2)
    return mfcc

def create_audio_ds(list_ds):
    batch = []
    for f in list_ds:
        audio = extract_mfcc(f)
        batch.append(audio)
    return tf.data.Dataset.from_tensor_slices(batch)
speaker_ds = list_ds.map(extract_speaker)
for speaker in speaker_ds.take(50):
    print(speaker)
    
speaker_encoder = preprocessing.LabelEncoder()
speaker_idx = speaker_encoder.fit_transform([bytes.decode(s.numpy()) for s in speaker_ds])
encoded_speaker_ds = tf.data.Dataset.from_tensor_slices(speaker_idx)
unique_speakers = len(speaker_encoder.classes_)
for es in encoded_speaker_ds.take(50):
    print(es)
 
sample_file = [os.path.join('C:/Users/kaka5/OneDrive/Desktop/voxforge/Aaron-20080318-kdl/wav/b0019.wav'),
               os.path.join('C:/Users/kaka5/OneDrive/Desktop/voxforge/bugsysservant-20091103-cob/wav'),
               os.path.join('C:/Users/kaka5/OneDrive/Desktop/voxforge/Campbell-20091230-set/wav')]
'''
               os.path.join('/content/drive/voxforge/DavidL-20091116-kth/wav/b0056.wav'),
               os.path.join('/content/drive/voxforge/ESimpray-20150125-svl/wav/b0025.wav'),
               os.path.join('/content/drive/voxforge/Fandark-20100822-acy/wav/b0003.wav'),
               os.path.join('/content/drive/voxforge/GamaBedolla-20150210-jbr/wav/b0404.wav'),
               os.path.join('/content/drive/voxforge/Hadlington-20130720-pwc/wav/a0210.wav'),
               os.path.join('/content/drive/voxforge/J0hnny_b14z3-20111219-ibu/wav/b0051.wav'),
               os.path.join('/content/drive/voxforge/Kai-20111021-apo/wav/b0049.wav'),
               os.path.join('/content/drive/voxforge/L1ttl3J1m-20090701-fhz/wav/a0185.wav'),
               os.path.join('/content/drive/voxforge/MARTIN0AMY-20111106-pwg/wav/a0491.wav'),
               os.path.join('/content/drive/voxforge/Nadim-20100515-efk/wav/b0276.wav'),
               os.path.join('/content/drive/voxforge/Otuyelu-20101107-crp/wav/b0209.wav'),
               os.path.join('/content/drive/voxforge/Paddy-20100120-msy/wav/b0092.wav')]
'''

sample_ds = tf.data.Dataset.from_tensor_slices(sample_file)
sample_input = create_audio_ds(sample_ds).batch(2)
output = new_model.predict(sample_input)

speaker_ids = output.argmax(axis=1)
speakers = speaker_encoder.inverse_transform(speaker_ids)
print(speakers)
print(output)