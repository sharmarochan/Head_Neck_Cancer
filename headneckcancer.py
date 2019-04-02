# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:14:35 2019

@author: Rochan.Sharma
"""


import json
import os
script_dir = r'C:\Users\rochan.sharma\Desktop\asng\data\docs'

corpus = []
tags = []

for root, dirs, files in os.walk(script_dir):  
    for filename in files:
        file = os.path.join(root, filename)
        print(file)
        f = open(file)
        data = json.load(f)
        f.close()
        job_description = str(data["jd_information"]['description'])
        print(job_description)
        corpus.append(job_description)
        job_keyword = data["api_data"]["job_keywords"][0]
        job_keyword = job_keyword.replace(' ','_')
        tags.append(job_keyword)
        
#        break



import sklearn.preprocessing as preprocessing
import sklearn.feature_extraction.text as text



encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(tags)


# all the tags that will be given labels
encoder.classes_


tfidf = text.TfidfVectorizer(corpus, stop_words="english", max_features=5000)

tfidf_matrix = tfidf.fit_transform(corpus)

tfidf_matrix.shape


import sklearn.model_selection as model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(tfidf_matrix,y,test_size=0.20, random_state=42 )


from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.layers import Dense, Activation

from keras.models import Sequential

from keras.utils import to_categorical


y_train.shape
x_train.shape

input_dim = x_train.shape[1]

unique_val = len(list(dict.fromkeys(tags)))

y_train=to_categorical(y_train,unique_val)
y_train.shape
print(type(x_train))    
#x_train=np.array(x_train)

import numpy as np
x_test = np.array(x_test)
model = Sequential()

model.add(Dense(10, input_shape=(5000,), activation='relu'))

model.add(Dense(100, activation='relu'))
model.add(Dense(unique_val, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()




history = model.fit(x_train, y_train, epochs=100, verbose=True,batch_size=10)


# convert the x_train in np array





