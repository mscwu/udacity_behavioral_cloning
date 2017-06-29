import csv
import cv2
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
# lines = []
# with open('../data/driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for line in reader:
# 		lines.append(line)

# images = []
# measurements = []
# for line in lines:
data = pd.read_csv('./data/driving_log.csv')
# print(data['IMR'][2])
train_samples, validation_samples = train_test_split(data, test_size=0.2)
# print (train_samples['IMC'].iloc[0:2])
# print (train_samples['IML'].iloc[0:2])
# print (train_samples['IMR'].iloc[0:2])
# print (train_samples['Steer'].iloc[0:2])
# print(len(train_samples['Steer'].iloc[0:2]))
# print(train_samples.shape)
# print (train_samples.iloc[0:2]['IMC'])
images = []
steer = []

def generator(samples, batch_size = 32):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples.iloc[offset:offset+batch_size]
			images = []
			steer = []
			for index, batch_sample in batch_samples.iterrows():
				name_c = batch_sample['IMC']
				name_l = batch_sample['IML']
				name_r = batch_sample['IMR']
				img_c = cv2.imread(name_c)
				img_l = cv2.imread(name_l)
				img_r = cv2.imread(name_r)
				correction = 0.2 # this is a parameter to tune
				steer_c = batch_sample['Steer']
				steer_l = steer_c + correction
				steer_r = steer_c - correction
				images.extend([img_c, img_l, img_r])
				steer.extend([steer_c, steer_l, steer_r])

			X_train = np.array(images)
			y_train = np.array(steer)
			yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
# for row in arange(data.shape[0]):
# 	steering_center = float(data['Steer'][row])

# 	# create adjusted steering measurements for the side camera images
# 	correction = 0.2 # this is a parameter to tune
#     steering_left = steering_center + correction
#     steering_right = steering_center - correction

#     img_center = 
#     img_left = 
#     img_right = 
ch, row, col = 160, 320, 3  # Trimmed image format
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
exit()