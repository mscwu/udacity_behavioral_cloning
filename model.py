import csv
import cv2
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

data = pd.read_csv('./data/driving_log.csv')

train_samples, validation_samples = train_test_split(data, test_size=0.2)

images = []
steer = []

def generator(samples, batch_size = 8):
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
				correction = 0.21 # this is a parameter to tune
				steer_c = batch_sample['Steer']
				steer_l = steer_c + correction
				steer_r = steer_c - correction
				# flip the images and steer angle left and right to augment the data
				# flip images only for steer angles more than +/- 0.75 degrees
				if np.absolute(steer_c) > 0.75:
					img_c_flipped = cv2.flip(img_c, 1)
					steer_c_flipped = -steer_c
					img_l_flipped = cv2.flip(img_l, 1)
					steer_l_flipped = -steer_c + correction
					img_r_flipped = cv2.flip(img_r, 1)
					steer_r_flipped = -steer_c - correction			
					images.extend([img_c, img_l, img_r, img_c_flipped, img_l_flipped, img_r_flipped])
					steer.extend([steer_c, steer_l, steer_r, steer_c_flipped, steer_l_flipped, steer_r_flipped])
				else:											
					images.extend([img_c, img_l, img_r])
					steer.extend([steer_c, steer_l, steer_r])
			X_train = np.array(images)
			y_train = np.array(steer)
			yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)


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
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4, verbose=1)
model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
exit()