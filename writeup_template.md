# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_img/cnn-architecture.png "Model Architecture"
[image2]: ./write_up_img/center.jpg "Centerline Driving"
[image3]: ./write_up_img/iml.jpg "Left Camera"
[image4]: ./write_up_img/imr.jpg "Right Camera"
[image5]: ./write_up_img/figure_1.png "Loss"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In my project, I used the model propose by NVIDIA. More details can be found [here](https://arxiv.org/abs/1604.07316)  

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).  
Total EPOCH number is selected such that overfitting is eliminated.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and multiple cameras correction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA model. I thought this model might be appropriate because it was a proven model used by NVIDIA and the implementation was very simple. The model size was also small which took about 1 min (15 secs each EPOCH) to train on a GTX 1060 6GB GPU.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Specifically, 20% of the data was used as validation set.  

The default model trained very well and no significant overfitting is present.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I believe the reason was that during one lap recording, most of the time the steering wheel stayed close to zero. This biased the model towards a zero steering wheel angle because this tended to minimize the MSE. To improve the driving behavior in these cases, I collected more data at corner entry and exit. This turned to be very effective.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:  

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Normalization     |                                     |
| Cropping          | 85x320x3 RGB image                  |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 41x158x24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 19x77x36      									|
| RELU					|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 8x37x48      									|
| RELU					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 6x35x64      									|
| RELU					|	
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x33x64      									|
| RELU					|	
| Flatten	      	| outputs 8448 				|
| Fully connected		| outputs 1164      									|
| Fully connected		| outputs 100      									|
| Fully connected		| outputs 50       									|
| Fully connected		| outputs 10       									|
| Fully connected		| outputs 1        									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded one lap of vehicle driving in clockwise direction.

I then collected driving data at curve entry and exit only to "teach" the car how to curve.

I did not use recovery lap as this turned out to be a very uneffective process. It depened largely on when to start/end the recording and how the car was recovered. When done incorrectly, it had negative effect on normal driving. This led me to abandon the idea of recovery lap and focus more on multiple camera usage in order to keep the car on the center of the lane.

To take advantage of the 3 cameras mounted on the car, I used a correction angle to compensate for the steer angle (code line 35 - 38). Basically, when the car went towards one side of the road, the center image would look like either left or right image taken from a car driving on the center. The final angle I used was 0.21 degrees.  

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would generalize the data (code line 41 - 52). More specifically, this techique was only applied to steer angles more than a certain threshold. Again, this method was used to balance the training dataset such that not all the data were concentrated around zero.  

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the convergence of training and validataion loss.  

![alt text][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
