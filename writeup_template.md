#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network 
- Data Exploration Jupyter Notebook - data_exploration.ipynb
- writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and Udacity provided data and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model used is summarized as follows:

- 4 Convolutional layers. The filter size is 3x3.  
- Used relu as a activation function, glorot_uniform for weight initialization and used maxpooling and 3 fully connected layers. 
- Added a droupout and it worked well.
- Used an Adam optimizer for training and learning rate = 0.0001, batch size : 32, 3 epochs.
- Also separated 20% of validation dataset after shuffling data. 


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on the Udacity data set to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. It was also tested successfully on the second track provided by Udacity.

####3. Model parameter tuning

The model used an Adam optimizer, with a learning rate of 0.0001

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, image flipping and cropping and shadow addition.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA and comma.a1 papers. I thought this model might be appropriate because they are a direct map to the current requirements. To these, I added multiple fully connected layers with dropout to prevent overfitting

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set using a 80/20 split.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. A majority of the success is attributed to the high level of data pre-processing and augumentation.

####2. Final Model Architecture

The final model architecture is:


- Conv1 (Convolution2D) 			(None, 64,64,3)	
- maxpooling2d_1 (MaxPooling2D) 	(None, 16, 16, 32) 	
- Conv2 (Convolution2D) 			(None, 8, 8, 64) 
- maxpooling2d_2 (MaxPooling2D) 	(None, 4, 4, 64) 	
- Conv3 (Convolution2D) 			(None, 4, 4, 128) 	
- maxpooling2d_3 (MaxPooling2D) 	(None, 2, 2, 128) 
- Conv4 (Convolution2D) 			(None, 2, 2, 128) 	
- flatten_1 (Flatten) 				(None, 512) 
- dropout_1 (Dropout) 				(None, 512) 
- FC1 (Dense) 						(None, 128) 
- dropout_2 (Dropout) 				(None, 128) 
- FC2 (Dense) 						(None, 128) 
- dropout_3 (Dropout) 				(None, 128) 	
- FC3 (Dense) 						(None, 64) 	
- dense_1 (Dense) 					(None, 1) 	



####3. Creation of the Training Set & Training Process

Initial Data Processing of the Udacity data is done in the Jupyter notebook (data_exploration.ipynb) showed that the steering data was imbalanced (refer to notebook for figures). Hence multiple data augmentation techniques were used as follows:

- Use left & right camera images to simulate recovery
	- Using left and right camera images to simulate the effect of car wandering off to the side, and recovering. Add a small angle .30 to the left camera and subtract a small angle of 0.30 from the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left.
- Center Image bias: Simce a majority of the images are with a steering angle of zero, some of these are dropped to reduce the propensity of the car to drive straight
- Flip the images horizontally
	- Since the dataset has a lot more images with the car turning left than right(because there are more left turns in the track), flip the image horizontally to simulate turing right and also reverse the corresponding steering angle.
- Brightness Adjustment
	- Adjust the brightness of the image to simulate driving in different lighting conditions by converting to HSV, randomly adding shadows and converting back to RGB
- Add Random Shadows
	- Although not demonstrated in the notebook, an illustration of this method is in the above mentioned blog post.
- Crop Images:
	- To reduced the training time, all images were cropped to remove the hood of the car and the horizon. Subsequently, these were resized to (64x64)

With these augmentation techniques, practically infinite unique images for training the neural network can be generated.

This methodology was used with the keras model-fit generator during the training phase, to generate 20000 images with 3000 validation images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced. 
