# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

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

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 180-192)togather with RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 176). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 195-201). 

The model was trained and validated on different data sets over multiple iterations to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 207).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and sharp turns data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different models and see how the car performed.

My first step was to use a convolution neural network model with lenet architecture this was a starting point for me as i was familiar with lenet in the previous projects. This gave me a starting point. The model was training well and both training and validation loss where decreasing. The model when tested made my car to move quite a bit in the straight regions of the road and it went out of track in the sharp turn area. Then i thought that the data supplied was not enough for the model to train properly so i generated more data from the existing data by image augmentation. I used random horizantal shifting and flipping of imgaes and steering data as a means to generate more data from the existing dataset. After doing this i re-run the model which gave be little better performance but that was not enough for completing a full round. 

Then as hinted in the lesson i went back and read the nvidia paper and saw the network they used. I re-created the exact network and trined a model on that data which gave me better results. Still at some regions the car used to go out of track specially near the dirt track steep turn. 

Next i introduced relu activation function to introduce non-linearity in the model, which gave better performance. Further i introduced few dropout layers to avoid over fitting. Further i experimetned with the convolutional layers to see which gave me the best result and finalized on the model provided with this writeup.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I have access to tesla p-100 machine with 32 gb ram and therefore i did'nt have to use generators. Infact i found that using generators will make my model train in the cpu only which was infact bad for me as the training is very slow compared to gpu training.

#### 2. Final Model Architecture

The final model architecture (model.py lines 169-203) consisted of the following layers.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
Conv1 (Convolution2D)            (None, 40, 160, 32)   896         cropping2d_1[0][0]               
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 20, 80, 32)    0           Conv1[0][0]                      
____________________________________________________________________________________________________
Conv2 (Convolution2D)            (None, 10, 40, 64)    18496       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 5, 20, 64)     0           Conv2[0][0]                      
____________________________________________________________________________________________________
Conv3 (Convolution2D)            (None, 5, 20, 128)    73856       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 3, 10, 128)    0           Conv3[0][0]                      
____________________________________________________________________________________________________
Conv4 (Convolution2D)            (None, 3, 10, 128)    65664       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3840)          0           Conv4[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 3840)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
FC1 (Dense)                      (None, 128)           491648      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128)           0           FC1[0][0]                        
____________________________________________________________________________________________________
FC2 (Dense)                      (None, 128)           16512       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128)           0           FC2[0][0]                        
____________________________________________________________________________________________________
FC3 (Dense)                      (None, 64)            8256        dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             65          FC3[0][0]                        
====================================================================================================
```

#### 3. Creation of the Training Set & Training Process

I used front, left and side camera images togather with augmented data to get the final training and validation data.
I defined methods which will select- front, left or right images also methods which will augment the data. The function `translate_image()` randomly translates the image horizantally making corresponding adjustments in the steering data also. Next function `brightness_img` randomly adjusts the brightness of the image, `flip_img` filps my image and steering data. The fist two function will help my model better generalize and the third function will help me better balance the data sets. Now my function `generate_train_data` will randomly gets augmented or unaugmened image and add it into the dataset. `data_size` parameter defines how many of these images are added. Thus these images will introduce enough variablity and balance in the dataset so that my model will generalize better.

For validataion i only used images from the front camera as those are the ones which my simulator is going to operate on. On trainig data i used both front as well as side image. When using side images i add a correction factor of .22 to them depending on the direction.



