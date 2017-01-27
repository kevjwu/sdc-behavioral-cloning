# End to End Learning with Convolutional Neural Nets in Keras

*Submitted as part of Term 1 of Udacity’s Self-Driving Car Nanodegree program.*

The model and pre-processing steps that I implemented for this problem are contained in my solution file, _model.py_. The solution to this project is derived from the 2016 NVIDIA paper, “End to End Learning for Self Driving Cars” (Link: https://arxiv.org/pdf/1604.07316.pdf). 

## Model

### Pre-Processing

Training images were resized to 66x200. Color images were used and RGB color channels converted to YUV color space before normalization (a common pre-processing step in computer vision that’s also used in the NVIDIA paper). 

I used Keras’s `BatchNormalization` method to normalize each color channel to a mean of 0 and std dev of 1 within each batch of training data (rather than over the entire dataset).  

### Network Architecture

After the normalization step, the overall architecture of the convolutional neural network used is as follows:

* Layer 1: Convolutional layer with 5x5 kernel, 2x2 stride, and depth of 24. 
* Layer 2: Convolutional layer with 5x5 kernel, 2x2 stride, and depth of 36.
* Layer 3: Convolutional layer with 5x5 kernel, 2x2 stride, and depth of 48. 
* Layer 4: Convolutional layer with 3x3 kernel, 1x1 stride, and depth of 64.
* Layer 5: Convolutional layer with 3x3 kernel, 1x1 stride, and depth of 64. 
* Layer 6: Fully connected layer of size 100.
* Layer 7: Fully connected layer of size 50.
* Layer 8: Fully connected layer of size 10.
* Output layer

### Activation functions

The NVIDIA paper did not specify the activation functions used in each layer. I used rectified linear units (RELU) following each of the convolutional and fully-connected layers.

I used tanh activation for the output layer because the target variable lay in the range [-1, 1].

### Dropout

Again, the NVIDIA paper did not mention the use of dropout layers to reduce overfitting. I found that introducing two dropout layers with dropout rate 50% following Layers 6 and 7 significantly improved the performance of the car on Track 1. 

| Dropout layers    | Validation Loss    | Testing Loss | Track 1 Success | 
| ------------------| -------------------| -------------| ----------------|
| None	 	    | 0.0147   		 | 0.0440    	| No 		  |
| 6	 	    | 0.0125   		 | 0.0376    	| No 		  |
| 7	 	    | 0.0136   		 | 0.0410    	| Yes 		  |
| 6 and 7	    | 0.0123   		 | 0.0458    	| Yes 		  |


While the model architecture with a dropout after layer 7 also made it all the way around the track, its had a higher validation loss and also tendency to make much wider turns around the track (a characteristic of the model not captured by our standard metrics). 

## Training

### Data generation

I generated ~50k unique images by driving the car around the simulated track. About a quarter of these images capture the car in “recovery mode”; I would drive the car to the edge of the lane or even a little past it, and start recording its path back to the center of the lane. This was essential for training the model to be able to self-correct when it veered off course. 

With the overall neural net architecture already determined, in some ways, generating training data was the most important task. Some takeaways:
 
* _Quantity mattered, but only up to a certain extent:_ overall, I had a hard time getting the car all the way around the track training on <20k data points, but after 40k, it didn’t seem to help much. 
* _Quality was more important:_ Garbage in, garbage out. To properly train the car I had to drive it well in training, although with sufficient data, the occasional crash/blowout was tolerable. The tricky part was training for recovery. Initially I only trained the car to recover from outside the lane lines; later, when I was wondering why my car kept on deviating outside the lane in testing, I realized that it was also necessary to recording images of the car to moving back to the center of the lane from within the edge of the lane. I had to do my best to avoid recording images from when I was steering the car off-course (intentional or not). 

### Fake Data

There were two ways I tried to augment the original dataset: 

1. **Oversampling turns:** I did this after some initial attempts in my which my model was clearly overestimating the frequency of steering angle 0 during testing.
	* I oversampled images in the training set where the steering angle was greater than ~15 degrees in either direction. . 
	* I also manually generated extra training data by driving laps around the track in which I would only record during curves. 

Looking back, I would recommend picking one approach or the other; since I was tweaking the model while also adding more training data, I found that the optimal oversampling ratio as well as the optimal steering angle threshold to use was constantly changing, adding an unnecessary level of complexity to the problem. 

This technique also turned out to be a bit of a double-edged sword; too much oversampling and the car would exhibit a strong tendency to weave from side to side in testing. Balancing the car’s ability to turn and its similarity to normal human driving behavior was a challenge. 

2. **Horizontal flips:** To generate novel images, I flipped each image in my dataset about the y-axis and negated the sign on the steering angle. 

| Training Data     | Validation Loss    | Testing Loss | Track 1 Success | 
| ------------------| -------------------| -------------| ----------------|
| No flips 	    | 0.0123   		 | 0.0458    	| Yes 		  |
| Horizontal flips  | 0.0131 		 | 0.0499    	| No 		  |

This actually did not improve my model’s performance on either the validation/testing set, or on its ability to navigate Track 1. My theory is that producing mirror images did not add any new informational content to our dataset, and actually made overfitting worse by effectively training on two copies of each image. 

### Objective function

I used an Adam optimizer for stochastic gradient descent with learning rate 1e-4. Since we are predicting a continuous y variable (steering angle) I used mean squared error as the loss function. 

### Training procedure

After shuffling the data, I split off 10% of the original dataset (before augmentation) to be used for validation. I used a batch size of 64 training and trained for only 5 epochs, as I found that validation loss and validation mean absolute error rarely decreased after 5 epochs. 

Testing data was generated by recording laps around Track 2. 

### Hardware

Using a generator function allowed me to bypass loading the entire dataset into memory, making it theoretically possible to train the model on my MacBook Pro with its quad-core 2.2ghz Intel Core i7 CPU. However, it basically made my Mac unusable during training, as it was using up 5 GB of RAM out of an available 16 and it’s CPU usage was >400%. 

I switched to using an AWS g2.2xlarge instance, which besides freeing up my local machine, sped up training times by ~10x.  

## Closing thoughts

The CNN architecture described in the NVIDIA paper is remarkably simple compared to some of the existing computer vision models (VGG, GoogLeNet, AlexNet). Unfortunately, while I was pretty easily able to implement a model that would take the car around Track 1, getting it to generalize to Track 2 was very difficult, even after implementing standard overfitting precautions such as dropout; almost all of my models crashed within the first two turns of the unseen track. 

Given that we only trained on Track 1 data, this isn’t all that surprising; besides incorporating other driving environments in the training data, here are some potential ideas on how to create a more generalizable model:

* Pre-processing/feature extraction: Restricting the field of vision and using techniques like the Hough transform to identify lane lines would help better isolate relevant areas of the image. 
* Smoothing: Averaging the steering angle for the last n observations in the training set in order to smooth the path of the car; this would help compensate for some degree of human error during training data generation (sudden changes in direction as a result of under/overshooting steering angle). 
* Custom objective function: MSE didn’t seem like the ideal choice of metric/loss function in this case. Over the course of trying different architectures/parameters, I found that low MSE didn’t necessarily correspond to a “good” driving path; in fact, it was the often the case that models with higher validation loss actually made it further around the track before crashing. 

