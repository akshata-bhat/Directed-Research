# Exploring-MNIST

* REFERENCE AND SOURCE CODE: The Ultimate Beginner Guide to Deep Learning in Python: https://elitedatascience.com/keras-tutorial-deep-learning-in-python 

* Task: Design the following models with the mentioned specifications and record the observations
* Framework : Keras
* Backend : Tensorflow
* Dataset : MNIST


# Model 1 a) 
* Input: 28x28 sized images
* Output: 0 – 9 labels
* Input Layer → Fully Connected Layer→ Output Layer

* Result:

60000/60000 [==============================] - 8s 138us/step - loss: 0.3942 - acc: 0.8840
Epoch 2/10
60000/60000 [==============================] - 8s 136us/step - loss: 0.2230 - acc: 0.9344
Epoch 3/10
60000/60000 [==============================] - 8s 136us/step - loss: 0.1846 - acc: 0.9458
Epoch 4/10
60000/60000 [==============================] - 8s 139us/step - loss: 0.1647 - acc: 0.9508
Epoch 5/10
60000/60000 [==============================] - 8s 138us/step - loss: 0.1477 - acc: 0.9555
Epoch 6/10
60000/60000 [==============================] - 8s 137us/step - loss: 0.1407 - acc: 0.9575
Epoch 7/10
60000/60000 [==============================] - 8s 136us/step - loss: 0.1283 - acc: 0.9601
Epoch 8/10
60000/60000 [==============================] - 8s 140us/step - loss: 0.1252 - acc: 0.9607
Epoch 9/10
60000/60000 [==============================] - 9s 143us/step - loss: 0.1202 - acc: 0.9630
Epoch 10/10
60000/60000 [==============================] - 8s 137us/step - loss: 0.1147 - acc: 0.9635


# Model 1 b) 

* Input layer → Conv layer1 → Conv layer2 → FC layer → Output layer
* Result:

Epoch 1/10 
60000/60000 [==============================] - 54s 896us/step - loss: 0.2475 - acc: 0.9242
Epoch 2/10
60000/60000 [==============================] - 53s 890us/step - loss: 0.1005 - acc: 0.9705
Epoch 3/10
60000/60000 [==============================] - 53s 891us/step - loss: 0.0789 - acc: 0.9769
Epoch 4/10
60000/60000 [==============================] - 53s 890us/step - loss: 0.0646 - acc: 0.9804
Epoch 5/10
60000/60000 [==============================] - 53s 888us/step - loss: 0.0556 - acc: 0.9830
Epoch 6/10
60000/60000 [==============================] - 53s 888us/step - loss: 0.0510 - acc: 0.9840
Epoch 7/10
60000/60000 [==============================] - 53s 891us/step - loss: 0.0443 - acc: 0.9861
Epoch 8/10
60000/60000 [==============================] - 53s 890us/step - loss: 0.0400 - acc: 0.9872
Epoch 9/10
60000/60000 [==============================] - 53s 890us/step - loss: 0.0376 - acc: 0.9884
Epoch 10/10
60000/60000 [==============================] - 53s 891us/step - loss: 0.0341 - acc: 0.9891


# Model 2

* Input: 28x28
* Ouput: 0, even label(1), odd label(2)
* Input layer → Conv layer1 → Conv layer2 → FC layer → Output layer	
* Result:

Epoch 1/10 
60000/60000 [==============================] - 54s 895us/step - loss: 0.1346 - acc: 0.9524
Epoch 2/10
60000/60000 [==============================] - 54s 894us/step - loss: 0.0578 - acc: 0.9810
Epoch 3/10
60000/60000 [==============================] - 54s 893us/step - loss: 0.0462 - acc: 0.9855
Epoch 4/10
60000/60000 [==============================] - 53s 891us/step - loss: 0.0369 - acc: 0.9883
Epoch 5/10
60000/60000 [==============================] - 53s 890us/step - loss: 0.0317 - acc: 0.9896
Epoch 6/10
60000/60000 [==============================] - 53s 887us/step - loss: 0.0298 - acc: 0.9904
Epoch 7/10
60000/60000 [==============================] - 53s 888us/step - loss: 0.0253 - acc: 0.9917
Epoch 8/10
60000/60000 [==============================] - 53s 887us/step - loss: 0.0226 - acc: 0.9925
Epoch 9/10
60000/60000 [==============================] - 53s 889us/step - loss: 0.0217 - acc: 0.9926
Epoch 10/10
60000/60000 [==============================] - 53s 889us/step - loss: 0.0197 - acc: 0.9934


# Observations:
* The prediction accuracy for model 2 with 3 labels is better than the other two models mainly cause there are only 3 classes to classify.
* The prediction accuracy for the model 1)b) is better compared to 1)a) because the input to the FC layer is after the feature extraction by the Convolutional Neural network layers, therefore prediction not being performed on raw data.  
