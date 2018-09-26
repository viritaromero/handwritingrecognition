#Importing Standard Scientific Python Library

import matplotlib.pyplot as plt

#Using a simple dataset of 8×8 gray level images of handwritten digits

from sklearn.datasets import load_digits


#Loading the dataset provided by scikit-learn

digits = load_digits()

#Analyzing a sample image, in this case we are using number 8

import pylab as pl 
pl.gray() 
pl.matshow(digits.images[8]) 
pl.show()

#Analyzing image pixels. Each element represents the pixel of our grayscale image. The value range from 0 to 255 for 8 bit pixel

digits.images[8]

#Visualizing first 15 images

images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
    
import random
from sklearn import ensemble

#Defining variables

n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Creating random indices. The integer division(//) should be used instead.

sample_index=random.sample(range(len(x)),len(x) //5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images

sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets

sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Using the Random Forest Classifier

classifier = ensemble.RandomForestClassifier()

#Fit model with sample data

classifier.fit(sample_images, sample_target)

#Attempt to predict validation data

score=classifier.score(valid_images, valid_target)
print ('Random Tree Classifier:\n') 
print ('Score\t'+str(score))

i=231

pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 

#Make sure to add an extra set of square brackets.

classifier.predict(x[[i]])


