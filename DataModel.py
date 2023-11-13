import numpy as np 
import cv2 as cv
import matplotlib as plt
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from keras import datasets, layers, models
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
import imghdr

#-----identify directory that data comes from---------
directory = 'data'

#--------specify what file extensions we wanna look for in our data----------
extensions = ['jpeg', 'jpg', 'bmp', 'png']

#print(os.listdir(os.path.join(directory, 'benign')))

#image = cv.imread(os.path.join('data', 'benign', '1514.jpg'))
#plt.imshow(image)
#plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
#plt.show()


#-----loop to trim off unwanted images that dont fit our criteria-----
for i in os.listdir(directory):
    for j in os.listdir(os.path.join(directory, i)):
        path = os.path.join(directory, i, j)
        try:
            pic = cv.imread(path)
            end = imghdr.what(path)
            if end not in extensions:
                os.remove(path)
        except Exception as e:
            print("Error w/ picture {}".format(path))

CNNdata = tf.keras.utils.image_dataset_from_directory('data')
CNNdata_iter = CNNdata.as_numpy_iterator()
stack = CNNdata_iter.next()

#-------------stack[0] contains the image data and stack[1] has the info for either benign or melanoma--------------


scaleddata = CNNdata.map(lambda x, y: (x/255, y))

#------------------------------------------------------------------------------
#---------A title of 0 means benign. A title of 1 means melanoma---------------
#------------------------------------------------------------------------------

group = scaleddata.as_numpy_iterator().next()
#print(group[0].max())
#pic, x = plt.subplots(ncols=4, figsize =(20, 20))
#for i, image in enumerate(group[0][:4]):
#    x[i].imshow(image)
#    x[i].title.set_text(group[1][i])
#plt.show()

#-------partition data----------
trainsize = int(len(CNNdata)*.7)
evaluatesize = int(len(CNNdata)*.2)+1
testsize = int(len(CNNdata)*.1)+1
# print(train)
# print(evaluate)
# print(test)
trainingdata = CNNdata.take(trainsize)
evaluatedata = CNNdata.skip(trainsize).take(evaluatesize)
testdata = CNNdata.skip(trainsize+evaluatesize).take(testsize)
print(len(testdata))

model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation = 'relu', input_shape = (256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])
#print(model.summary())

log = 'log'
callback = tf.keras.callbacks.TensorBoard(log_dir=log)
final = model.fit(trainingdata, epochs=20, validation_data = evaluatedata, callbacks =[callback])
# print(final.history)

# visual = plt.figure()
# plt.plot(final.history['accuracy'], color = 'green', label = 'accuracy')
# plt.plot(final.history['val_accuracy'], color = 'red', label = 'val_accuracy')
# visual.suptitle('Accuracy', fontsize = 16)
# plt.legend()
# plt.show()


#------ lets evaluate the data ------------
precision = Precision()
biacc = BinaryAccuracy()
recall = Recall()

for i in testdata.as_numpy_iterator():
    X, y = i
    #predicted value
    ypred = model.predict(X)
    #actual vs predicted
    precision.update_state(y, ypred)
    biacc.update_state(y, ypred)
    recall.update_state(y, ypred)

print('Precision:', precision.result().numpy())
print('Recall:', recall.result().numpy())
print('BinaryAccuracy:', biacc.result().numpy())

benigntestimg = cv.imread('b3.jpg')
resizedtest1 = tf.image.resize(benigntestimg, (256, 256))
benignpred = model.predict(np.expand_dims(resizedtest1/255, 0))
print(benignpred)
if benignpred < 0.5:
    print("Predicted Class is Benign")
else:
    print("Predicted Class is Melanoma")

melanomatestimg = cv.imread('m3.jpg')
resizedtest2 = tf.image.resize(melanomatestimg, (256, 256))
melanomapred = model.predict(np.expand_dims(resizedtest2/255, 0))
print(melanomapred)
if melanomapred < 0.5:
    print("Predicted Class is Benign")
else:
    print("Predicted Class is Melanoma")

#--------save the model for use------------------
#model.save(os.path.join('models', 'modelV1.0'))

#------Line to load model in future so you dont need to regenerate it every time--------
#loadedmodel = load_model(os.path.join('models', 'modelV1.0'))
