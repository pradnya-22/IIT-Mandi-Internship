
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


Classifier = Sequential()

Classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))
Classifier.add(Conv2D(32, (3, 3), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))
Classifier.add(Flatten())


Classifier.add(Dense(units=128, activation='relu')) 


Classifier.add(Dense(units=3, activation='softmax'))


Classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.4,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


training_set = train_datagen.flow_from_directory('/Users/bhaveshgayakwad/Downloads/Covid19-dataset/train',
                                                 target_size=(64, 64),
                                                 batch_size=4,
                                                 class_mode='categorical') 

test_set = test_datagen.flow_from_directory('/Users/bhaveshgayakwad/Downloads/Covid19-dataset/test',
                                             target_size=(64, 64),
                                             batch_size=4,
                                             class_mode='categorical')


Classifier.fit_generator(training_set,
                          steps_per_epoch=40,
                          epochs=10,
                          validation_data=test_set,
                          validation_steps=8)


from tensorflow.keras.preprocessing import image

test_image = image.load_img('/Users/bhaveshgayakwad/Downloads/Data2/COVID/COVID_3.png',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


result = Classifier.predict(test_image)


predicted_class = np.argmax(result[0])  


class_names = training_set.class_indices 
predicted_class_name = list(class_names.keys())[predicted_class]

print(f"Predicted class: {predicted_class_name}")
