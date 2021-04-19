from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Dropout, Activation, Flatten,AveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2
import keras.backend as K

from utils import *
    
shape = (224, 224, 3)
X_train, X_test = setup_generator('/content/sample_data/food-101/food-101/train', '/content/sample_data/food-101/food-101/test', 32, shape[:2])

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
x = base_model.output
x = AveragePooling2D()(x)

x = Dropout(.5)(x)
x = Flatten()(x)
predictions = Dense(X_train.num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
opt = SGD(lr=.1, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='/content/sample_data/food-101/food-101/model4_lan2.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('/content/sample_data/food-101/food-101/model4.log')
def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return .0002
    elif epoch < 15:
        return 0.00002
    else:
        return .0000005
lr_scheduler = LearningRateScheduler(schedule)
model.summary()
