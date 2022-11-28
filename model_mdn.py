from .mdn import *

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomTranslation(
    (-0.5, 0.3) ,
    (-0.5, 0.3) ,
    fill_mode='wrap',
    interpolation='bilinear'
)
])


model = models.Sequential()
model.add(data_augmentation)
model.add(layers.Conv2D(128, (4, 4), activation='selu', input_shape=(64, 64, 1)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (4, 4), activation='selu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='selu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='selu'))
model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='selu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, (2, 2), activation='selu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(8, (2, 2), activation='selu'))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='selu'))
#model.add(layers.Dropout(0.15))

model.add(layers.Dense(16, activation='tanh'))

#model.add(layers.Dropout(0.15))

model.add(MDN(2,1))
