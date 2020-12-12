from setting import *

from training import generate_training_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

display_width = 400
display_height = 400

green = (0,255,0)
blue = (0,0,255)
black = (0,0,0)
white = (255,255,255)

pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()

training_data_x, training_data_y = generate_training_data(display,clock)

model = Sequential()
model.add(Dense(units=30,input_dim=7, activation='tanh'))
model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=28, activation='tanh'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit((np.array(training_data_x).reshape(-1,7)),( np.array(training_data_y).reshape(-1,3)), batch_size = 128,epochs= 7)


model.save_weights('model_1.h5')
model_json = model.to_json()
with open('model.json_1', 'w') as json_file:
    json_file.write(model_json)