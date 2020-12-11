from setting import *
from model import *
from training import generate_training_data

from keras.models import Sequential
from keras.layers import Dense, Dropout

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


# model = Sequential()
# model.add(Dense(units=9,input_dim=7))
# model.add(Dropout(0,5))
# model.add(Dense(units=15, activation='relu'))
# model.add(Dense(output_dim=3,  activation = 'softmax'))
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit((np.array(training_data_x).reshape(-1,7)),( np.array(training_data_y).reshape(-1,3)), batch_size = 256,epochs= 3)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='.'  # каталог, куда сохраняются обученные сети
)

tuner.search_space_summary()

tuner.search((np.array(training_data_x).reshape(-1,7)),  # Данные для обучения
             ( np.array(training_data_y).reshape(-1,3)),  # Правильные ответы
             batch_size=256,  # Размер мини-выборки
             epochs=3,
             validation_split=0.2,
             verbose=1
             )
tuner.results_summary()

# model.save_weights('model.h5')
# model_json = model.to_json()
# with open('model.json', 'w') as json_file:
#     json_file.write(model_json)