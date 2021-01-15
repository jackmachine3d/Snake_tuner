from setting import *
from model import *
from training import generate_training_data
from kerastuner import RandomSearch


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

"""Создание тюнера"""
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    directory='C:\\Users\\agolovanov\\PycharmProjects\\Snake_tuner\\model'  # каталог, куда сохраняются обученные сети
)

tuner.search_space_summary()

tuner.search((np.array(training_data_x).reshape(-1,7)),
             ( np.array(training_data_y).reshape(-1,3)),
             batch_size=2048,
             epochs=7,
             validation_split=0.2,
             verbose=1

             )
tuner.results_summary()

# Вывод лучших моделей
models = tuner.get_best_models(num_models=10)

with open("result.txt",'w') as f:
    f.write(models)


