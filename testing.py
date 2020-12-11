import pygame as pygame
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from setting import *
from keras.models import model_from_json

conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
conf.log_device_placement = True

session = tf.compat.v1.Session(config=conf)
set_session(session)



def start_game(model, display, clock):
    max_score = 3
    avg_score = 0
    test_games = 1000
    steps_per_game = 2000

    for _ in range(test_games):
        snake_start, snake_position, target_position, score = first_position()

        count_same_direction = 0
        prev_direction = 0

        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, target_position)
            predictions = []

            predicted_direction = np.argmax(np.array(model.predict(np.array([is_left_blocked, is_front_blocked, \
                                                                             is_right_blocked,
                                                                             apple_direction_vector_normalized[0], \
                                                                             snake_direction_vector_normalized[0],
                                                                             apple_direction_vector_normalized[1], \
                                                                             snake_direction_vector_normalized[
                                                                                 1]]).reshape(-1, 7)))) - 1

            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                break
            snake_position, target_position, score = play_game(snake_start, snake_position, target_position,
                                                              button_direction, score, display, clock)

            if score > max_score:
                max_score = score

        avg_score += score

    return max_score, avg_score / 1000


json_file = open('model.json', 'r')
loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('model.h5')


display_width = 500
display_height = 500
pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()
max_score, avg_score = start_game(model,display,clock)
print("Maximum score achieved is:  ", max_score)
print("Average score achieved is:  ", avg_score)