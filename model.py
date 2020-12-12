from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(hp):

    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Dense(units=hp.Int('units_input',  # Полносвязный слой с разным количеством нейронов
                                 min_value=10,  # минимальное количество нейронов - 128
                                 max_value=32,  # максимальное количество - 1024
                                 step=2),
                    input_dim=7,
                    activation=activation_choice))
    # Создаем цикл для динамического выбора кол-во скрытых слоев + кол-во нейронов в скрытом слое
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(Dense(units=hp.Int('units_' + str(i),
                                     min_value=10,
                                     max_value=32,
                                     step=2),
                        activation=activation_choice))

    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

