import os
# no mostrar alertas de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class DQNAgent:
    def __init__(self, stateSize, actionSize, verbose=False):
        """ Inicializar el agente

        Args:
            stateSize (int): longitud del estado entregado por el ambiente
            actionSize (int): longitud de la acción tomada por el agente
            verbose (bool, optional): habilitar los comentarios. Defaults to False.
        """
        self.verbose = verbose
        # parámetros para el modelo
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learning_rate = 0.001
        # modelo generado para el aprendizaje del agente
        self.model = self._build_model()
        # reservamos memoria para almacenar los estados requeridos para el
        # aprendizaje
        self._memory = deque(maxlen=2000)

    def _build_model(self):
        """ Se define el modelo de red neuronal que se empleará para el aprendizaje
            del agente.
        Returns:
            keras.models.Sequential: modelo definido
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(5, input_dim=self.stateSize, activation="relu"))
        model.add(keras.layers.Dense(10, activation="relu"))
        model.add(keras.layers.Dense(self.actionSize, activation="linear"))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


    def capacity(self):
        """ Entregar la capacidad actual de nuestro buffer de memoria

        Returns:
            int: capacidad actual de nuestro buffer de memoria
        """
        return len(self._memory)


    def act(self, state):
        # generamos nuestros Q-values mediante la predicción con nuestro modelo
        QValues = self.model.predict(state)
        if self.verbose: print('QValues:', QValues, end=' ')
        # retornamos la mejor accion generada
        return np.argmax(QValues[0])


    def memorize(self, state, action, reward, newState, done):
        """ Alamacenar en la memoria del agente el estado general del ambiente en
            un instante del entrenamiento 

        Args:
            state (numpy.ndarray): estado actual del ambiente
            action (numpy.int64): acción realizada por el agente
            reward (float): recompensa otorgada en base a la acción
            newState (numpy.ndarray): estado nuevo a partir de la acción generada
            done (bool): si el ambiente ha terminado su ejecución o no
        """
        self._memory.append((state, action, reward, newState, done))

    def load(self, filename):
        """ Cargar una matriz de pesos en el modelo

        Args:
            filename (string): nombre del archivo con la matriz de pesos
        """
        self.model.load_weights(filename)

    def save(self, filename):
        """ Guardar la matriz de pesos actual del modelo

        Args:
            filename (string): nombre del archivo con la matriz de pesos
        """
        self.model.save_weights(filename)


    def learn(self, batchSize, gamma=0.95):
        """ Entrenar el modelo en base a la memoria acumulada del agente

        Args:
            batchSize (int): porción de la memoria que se empleará para 
                             entrenar el modelo
        """
        if self.verbose: print('APRENDIENDO...')
        # se extraen un batch con los estados más recientes
        batch = random.sample(self._memory, batchSize)

        for state, action, reward, newState, done in batch:
            # empleamos la recompensa del estado general como target para entrenar,
            # esto en caso de que en dicho estado el ambiente siga activo. En caso contrario,
            # se calcula la recompensa correspondiente según la ecuación de Bellman
            target = reward if not done else reward + gamma*np.amax(self.model.predict(newState)[0])
            # calculamos los Q-values con nuestro modelo actual
            target_f = self.model.predict(state)
            # reemplazamos el Q-value correspondiente a la acción generada en el estado general
            # por la recompensa calculada
            target_f[0][action] = target
            # con los Q-values como target, entrenamos nuestro modelos
            self.model.fit(state, target_f, epochs=1, verbose=0)


# https://github.com/gelanat/reinforcement-learning/blob/master/SARSA.ipynb
# https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287