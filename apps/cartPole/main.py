import os
from pathlib import Path

import gym
import numpy as np
from utils  import DQNAgent


def learning(env, path, episodes, timesteps, batchSize=32, saving=False, verbose=False):
    # DEFINIR EL AGENTE
    agent = DQNAgent(stateSize=env.observation_space.shape[0], actionSize=env.action_space.n, verbose=verbose)
 
    # para cada epoca de entrenamiento
    for episode in range(episodes):
        if verbose:
            print('epoca:', episode)

        # obtenemos el estado actual del ambiente y lo redimensionamos para
        # el modelo de aprendizaje del agente
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])

        # durante X pasos generaremos una acción sobre el ambiente
        # y obtendremos una recompensa.
        for _ in range(timesteps):
            # rendereamos el ambiente para poder observarlo
            env.render()
            # AGENTE EJECUTA ACCIÓN DADO EL ESTADO ACTUAL
            action = agent.act(state)
            # obtendremos el estado nuevo del ambiente ante el estimulo dado por el agente
            # y también lo redimensionamos
            newState, reward, done, _ = env.step(action) 
            newState = np.reshape(newState, [1, env.observation_space.shape[0]])
            # ALMACENAMOS EN MEMORIA DEL AGENTE EL ESTADO GENERAL DEL AMBIENTE 
            agent.memorize(state, action, reward, newState, done)

            if verbose:
                print('estado:', newState, end=' ')
                # mostramos la recompensa y la determinación que terminó el entrenamiento
                print('accion:', action, 'recompensa:', reward, 'funcionando...' if not done else 'terminado.')

            # actualizamos el estado actual con el nuevo
            state = newState
            # si terminó el entrenamiento, salimos
            if done:
                break

            # SI SE TIENEN LOS DATOS SUFICIENTES, ENTRENAR EL MODELO DEL AGENTE
            if agent.capacity() > batchSize:
                agent.learn(batchSize)

        if saving:
            agent.save(path)
        # cerramos el ambiente
        env.close()


if __name__ == "__main__":
    # cambiando color de la consola, sin utilidad en el ejercicio
    os.system('color 3')

    dirPath = os.path.join(Path(__file__).parent, 'data')
    if not os.path.exists(dirPath):
        os.makedirs(dirPath, exist_ok=True)

    learning(env=gym.make('CartPole-v0'),
             path=os.path.join(dirPath, 'cartpole-dqn.h5'),
             episodes=10,
             timesteps=1000,
             batchSize=32,
             saving=True,
             verbose=True)