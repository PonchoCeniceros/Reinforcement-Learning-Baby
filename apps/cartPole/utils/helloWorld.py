import gym
from time import sleep


def helloWorld(env, timesteps):
    # una sola epoca de entrenamiento
    env.reset()
    # durante X pasos generaremos una acción sobre el ambiente
    # y obtendremos una recompensa.
    for _ in range(timesteps):
        # rendereamos el ambiente para poder observarlo
        env.render()
        # generamos una acción random. por ahora solo nos interesa
        # observar la recompensa que obtendremos y determinar cuando hemos
        # fracasado. Tal caso se da cuando el poste está a ~12° de la normal
        _, reward, done, _ = env.step(( action := env.action_space.sample() ))
        # mostramos la recompensa y la determinación que terminó el entrenamiento
        print('accion:', action, 'recompensa:', reward, 'funcionando...' if not done else 'terminado.')
        # esperamos un seg para observar detenidamente el proceso
        sleep(1)
    # cerramos el ambiente
    env.close()


def simpleAction(env, timesteps, action):
    # una sola epoca de entrenamiento
    env.reset()
    # durante X pasos generaremos una acción sobre el ambiente
    # y obtendremos una recompensa.
    for _ in range(timesteps):
        # rendereamos el ambiente para poder observarlo
        env.render()
        # generamos una acción random. por ahora solo nos interesa
        # observar la recompensa que obtendremos y determinar cuando hemos
        # fracasado. Tal caso se da cuando el poste está a ~12° de la normal
        obs, reward, done, _ = env.step(action)
        # en este caso obs contiene el estado del ambiente a la acción generada. El estado del ambiente
        # cart pole es:
        # 1.La posición del carro a lo largo del eje horizontal unidimensional
        # 2.La velocidad del carro
        # 3.El ángulo del poste
        # 4.La velocidad angular del poste         
        print('estado:', obs, end=' ')
        # mostramos la recompensa y la determinación que terminó el entrenamiento
        print('accion:', action, 'recompensa:', reward, 'funcionando...' if not done else 'terminado.')
        # esperamos un seg para observar detenidamente el proceso
        sleep(1)
        # si terminó el entrenamiento, salimos
        if done :
            break
    # cerramos el ambiente
    env.close()

