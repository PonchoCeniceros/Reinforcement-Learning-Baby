import os
# no mostrar alertas de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .agents import DQNAgent
# solo para python 3.8 o >
# from .helloWorld import helloWorld, simpleAction