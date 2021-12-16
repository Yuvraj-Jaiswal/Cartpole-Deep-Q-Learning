import gym

env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n

##

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24,activation="relu"))
model.add(Dense(24,activation="relu"))
model.add(Dense(actions,activation="linear"))

policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000,window_length=1)
agent = DQNAgent(model,policy,memory=memory,nb_actions=actions,nb_steps_warmup=10,target_model_update=1e-2)

agent.compile(Adam(),metrics=["mae"])
# agent.fit(env,nb_steps=20000,visualize=False)
agent.load_weights('AgentWeights')
##
import numpy as np
Test_agent = agent.test(env,nb_episodes=5,visualize=True)
print(np.mean(Test_agent.history['episode_reward']))
env.close()

