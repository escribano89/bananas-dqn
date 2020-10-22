# -*- coding: utf-8 -*-
from unityagents import UnityEnvironment
from agents.deep_q_network import DQN
import matplotlib.pyplot as plt
import numpy as np

banana_executable_path = "D:\\source\\bananas-dqn\\Banana.exe"
# Prepare the unity environment
env = UnityEnvironment(file_name=banana_executable_path)

# Get the default brain info
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=True)[brain_name] 
brain = env.brains[brain_name]
state_size = len(env_info.vector_observations[0])
action_size = brain.vector_action_space_size

is_training = False

if is_training:
    # Use our DQN agent to train the model
    agent = DQN(state_size, action_size)
    scores = agent.train(env, brain_name)
    
    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

else:
    # test the trained agent
    agent = DQN(state_size, action_size)
    agent.test(env, brain_name, 'checkpoint.pth' )

# Show the result of our training

# Close the environment
env.close()


