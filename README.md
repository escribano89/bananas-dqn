# Udacity - Deep Reinforcement Learning Nanodegree (Navigation)

## Project Details

This is the first project of the Deep Reinforcement Learning nanodegree. The goal of the agent is to gather yellow bananas and avoid the blue ones. The environment has 4 discrete actions (the agent's movement, forward, backward turn left and right) and a continuous observation space with 37 values representing the speed and the ray-based perception of objects around the agent's forward direction.

        Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
        
For each yellow banana that is collected, the agent is given a reward of +1. The blue ones give -1 reward. We consider that the problem is solved if the agent receives an average reward (over 100 episodes) of at least +13.

The project has been developed using the *Lunar Landing* solution project as a foundation.
        
## Requirements
In order to prepare the environment, follow the next steps after downloading this repository:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	* __Windows__: 
	```bash
	conda create --name dqn python=3.6 
	activate drlnd
	```
* Min install of OpenAI gym
	* If using __Windows__, 
		* download [swig for windows](http://www.swig.org/Doc1.3/Windows.html) and add it the PATH of windows
		* install [ Microsoft Visual C++ Build Tools ](https://visualstudio.microsoft.com/es/downloads/).
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install the dependencies under the folder python/
```bash
	cd python
	pip install .
```
* Create an IPython kernel for the `drlnd` environment
```bash
	python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
* Download the Unity Environment (thanks to Udacity) which matches your operating system
	* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
	* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
	* [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
	* [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

* Unzip the downloaded file and move it inside the project's root directory
* Change the kernel of you environment to `drlnd`
* Open the **main.py** file and change the path to the unity environment appropriately (banana_executable_path)

## Getting started

If you want to train the agent, execute the **main.py** file setting the ***is_training*** variable to True. Otherwise if you want to test your agent then set that variable to False. 
In case of reaching the goal the weights of the neural network will be stored in the checkpoint file in the root folder.

By default, the main.py file is prepared to test our model.

## Resources

* report.pdf: A document that describes the details of the implementation and future proposals.
* agents/deep_q_network: the implemented agent using a deep q network architecture
* models/neural_network: the neural network model
* utils/replay_buffer: a class for handling the experience replay
* python/: needed files to run the unity environment
* main.py: Entry point to train or test the agent
* checkpoint.pth: Our model's weights ***(Solved in less than 800 episodes)***

## Trace of the training

* Episode 100	Average Score: 1.52
* Episode 200	Average Score: 6.27
* Episode 300	Average Score: 8.84
* Episode 400	Average Score: 10.87
* Episode 500	Average Score: 11.69
* Episode 600	Average Score: 11.52
* Episode 700	Average Score: 13.79
* Episode 740	Average Score: 14.03
* Environment solved in **740 episodes**!	Average Score: **14.03**

![Training](https://github.com/escribano89/bananas-dqn/blob/main/training.png)

## Video

You can find an example of the trained agent [here](https://youtu.be/sDLG-Xxp-l8)

[![Navigation](https://img.youtube.com/vi/sDLG-Xxp-l8/0.jpg)](https://www.youtube.com/watch?v=sDLG-Xxp-l8)
