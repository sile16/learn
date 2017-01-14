import gym
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1] #hack way to get number of actions from last outputshape of model

        #This breaks if we provide a 1 dimensional array as input, need to fix.
        env_dim = self.memory[0][0][0].shape[1]

        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class matt_rl():
    def __init__(self, env ):
        self.env = env

        self.epsilon = .95  # exploration
        self.epsilon_reduction = .95
        self.epsilon_minimum = .05

        # todo Change Exploration over time in a more intelligent way

        self.num_actions = env.action_space.n  # [move_left, stay, move_right]
        self.batch_size=10
        max_memory = 1000
        hidden_size = 120

        self.episode_best = float('-inf')
        self.episode_cum_reward=0


        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_shape=env.observation_space.shape, activation='relu'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=.2), "mse")

        # If you want to continue training from a previous model, just uncomment the line bellow
        #self.load_model()

        # Initialize experience replay object
        self.exp_replay = ExperienceReplay(max_memory=max_memory,discount=.6)
        self.loss = 0.

    def evaluate_learning(self,loss,reward,done):
        #Track cumlative reward for the whole episode
        self.episode_cum_reward += reward
        if done:
            if self.episode_cum_reward > self.episode_best:
                self.episode_best = self.episode_cum_reward
                self.epsilon_update(done)
                print("Best Reward : {}   Epsilon: {}".format(self.episode_best,self.epsilon))

            self.episode_cum_reward = 0

    def epsilon_update(self,done):
        self.epsilon = max(self.epsilon_minimum,self.epsilon*self.epsilon_reduction)


    def get_action(self, observation):
        self.last_observation = observation
        #reduce probability of doing doming random
        if np.random.rand() <= self.epsilon:
            # Todo better than random?  How but changing probability of 1 variable more than others
            # Or specifically doing opposite of our Q function
            return self.env.action_space.sample()
        else:
            q = self.model.predict(observation)
            return np.argmax(q[0])

    def learn(self, action, observation, reward, done):
        # store experience
        self.exp_replay.remember([self.last_observation, action, reward, observation], done)

        # adapt model
        inputs, targets = self.exp_replay.get_batch(self.model, batch_size=self.batch_size)
        loss = self.model.train_on_batch(inputs, targets)

        self.evaluate_learning(loss,reward,done)

        #self.epsilon_update(done)

    def save_model(self):
        self.model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)


    def load_model(self):
        self.model.load_weights("model.h5")


if __name__ == "__main__":
    # parameters

    #env = gym.make('Acrobot-v0')
    env = gym.make('CartPole-v0')
    #env = gym.make('MountainCar-v0')
    #env = gym.make('Pendulum-v0')
    env.reset()

    episodes = 100000

    # initilize AI
    ai = matt_rl(env)
    completed_steps = list()
    t=0

    for i_episode in xrange(episodes):
        observation = env.reset().reshape([1,-1])


        for t in xrange(200):
            env.render()
            action = ai.get_action(observation)
            observation, reward, done, info = env.step(action)
            observation = np.reshape(observation,[1,-1])
            ai.learn(action, observation, reward, done)

            if done:
                break


        completed_steps.append(t)


        last_50 = completed_steps[-50:]
        average = 1.*sum(last_50)/len(last_50)


        if(i_episode % 20 == 0):
            print("episode {} Loss {} average_steps {} best_reward {}".format(i_episode,ai.loss,average,ai.episode_best))



    # Save trained model weights and architecture, this will be used by the visualization code
    ai.save_model()