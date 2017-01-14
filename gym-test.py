import gym

env = gym.make('CartPole-v0')
env.reset()

for i_episode in xrange(1):
    observation = env.reset()
    for t in xrange(205):
      env.render()
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)

      print("ob: {} reward {} done {}".format(observation,reward,done))      
      if done:
          print("episode finished after {} timesteps".format(t+1))
          break
