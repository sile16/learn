import gym
import random
env = gym.make('CartPole-v0')

env.monitor.start('/tmp/cartpole-experiment-3', force=True)

# its simulated annealing like

bestSteps = 0
best = [0, 0, 0, 0]
alpha = 1

for i_episode in xrange(80):

    test = [best[i] + (random.random() - 0.5)*alpha for i in range(4)]

    score = 0
    for ep in range(10):  # <-- key thing was to figure out that you need to do 10 tests per point
        observation = env.reset()
        for t in xrange(200): # <-- because you can't go over 200 you need to gain score hight else where
            env.render()
            if sum(observation[i]*test[i] for i in range(4)) > 0:
                action = 1
            else:
                action = 0
            observation, reward, done, info = env.step(action)
            if done:
                break

        score += t

    if bestSteps < score:
        bestSteps = score
        best = test
        alpha *= .9

    print "test:", "[%+1.2f %+1.2f %+1.2f %+1.2f]" % tuple(test), score,
    print "best:", "[%+1.2f %+1.2f %+1.2f %+1.2f]" % tuple(best), bestSteps, alpha


print "best", best, bestSteps

env.monitor.close()
