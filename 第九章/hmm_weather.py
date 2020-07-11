'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/5/9
'''

import numpy as np
from hmmlearn import hmm
# 隐藏状态：3个天气
states = ["sum", "cloud", "rain"]
n_states = len(states)
# 观测状态：3种行为
observations = ["play", "shop","sleep"]
n_observations = len(observations)
start_probability = np.array([0.5, 0.375, 0.125])
transition_probability = np.array([
  [0.5, 0.375, 0.125],
  [0.25, 0.125, 0.625],
  [0.25, 0.375, 0.375]
])
emission_probability = np.array([
  [0.6, 0.4, 0.0],
  [0.4, 0.5, 0.1],
  [0.1, 0.2, 0.7]
])
#用于离散观测状态
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability
print(model)
# 打球，购物和打球
seen = np.array([0,1,0]).reshape(1, -1).T
logprob,  behaviour = model.decode(seen, algorithm="viterbi")
print(np.array(states)[behaviour])
print(model.score(seen))
# 得到观测序列的概率 ln0.06228≈−2.776
print(np.exp(model.score(seen)))

