from time import time
import numpy as np
from scipy import sparse as sp
from multiprocessing import Pool, Manager, cpu_count
from utils import ArrEq, Verbose
from numpy import savez

__all__ = [
  'States',
  'Actions',
  'Rewards',
  'Policy',
  'StateTransitionProbability',
  'MarkovDecisionProcess'
]



class States:

  def __init__(self, state_lists, cycles=None, terminal_states=None, dtype=np.float32):
    self.__data = None
    self.__cycles = None
    self.terminal_states = None
    self.update(
      state_lists, cycles=cycles, terminal_states=terminal_states, dtype=dtype
    )


  def update(self, state_lists, cycles=None, terminal_states=None, dtype=np.float32):

    self.__data = [
      np.array(state_list, dtype=dtype) for state_list in state_lists
    ]
    if cycles is None:
      self.__cycles = [np.inf] * len(state_lists)
    elif len(cycles)==len(state_lists):
      self.__cycles = np.array(cycles, dtype=dtype)
    else:
      raise ValueError(
        "operands could not be broadcast together with shapes ({},) ({},)".format(len(state_lists), len(cycles))
      )
    if terminal_states is None:
      self.terminal_states = []
    else:
      self.terminal_states = list(
        [np.array(state, dtype=self.dtype) for state in terminal_states]
      )


  def __getitem__(self, key):

    if len(self.__data)==1:
      return np.array(self.__data[0][key])
    if isinstance(key, int):
      indices = np.unravel_index(key, shape=self.shape)
      return np.array(
        [state_list[idx] for (state_list, idx) in zip(self.__data, indices)],
        dtype=self.dtype
      )
    elif isinstance(key, tuple):
      if np.any([isinstance(slice, k) for k in key]):
        raise KeyError("class States does not support slice method.")
      if len(key)==1:
        indices = np.unravel_index(key[0], shape=self.shape)
        return np.array(
          [state_list[idx] for (state_list, idx) in zip(self.__data, indices)],
          dtype=self.dtype
        )
      elif len(key)==len(self.shape):
        return np.array(
          [state_list[k] for (state_list, k) in zip(self.__data, key)],
          dtype=self.dtype
        )
      else:
        raise KeyError("Number of indices mismatch.")
    else:
      raise KeyError("Unsupported key type.")


  def __iter__(self):

    class SubIterator:
      def __init__(cls, data, shape, num_states, dtype):
        cls.dtype = dtype
        cls.__data = data
        cls.__shape = shape
        cls.__num_states = num_states
        cls.__dim = len(cls.__shape)
        cls.__counters = [0] * cls.__dim
        cls.__total_count = 0
      def __next__(cls):
        cls.__total_count += 1
        if cls.__total_count > cls.__num_states:
          raise StopIteration()
        else:
          ret = [state_list[count] for count, state_list in zip(cls.__counters, cls.__data)]
        for d in range(cls.__dim)[::-1]:
          if cls.__counters[d] < (cls.__shape[d]-1):
            cls.__counters[d] += 1
            break
          else:
            cls.__counters[d] = 0
        return np.array(ret, dtype=cls.dtype)
    return SubIterator(self.__data, self.shape, self.num_states, self.dtype)


  @property
  def shape(self):
    return tuple(
      [len(state_list) for state_list in self.__data]
    )


  @property
  def dtype(self):
    return self.__data[0].dtype


  @property
  def num_states(self):
    return np.prod(self.shape)


  def computeBarycentric(self, item):

    if not isinstance(item, np.ndarray):
      item = np.array(item, dtype=self.dtype)
    i = []
    p = []
    for (state_list, cycle, x) in zip(self.__data, self.__cycles, item):
      idx = np.searchsorted(state_list, x)
      if idx < len(state_list):
        if x == state_list[idx]:
          i.append(np.array([idx], dtype=int))
          p.append(np.ones((1,), dtype=self.dtype))
        else:
          d1 = state_list[idx] - x
          if idx == 0:
            if cycle is np.inf:
              i.append(np.array([idx], dtype=int))
              p.append(np.ones((1,), dtype=self.dtype))
            else:
              d2 = x - state_list[-1] + cycle
              i.append(np.array([idx, len(state_list)-1]))
              p.append(np.array([d1, d2]) / (d1 + d2))
          else:
            d2 = x - state_list[idx - 1]
            i.append(np.array([idx - 1, idx], dtype=int))
            p.append(np.array([d1, d2]) / (d1 + d2))
      else:
        if cycle is np.inf:
          i.append(np.array([idx-1], dtype=int))
          p.append(np.ones((1,), dtype=self.dtype))
        else:
          d1 = x - state_list[-1]
          d2 = state_list[0] - x + state_list[0]
          i.append(np.array([0, idx-1]))
          p.append(np.array([d1, d2]) / (d1 + d2))
    indices = np.zeros((1,), dtype=int)
    probs = np.ones((1,), dtype=self.dtype)
    for dim, (idx, prob) in enumerate(zip(i, p)):
      indices = np.repeat(indices, len(idx)) + np.tile(idx, len(indices))
      if dim < len(self.shape)-1:
        indices *= self.shape[dim+1]
      probs = np.repeat(probs, len(idx)) * np.tile(prob, len(probs))
    return indices, probs


  def index(self, state):

    if len(state)==len(self.__data):
      n = 0
      for idx, x in enumerate(state):
        n += np.argmin(np.abs(self.__data[idx] - x)) * np.prod(self.shape[idx+1:], dtype=int)
      return n
    else:
      raise ValueError(
        "operands could not be broadcast together with shapes ({},) ({},)".format(len(self.__data), len(state))
      )


  def info(self, return_data=False, return_cycles=False):

    if return_data:
      if return_cycles:
        return self.__data, self.__cycles
      else:
        return self.__data
    else:
      if return_cycles:
        return self.__cycles
      else:
        return None

  # End of class States 



class Actions:

  def __init__(self, action_list, dtype=np.float32):

    self.__data = None
    self.__data_ndarr = None
    self.update(action_list, dtype)


  def index(self, item):

    if isinstance(item, np.ndarray):
      return self.__data_ndarr.index(item)
    else:
      return self.__data.index(item)


  def __getitem__(self, key):

    return self.__data[key]


  def __iter__(self):
    return self.__data.__iter__()


  @property
  def dtype(self):
    return self.__data[0].dtype


  @property
  def shape(self):
    return (len(self.__data),) + self.__data[0].shape


  @property
  def num_actions(self):
    return len(self.__data)


  def update(self, action_list, dtype=np.float32):
    self.__data = list()
    self.__data_ndarr = list()
    for item in action_list:
      self.__data.append(np.array(item, dtype=dtype))
      self.__data_ndarr.append(ArrEq(self.__data[-1]))


  def tolist(self):
    return self.__data


  def toarray(self):
    return np.array(self.__data)

  # End of class Actions



class Rewards:

  def __init__(self, states, actions, dtype=np.float32, sparse=False):

    shape = (states.num_states, actions.num_actions)
    if sparse:
      self.__data = sp.dok_matrix(shape, dtype=dtype)
    else:
      self.__data = np.zeros(shape, dtype=dtype)


  def __setitem__(self, key, val):

    self.__data[key] = val


  def __getitem__(self, key):

    return self.__data[key]


  def __iter__(self):

    return self.__data.__iter__()


  @property
  def dtype(self):
    return self.__data.dtype


  @property
  def shape(self):
    return self.__data.shape


  @property
  def issparse(self):
    return isinstance(self.__data, sp.spmatrix)


  def tocsr(self):

    if self.issparse:
      if not isinstance(self.__data, sp.csr_matrix):
        self.__data = self.__data.tocsr()
    else:
      self.__data = sp.csr_matrix(self.__data, dtype=self.dtype)
    return self.__data


  def todok(self):

    if self.issparse:
      if not isinstance(self.__data, sp.dok_matrix):
        self.__data = self.__data.todok()
    else:
      self.__data = sp.dok_matrix(self.__data, dtype=self.dtype)
    return self.__data


  def toarray(self, copy=False):

    if self.issparse:
      return self.__data.toarray()
    else:
      if copy:
        return self.__data.copy()
      else:
        return self.__data


  def update(self, data):
    self.__data = data


  def load(self, filename):
    filetype = filename.split('.')[-1]
    if filetype=='npz':
      self.__data = sp.load_npz(filename)
    elif filetype=='npy':
      self.__data = np.load(filename)


  def save(self, filename):
    if self.issparse:
      self.__data = sp.save_npz(filename, self.__data)
    else:
      self.__data = np.save(filename, self.__data)

  # End of class Rewards



class StateTransitionProbability:

  def __init__(self, states, actions, dtype=np.float32):

    self.__data = sp.dok_matrix(
      (states.num_states * actions.num_actions, states.num_states),
      dtype=dtype
    )


  def __setitem__(self, key, val):

    if isinstance(key, tuple):
      if len(key)==1:
        return self.__data[key[0]]
      elif len(key)==2:
        self.__data[key] = val
      elif len(key)==3:
        if isinstance(key[0], slice):
          raise IndexError('First index does not supports slice method.')
        if isinstance(key[1], slice):
          start = key[1].start
          stop = key[1].stop
          step = key[1].step
          shape = self.shape
          self.__data[
            slice(
              np.ravel_multi_index((key[0], start), shape[:2]),
              np.ravel_multi_index((key[0], stop), shape[:2]),
              step
            ),
            key[2]
          ] = val
        else:
          self.__data[
            np.ravel_multi_index(key[:2], self.shape[:2]),
            key[2]
          ] = val
      else:
        raise IndexError('Indices mismatch.')
    elif isinstance(key, int) or isinstance(key, slice):
      self.__data[key] = val
    else:
      raise IndexError('Indices mismatch.')


  def __getitem__(self, key):

    if isinstance(key, tuple):
      if len(key)==1:
        return self.__data[key[0]]
      elif len(key)==2:
        return self.__data[key]
      elif len(key)==3:
        if isinstance(key[0], slice):
          raise IndexError('First index does not supports slice method.')
        if isinstance(key[1], slice):
          start = key[1].start
          stop = key[1].stop
          step = key[1].step
          shape = self.shape
          return self.__data[
            slice(
              np.ravel_multi_index((key[0], start), shape[:2]),
              np.ravel_multi_index((key[0], stop), shape[:2]),
              step
            ),
            key[2]
          ]
        else:
          return self.__data[
            np.ravel_multi_index(key[:2], self.shape[:2]),
            key[2]
          ]
    elif isinstance(key, int) or isinstance(key, slice):
      return self.__data[key]
    else:
      raise IndexError('Indices mismatch.')


  def __iter__(self):
    return self.__data.__iter__()


  @property
  def dtype(self):
    return self.__data.dtype


  @property
  def shape(self):
    sa, s = self.__data.shape
    return (s, sa // s, s)


  def dot(self, other):
    return self.__data.dot(other)


  def tocsr(self):
    if not sp.isspmatrix_csr(self.__data):
      self.__data = self.__data.tocsr()
  

  def todok(self):
    if not sp.isspmatrix_dok(self.__data):
      self.__data = self.__data.todok()


  def tospmat(self):
    return self.__data


  def toarray(self):
    return self.__data.toarray()


  def update(self, data):
    self.__data = data


  def load(self, filename):
    self.__data = np.load(filename, allow_pickle=True)


  def save(self, filename):
    np.save(filename, self.__data, allow_pickle=True)

  # End of class StateTransitionProbability 



class Policy:

  def __init__(self, states, actions, dtype=int):
    self.__states = states
    self.__actions = actions
    self.__data  = None
    self.__I = np.eye(self.__actions.num_actions, dtype=bool)
    self.reset(dtype=dtype)


  def __setitem__(self, key, val):
    self.__data[key] = min(val, self.__actions.num_actions - 1)


  def __getitem__(self, key):
    return self.__data[key]


  def __iter__(self):
    return self.__data.__iter__()


  def __str__(self):
    return self.__data.__str__()


  @property
  def dtype(self):
    return self.__data.dtype


  @property
  def shape(self):
    return self.__data.shape


  def one_hot(self):
    return self.__I[self.__data]


  def get_action(self, state):
    S, P = self.__states.computeBarycentric(state)
    action = 0
    for s, p in zip(S, P):
      action += p * self.__actions[int(self.__data[s])]
    return action


  def update(self, data):
    self.__data = data


  def reset(self, dtype=int):
    self.__data  = np.random.randint(
      0, self.__actions.num_actions, self.__states.num_states, dtype=dtype
    )


  def toarray(self, copy=False):
    if copy:
      return self.__data.copy()
    else:
      return self.__data


  def load(self, filename):
    self.__data = np.load(filename)


  def save(self, filename):
    np.save(filename, self.__data)

  # End of class Policy



class MarkovDecisionProcess:

  def __init__(self,
    states=None, actions=None, rewards=None,
    state_transition_probability=None, policy=None,
    discount = 0,
  ):

    self.states  = States([[]]) if states is None else states
    self.actions = Actions([]) if actions is None else actions
    self.rewards = Rewards(self.states, self.actions) if rewards is None else rewards
    self.discount = min(
      np.array(discount, dtype=self.rewards.dtype).item(),
      np.array(1, dtype=self.rewards.dtype).item()-np.finfo(self.rewards.dtype).eps
    )
    self.state_transition_probability = StateTransitionProbability(
      self.states, self.actions
    ) if state_transition_probability is None else state_transition_probability
    self.policy = Policy(self.states, self.actions) if policy is None else policy
    self.__sampler = None
    self.__sample_reward = False
    self.__n_repeat = 1


  def _worker(self, queue, state):
    spmat = sp.dok_matrix(
      self.state_transition_probability.shape[1:],
      dtype=self.state_transition_probability.dtype
    )
    if self.__sample_reward:
      arr = np.zeros(self.rewards.shape[1:], dtype=self.rewards.dtype)
    if not ArrEq(state) in self.states.terminal_states:
      for action_idx, action in enumerate(self.actions):
        for _ in range(self.__n_repeat):
          if self.__sample_reward:
            next_state, reward = self.__sampler(state, action)
            arr[action_idx] += reward / self.__n_repeat
          else:
            next_state = self.__sampler(state, action)
          next_state_indices, probs = self.states.computeBarycentric(next_state)
          for next_state_idx, prob in zip(next_state_indices, probs):
            spmat[action_idx, next_state_idx] += prob / self.__n_repeat
    queue.put(1)
    if self.__sample_reward:
      return np.array([spmat.tocsr(), arr], dtype=object)
    else:
      return spmat.tocsr()


  def sample(self, sampler, n_repeat=1, sample_reward=False, verbose=True):

    verbose = Verbose(verbose)
    verbose('Start sampling...')
    start_time = time()
    self.__sampler = sampler
    self.__sample_reward = sample_reward
    self.__n_repeat = max(n_repeat, 1)
    queue = Manager().Queue()
    with Pool(cpu_count()) as p:
      data = p.starmap_async(
        self._worker,
        [(queue, state) for state in self.states]
      )
      counter = 0
      tic = time()
      while counter < self.states.num_states:
        counter += queue.get()
        if time()-tic > 0.1:
          progress = counter / self.states.num_states
          rt = (time() - start_time) * (1 - progress) / progress
          rh = rt // 3600
          rt %= 3600
          rm = rt // 60
          rs = rt % 60
          progress *= 100
          verbose('Sampling progress: %5.1f %%... (%dh %dm %ds rem.)'%(progress, rh, rm, rs))
          tic = time()
      if self.__sample_reward:
        data = np.array(
          data.get(),
          dtype=object
        )
        self.state_transition_probability.update(
          sp.vstack(data[:, 0])
        )
        self.rewards.update(
          np.array(
            data[:, 1].tolist(),
            dtype=self.rewards.dtype
          )
        )
      else:
        self.state_transition_probability.update(
          sp.vstack(
            data.get()
          )
        )
    self.__sampler = None
    end_time = time()
    verbose('Sampling is done. %f (sec) elapsed.\n'%(end_time - start_time))


  def load(self, filename):

    data = np.load(filename, allow_pickle=False)
    state_lists = []
    for idx in range(data['states.num_lists'].item()):
      state_lists.append(data['states.data.'+str(idx)])
    self.states = States(
      state_lists,
      cycles = data['states.cycles'],
      terminal_states = data['states.terminal_states']
    )

    self.actions = Actions(data['actions.data'])

    self.policy = Policy(
      self.states,
      self.actions
    )
    self.policy.update(
      data['policy.data']
    )
    self.discount = data['discount'].item()
    

  def save(self, filename):

    if not sp.isspmatrix_csr(self.state_transition_probability.tospmat()):
      self.state_transition_probability.tocsr()

    kwargs = {
      'states.num_lists': len(self.states.shape),
      'states.cycles': self.states.info(return_cycles=True),
      'states.terminal_states': self.states.terminal_states,
      'actions.data': self.actions.toarray(),
      'rewards.issparse': self.rewards.issparse,
      'state_transition_probability.data': self.state_transition_probability.tospmat().data,
      'state_transition_probability.indices': self.state_transition_probability.tospmat().indices,
      'state_transition_probability.indptr': self.state_transition_probability.tospmat().indptr,
      'policy.data': self.policy.toarray(),
      'discount': self.discount
    }
    for idx, state_list in enumerate(self.states.info(return_data=True)):
      kwargs['states.data.'+str(idx)] = state_list
    if self.rewards.issparse:
      kwargs['rewards.data'] = self.rewards.tocsr().data,
      kwargs['rewards.indices'] = self.rewards.tocsr().indices,
      kwargs['rewards.indptr'] = self.rewards.tocsr().indptr,
    else:
      kwargs['rewards.data'] = self.rewards.toarray()

    savez(filename, **kwargs)

  # End of class MarkovDecisionProcess


if __name__=="__main__":

  states = States(np.linspace(0, 1, 100))

  actions = Actions(np.linspace(0, 1, 10))

  rewards = Rewards(states, actions)

  state_transition_prob = StateTransitionProbability(states, actions)
  for s in range(states.num_states):
    for a in range(actions.num_actions):
      state_transition_prob[s, a, s] = 1.

  policy = Policy()

  mdp = MarkovDecisionProcess(
    states=states,
    actions=actions,
    rewards=rewards,
    state_transition_probability=state_transition_prob,
    policy=policy,
    discount=0.99
  )
