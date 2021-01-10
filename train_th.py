# Training MNIST WANN with PyTorch
import json
import os
import subprocess
import sys
import torch 
import config
import numpy as np
import ann_th as ann
from model_th import make_model, simulate
from es_th import SGD, Adam
import argparse
import time

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=160)

final_mode = True
render_mode = False

#final_mode = False; render_mode = True # VIEW: toggle with comment to view trials

RENDER_DELAY = False
record_video = False
MEAN_MODE = False

record_rgb = False

if record_rgb:
      import imageio

device = 'cpu'


class OptimizerInfo(object):
  def __init__(self, num_params, mu=0.):
    self.num_params = num_params
    self.mu = mu


def get_act_with_params(params, model, x):
  assert(len(params) == len(model.wVec))
  annOut = ann.act(params, model.aVec, model.input_size, model.output_size, x, device=device)
  action = ann.selectAct(annOut, model.action_select, device)
  return action


def cross_entropy_loss(params, model, X, y):
  m = y.shape[0]
  action = get_act_with_params(params, model, X)
  log_likelihood = -torch.log(action.gather(1, y.view(-1,1).long()))
  loss = torch.sum(log_likelihood) / m
  return loss


def accuracy_batch(params, model, X, y):
  with torch.no_grad():
    m = y.shape[0]
    action = get_act_with_params(params, model, X)
    p = torch.argmax(action, axis=1)
    acc = torch.sum(p==y) / len(y)
    return acc.item()


def accuracy(params, model, X_data, y_data):
  m = y_data.shape[0]
  correct_tot = 0
  batch_size = 1000
  batch_first = -batch_size
  batch_last = 0
  train_order = np.arange(m)

  while True:
    ## Update batch index
    batch_first += batch_size
    batch_last += batch_size
    if batch_last > m:
      batch_last = m

    batch_idx = train_order[batch_first:batch_last]
    X = X_data[batch_idx,:]
    y = y_data[batch_idx]

    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    correct_tot += len(batch_idx) * accuracy_batch(params, model, X, y)

    if batch_last == m:
      break

  acc = correct_tot / m
  return acc


def make_optimizer(config):
  optimizer = None
  pi = config['pi']
  learning_rate = config['learning_rate']
  momentum = config['momentum']
  beta1 = config['beta1']
  beta2 = config['beta2']

  if config['name'] == 'sgd':
    optimizer = SGD(pi, learning_rate, momentum)
  elif config['name'] == 'adam':
    optimizer = Adam(pi, learning_rate, beta1, beta2)

  return optimizer


## params is parameter vectors with same size as model.wVec
## smallp is the compressed version, only taking nonzero entries in params
def encode_params(params, wKey):
  ## wKey: list of idx of nonzero entries in model.wVec
  p = params[wKey]
  return p


def train(model, myseed, opt_config, expid, batch_size, epochs, initializer):
  np.random.seed(myseed)

  X_train = model.env.trainSet[:50000,:]
  y_train = model.env.target[:50000]
  X_valid = model.env.trainSet[50000:,:]
  y_valid = model.env.target[50000:]

  pi = OptimizerInfo(len(model.wVec))
  opt_config['pi'] = pi
  optimizer = make_optimizer(opt_config)
  optimizer.to(device)
  del opt_config['pi']  # No need to save it

  savemap = {
    'opt_config': opt_config,
    'epochs': epochs,
    'batch_size': batch_size,
    'params_len': model.wVec.shape[0],
    'idx_nz': model.wKey.tolist(),
    'seed': myseed,
    'initializer': initializer,
  }
  savename = f"sgd_zoo/mnist.wann.{opt_config['name']}.{expid}.{myseed}.json"
  print('Data will be saved to: ', savename) 

  ## Initialize params
  initp = np.random.uniform(-1, 1, size=model.param_count)
  if initializer == 'he_uniform':
    initp *= np.sqrt(6/model.input_size)
  elif initializer == 'glorot_uniform':
    initp *= np.sqrt(6/(model.input_size+model.output_size))
  elif initializer == 'lecun_uniform':
    initp *= np.sqrt(3/model.input_size)

  params = np.zeros(len(model.wVec))
  params[model.wKey] = initp
  params = torch.tensor(params, device=device, requires_grad=True, dtype=torch.float)

  train_acc_list = []
  valid_acc_list = []

  valid_acc = accuracy(params, model, X_valid, y_valid)
  train_acc = accuracy(params, model, X_train, y_train)
  print('***** After init *****')
  print('Valid Acc =', valid_acc)
  print('Train Acc =', train_acc)
  print()

  start_time = time.time()
  t = 0

  for epoch in range(epochs):
    print('***** Epoch', epoch, '*****')
    train_order = np.random.permutation(len(y_train))
    batch_first = -batch_size
    batch_last = 0
    while True:
      t += 1
      if t % 50 == 0:
        print('-- Iter', t)

      ## Update batch index
      batch_first += batch_size
      batch_last += batch_size
      if batch_last > len(y_train):
        batch_last = len(y_train)
      optimizer.t = t  # Hack for Adam to update t without calling update()

      ## Create batch
      batch_idx = train_order[batch_first:batch_last]
      X = X_train[batch_idx,:]
      y = y_train[batch_idx]
      X = torch.tensor(X).to(device)
      y = torch.tensor(y).to(device)

      loss = cross_entropy_loss(params, model, X, y)
      loss.backward()

      ## Update params
      with torch.no_grad():
        mygrad = params.grad
        params_update = optimizer._compute_step(mygrad)
        params += params_update
        newp = torch.zeros(len(model.wVec), requires_grad=True).to(device)
        newp[model.wKey] = params[model.wKey]
        params = newp   # This line zero'ed grad
      params.requires_grad = True

      if batch_last == len(y_train):  # Last batch in epoch
        print('Epoch done at iter', t)
        break

    ## Compute stats
    valid_acc = accuracy(params, model, X_valid, y_valid)
    train_acc = accuracy(params, model, X_train, y_train)
    print('Valid Acc =', valid_acc)
    print('Train Acc =', train_acc)

    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    smallp = encode_params(params, model.wKey).detach().cpu().numpy()

    savemap['params_nz_e'+str(epoch)] = smallp.tolist()
    savemap['train_acc_list'] = train_acc_list
    savemap['valid_acc_list'] = valid_acc_list

    with open(savename, 'w') as f:
      json.dump(savemap, f, indent=4)
      print('Saved in json')
    print()

  print('Data saved in: ', savename)
  print('Took {} seconds'.format(time.time()-start_time))


def main():

  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
  parser.add_argument('gamename', type=str, help='robo_pendulum, robo_ant, robo_humanoid, etc.')
  parser.add_argument('-f', '--filename', type=str, help='json filename', default='none')
  parser.add_argument('-e', '--eval_steps', type=int, default=100, help='evaluate this number of step if final_mode')
  parser.add_argument('-s', '--seed_start', type=int, default=1, help='initial seed')
  parser.add_argument('-w', '--single_weight', type=float, default=-100, help='single weight parameter')
  parser.add_argument('--stdev', type=float, default=2.0, help='standard deviation for weights')
  #parser.add_argument('--sweep', type=int, default=-1, help='sweep a set of weights from -2.0 to 2.0 sweep times.')
  #parser.add_argument('--lo', type=float, default=-2.0, help='slow side of sweep.')
  #parser.add_argument('--hi', type=float, default=2.0, help='high side of sweep.')

  ## Additional args
  parser.add_argument('--init', type=str, default='he_uniform', help='initializer.')
  parser.add_argument('--expid', type=str, default='exp', help='experiment id.')
  parser.add_argument('--opt', type=str, default='none', help='optimizer name.')
  parser.add_argument('--lr', type=float, default=0, help='learning rate.')
  parser.add_argument('--mm', type=float, default=0.9, help='momentum for SGD.')
  parser.add_argument('--b1', type=float, default=0.99, help='beta1 for Adam.')
  parser.add_argument('--b2', type=float, default=0.999, help='beta2 for Adam.')
  parser.add_argument('--batch', type=int, default=128, help='batch size.')
  parser.add_argument('--device', type=str, default='cpu', help='use gpu or not.')


  args = parser.parse_args()

  assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

  gamename = args.gamename

  use_model = False

  game = config.games[gamename]

  filename = args.filename
  if filename != "none":
    use_model = True
    print("filename", filename)

  the_seed = args.seed_start

  model = make_model(game)
  print('model size', model.param_count)

  eval_steps = args.eval_steps
  single_weight = args.single_weight
  weight_stdev = args.stdev
  #num_sweep = args.sweep
  #sweep_lo = args.lo
  #sweep_hi = args.hi

  model.make_env(render_mode=render_mode)

  if use_model:
    model.load_model(filename)
  else:
    if single_weight > -100:
      params = model.get_single_model_params(weight=single_weight-game.weight_bias) # REMEMBER TO UNBIAS
      print("single weight value set to", single_weight)
    else:
      params = model.get_uniform_random_model_params(stdev=weight_stdev)-game.weight_bias
    model.set_model_params(params)

  if args.opt == 'none':
    args.opt = 'adam'

  if args.lr <= 0:
    if args.opt == 'adam':
      args.lr = 0.01
    elif args.opt == 'sgd':
      args.lr = 2.0
  
  print(f'Using {args.opt} as optimizer')

  opt_config = {
    'name': args.opt,
    'learning_rate': args.lr,
    'momentum': args.mm,
    'beta1': args.b1,
    'beta2': args.b2,
  }

  if args.device == 'gpu':
      global device
      device = 'cuda:0'

  train(model, the_seed, opt_config, args.expid, args.batch, args.eval_steps, args.init)


main()

