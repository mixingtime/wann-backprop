# Testing WANN
import json
import os
import subprocess
import sys
import config
import numpy as np
from model import make_model, simulate
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


def test(model, test_file, test_epoch):
  print('***** Begin testing *****')
  start_time = time.time()
  with open(test_file, 'r') as f:
    data = json.load(f)
    if test_epoch < 0:
      epochs = data['epochs']
    params_nz = data['params_nz_e'+str(epochs-1)]
    model.set_model_params(params_nz)
    reward, steps_taken = simulate(model, train_mode=False, render_mode=False, num_episode=1, seed=0)
    print(f"Test acc:", reward[0])

  print('Took {} seconds'.format(time.time()-start_time))


def main():

  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
  parser.add_argument('gamename', type=str, help='robo_pendulum, robo_ant, robo_humanoid, etc.')
  parser.add_argument('-f', '--filename', type=str, help='json filename', default='none')
  parser.add_argument('--test_file', type=str, default="", help='test file.')
  parser.add_argument('--test_epoch', type=int, default=-1, help='test epoch.')


  args = parser.parse_args()

  assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

  gamename = args.gamename

  use_model = False

  game = config.games[gamename]

  filename = args.filename
  if filename != "none":
    use_model = True
    print("filename", filename)

  model = make_model(game)
  print('model size', model.param_count)

  model.make_env(render_mode=render_mode)

  test(model, args.test_file, args.test_epoch)


main()

