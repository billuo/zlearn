import subprocess as subproc
from itertools import product
import os.path as path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_param_grid(params):
    if not any(map(lambda x: type(x) == list, params.values())):
        return [params]
    param_new = {}
    param_fixed = {}

    for key, value in params.items():
        if isinstance(value, list):
            param_new[key] = value
        else:
            param_fixed[key] = value

    items = sorted(param_new.items())
    keys, values = zip(*items)

    params_exp = []
    for v in product(*values):
        param_exp = dict(zip(keys, v))
        param_exp.update(param_fixed)
        params_exp.append(param_exp)

    return params_exp


def train(param, summary_filename):
    cmd_args = [param['exe'],
                '--threads', '4',
                param['task'],
                'train', param['model'],
                '--input', param['input'],
                '--no-output',
                '--summary', summary_filename,
                '--split', '4:1',
                '--opt', param['opt'],
                '--metric', param['metric'],
                '-n', str(param['epochs']),
                '--window', str(param['window']),
                '-k', str(param['k']),
                '-m', '3',
                '-r', str(param['r']),
                '--lr', str(param['lambda']),
                '--alpha', str(param['alpha']),
                '--gamma', str(param['gamma']),
                '--beta1', str(param['beta1']),
                '--beta2', str(param['beta2']),
                ]
    print(' '.join(cmd_args))
    subproc.call(cmd_args, bufsize=0)
    # with open('tuner.log', 'a') as log_file:
    #     subproc.call(cmd_args, stdout=log_file, shell=True)


def best_loss(summ):
    return summ.iloc[summ.is_best.first_valid_index()][1]


def best_metric(summ):
    return summ.iloc[summ.is_best.first_valid_index()][2]


opts = ['sgd', 'adagrad', 'rmsprop', 'adam', 'adam-unbias', 'amsgrad', 'momentum']
models = ['LM', 'FM', 'FFM', 'HOFM']
# models = ['HOFM']

data_dir = '/home/lz1008/Homework/GD/PyProject/data/ml-100k/'
input_file = path.join(data_dir, 'my.ffm')
task = '--binary'
metric = 'auc'
epochs = 50
window = 3
for model in models:
    results = {}
    for opt in opts:
        print('{} with {}:'.format(model, opt))
        fixed_param = {
            'exe': './cmake-build-release/src/zlearn',
            'input': input_file,
            'task': task,
            'model': model,
            'opt': opt,
            'metric': metric,
            'epochs': epochs,
            'window': window,
        }
        tune_param = {
            'lambda': list(np.geomspace(0.00002, 0.002, num=5)),
            'r': list(np.geomspace(0.00002, 0.2, num=9)),
            'k': 4,
            'alpha': 0.9,
            'gamma': 0.9,
            'beta1': 0.9,
            'beta2': 0.99,
        }
        if model == 'FM' or model == 'HOFM':
            tune_param['k'] = list(np.logspace(2, 6, num=5, base=2).astype(int))
        elif model == 'FFM':
            tune_param['k'] = list(np.logspace(2, 4, num=3, base=2).astype(int))
        if opt == 'rmsprop':
            tune_param['r'] = list(np.geomspace(0.000001, 0.0001, num=5))
        #     tune_param['alpha'] = [0.9, 0.99]
        # elif opt == 'momentum':
        #     tune_param['gamma'] = [0.9, 0.99]
        # elif opt == 'adam' or opt == 'adam-unbias' or opt == 'amsgrad':
        #     tune_param['beta1'] = [0.9, 0.99]
        #     tune_param['beta2'] = [0.99, 0.999]

        summaries = []
        grid = generate_param_grid(tune_param)
        for i in range(len(grid)):
            p = grid[i]
            print('{}/{}; {}'.format(i + 1, len(grid), p))
            # filename = '{}:{}:{}.summary'.format(model, opt, metric)
            filename='summary'
            summary_file = path.join(data_dir, filename)
            param = {**fixed_param, **p}
            train(param, summary_file)
            summary = pd.read_csv(summary_file)
            summaries.append((param, summary))

        best_param, best_summary = max(summaries, key=lambda k: best_metric(k[1]))
        results[opt] = (best_param, best_summary)

    df = pd.DataFrame(index=np.arange(epochs))
    with open(path.join(data_dir, '{}.txt'.format(model)), 'w') as file:
        for opt in opts:
            best_param, best_summary = results[opt]
            file.write('best parameters for {}:\n{}\n'.format(opt, best_param))
            df[opt] = best_summary['tt_metric']
    ax = df.plot()
    plt.grid(True)
    ax.set_xticks(df.index[0::5])
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric.upper())
    plt.savefig(path.join(data_dir, '{}.png'.format(model)), dpi=600)
    plt.cla()
