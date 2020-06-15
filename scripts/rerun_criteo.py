import re
import os.path as path
import subprocess as subproc

data_dir = '/home/lz1008/Downloads/dac/'


def convert_csv():
    subproc.call(['./cmake-build-release/src/utility/csv2ffm',
                  path.join(data_dir, 'train-1m.txt'),
                  path.join(data_dir, 'train-1m.ffm'),
                  '--sep', 'tab',
                  '--encode', 'n13c26',
                  ])


def read_params(model):
    file = open(path.join(data_dir, '{}.txt'.format(model)), 'r')
    params = {}
    while True:
        line1 = file.readline()
        if len(line1) == 0: break
        line2 = file.readline()
        assert len(line2) > 0

        re1 = re.match('best parameters for ([a-z-]+)', line1)
        opt = re1.group(1)
        param = eval(line2)
        params[opt] = param
    return params


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


if __name__ == '__main__':
    # convert_csv()
    models = ['LM', 'FM', 'FFM', 'HOFM']
    for model in models:
        params = read_params(model)
        for opt in params:
            param = params[opt]
            param['input'] = path.join(data_dir, 'train-1m.ffm')
            train(param, 'summary')
