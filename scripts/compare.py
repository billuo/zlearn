import xlearn as xl

from subprocess import call
import os.path as path


def run_xlearn():
    if MODEL == 'LM':
        model = xl.create_linear()
    elif MODEL == 'FM':
        model = xl.create_fm()
    else:
        assert MODEL == 'FFM'
        model = xl.create_ffm()
    model.setTrain(TRAIN)
    model.setValidate(TEST)
    if WINDOW == 0:
        model.disableEarlyStop()
    param = {
        'task': TASK,
        'epoch': EPOCH,
        'opt': OPT,
        'metric': METRIC,
        'k': K,
        'lr': LEARNING_RATE,
        'lambda': LAMBDA,
    }
    model.fit(param, './xlearn.model')


def run_zlearn():
    cmd_args = ['./cmake-build-release/src/zlearn',
                '--threads', '4',
                TASK == 'binary' and '--binary' or '--regression',
                'train', MODEL,
                '--opt', OPT,
                '--metric', METRIC,
                '-n', str(EPOCH),
                '--window', str(WINDOW),
                '-k', str(K),
                '-m', str(M),
                '-r', str(LEARNING_RATE),
                '--lr', str(LAMBDA),
                '--alpha', str(ALPHA),
                '--gamma', str(GAMMA),
                '--beta1', str(BETA1),
                '--beta2', str(BETA2),
                '--input', TRAIN,
                '--test', TEST,
                ]
    print('command to run:')
    print(' '.join(cmd_args))
    call(cmd_args)


if __name__ == '__main__':

    TASK = 'reg'
    MODEL = 'HOFM'
    M = 3
    K = 64
    # OPT = 'sgd'
    # OPT = 'adagrad'
    OPT = 'amsgrad'
    # OPT = 'momentum'
    # OPT = 'rmsprop'
    # OPT = 'adam'
    # OPT = 'adam-unbias'
    WINDOW = 5
    EPOCH = 100
    LEARNING_RATE = 0.001
    LAMBDA = 0.00002
    ALPHA = 0.99
    GAMMA = 0.9
    BETA1 = 0.9
    BETA2 = 0.99

    if TASK == 'reg':
        data_dir = '/home/lz1008/Homework/GD/PyProject/data/ml-100k/'
        TRAIN = path.join(data_dir, 'train.ffm')
        TEST = path.join(data_dir, 'test.ffm')
        METRIC = 'rmsd'
    else:
        assert TASK == 'binary'
        TRAIN = '/home/lz1008/Homework/GD/PyProject/data/dac_sample/train.ffm'
        TEST = '/home/lz1008/Homework/GD/PyProject/data/dac_sample/test.ffm'
        METRIC = 'auc'

    run_zlearn()
    # run_xlearn()
