import os.path as path
from subprocess import call
from sklearn.metrics import roc_auc_score

data_dir = '/home/lz1008/Homework/GD/PyProject/data/dac_sample/'


def convert_csv():
    call(['./cmake-build-release/src/utility/csv2ffm',
          path.join(data_dir, 'dac_sample.txt'),
          path.join(data_dir, 'dac_sample.ffm'),
          '--sep', 'tab',
          '--encode', 'n13c26',
          ])


def train():
    cmd_args = ['./cmake-build-release/src/zlearn',
                '--threads', '4',
                '--binary',
                'train', 'FM',
                '--opt', 'rmsprop',
                '--metric', 'auc',
                '-n', '100',
                '--window', '5',
                '-k', '8',
                '-r', '0.000002',
                '--lr', '0.0002',
                '--input', path.join(data_dir, 'dac_sample.ffm'),
                '--split', '4:1',
                # '--dump-train', path.join(data_dir, 'train.ffm'),
                # '--dump-test', path.join(data_dir, 'test.ffm'),
                # '--input', path.join(data_dir, 'train.ffm'),
                # '--test', path.join(data_dir, 'test.ffm'),
                ]
    call(cmd_args)


def predict():
    cmd_args = ['./cmake-build-release/src/zlearn',
                'predict',
                '--model', path.join(data_dir, 'dac_sample.ffm.bin'),
                '--input', path.join(data_dir, 'test.ffm'),
                '--output', path.join(data_dir, 'test.out'),
                ]
    call(cmd_args)


def measure():
    predicted = open(path.join(data_dir, 'test.out'))
    truth = open(path.join(data_dir, 'test.ffm'))
    P = []
    T = []
    i = 0
    while True:
        p = predicted.readline()
        t = truth.readline()
        i += 1
        if len(p) == 0:
            break
        assert len(t) != 0
        t = t.split(' ')[0]
        P.append(float(p[:-1]))
        T.append(float(t))
    print(roc_auc_score(T, P))


if __name__ == '__main__':
    # convert_csv()
    train()
    # predict()
    # measure()
