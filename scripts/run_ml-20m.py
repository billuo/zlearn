import pandas as pd

import os.path as path
from subprocess import call
from sklearn.metrics import mean_squared_error

data_dir = '/home/lz1008/Homework/GD/PyProject/data/ml-20m/'


def prepare_csv():
    movie = pd.read_csv(path.join(data_dir, 'movie.csv'))
    movie['movieId'] = movie['movieId'].astype(int)
    genre_names = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                   "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                   "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"]
    genres_map = {}
    for tp in movie.itertuples():
        genres_map[tp[1]] = tp[3]

    rating = pd.read_csv(path.join(data_dir, 'rating.csv'))
    rating['movieId'] = rating['movieId'].astype(int)

    output = open(path.join(data_dir, 'rating_g.csv'), 'w')
    output.write(','.join(["rating", "userId", "movieId"] + genre_names) + '\n')
    for tp in rating.itertuples():
        genres = genres_map[tp[2]].split('|')
        onehot = [genre in genres and '1' or '' for genre in genre_names]
        output.write('{},{},{},{}\n'.format(tp[3], tp[1], tp[2], ','.join(onehot)))


def convert_csv():
    call(['./cmake-build-release/src/utility/csv2ffm',
          path.join(data_dir, 'rating_g.csv'),
          path.join(data_dir, 'rating_g.ffm'),
          '--header',
          '--per-column',
          '--encode', 'c2n19',
          '--group', '1;2;3-21',
          ])


def train():
    cmd_args = ['./cmake-build-release/src/zlearn',
                '--threads', '4',
                '--regression',
                'train', 'FM',
                '--opt', 'sgd',
                '--metric', 'rmsd',
                '-n', '100',
                '--window', '5',
                '-k', '4',
                '-r', '0.002',
                '--lr', '0.00002',
                '--input', path.join(data_dir, 'rating_g.ffm'),
                '--split', '4:1',
                '--dump-train', path.join(data_dir, 'train.ffm'),
                '--dump-test', path.join(data_dir, 'test.ffm'),
                ]
    print(' '.join(cmd_args))
    # call(cmd_args)


def predict():
    cmd_args = ['./cmake-build-release/src/zlearn',
                'predict',
                '--model', path.join(data_dir, 'rating_g.ffm.bin'),
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
    print(mean_squared_error(T, P))


if __name__ == '__main__':
    # prepare_csv()
    # convert_csv()
    train()
    # predict()
    # measure()
