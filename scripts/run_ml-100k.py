import math
import pandas as pd

import os.path as path
from subprocess import call
from sklearn.metrics import mean_squared_error

data_dir = '/home/lz1008/Homework/GD/PyProject/data/ml-100k/'


def prepare_csv():
    rating = pd.read_csv(path.join(data_dir, 'u.data'), sep='\t',
                         names=['userId', 'itemId', 'rating', 'timestamp'])
    rating.drop('timestamp', axis=1, inplace=True)
    data = rating

    movie_ratings = rating.groupby('itemId').count()
    movie_ratings['n_movie_rating'] = movie_ratings['rating'].astype(int).apply(math.log2)
    movie_ratings.drop(['userId', 'rating'], axis=1, inplace=True)
    data = pd.merge(data, movie_ratings, on='itemId')

    user_ratings = rating.groupby('userId').count()
    user_ratings['n_user_rating'] = user_ratings['rating'].astype(int).apply(math.log2)
    user_ratings.drop(['itemId', 'rating'], axis=1, inplace=True)
    data = pd.merge(data, user_ratings, on='userId')

    users = pd.read_csv(path.join(data_dir, 'u.user'), sep='|',
                        names=['userId', 'age', 'gender', 'occupation', 'zip code'])
    users.drop('zip code', axis=1, inplace=True)
    data = pd.merge(data, users, on='userId')

    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movie_cols = ['itemId', 'title', 'releaseDate', 'videoReleaseDate', 'IMDb',
                  'unknown'] + genres
    movie = pd.read_csv(path.join(data_dir, 'u.item'), sep='|', names=movie_cols)
    movie.drop(['title', 'unknown', 'releaseDate', 'videoReleaseDate', 'IMDb'], axis=1,
               inplace=True)
    data = pd.merge(data, movie, on='itemId')

    for genre in genres:
        data[genre] = data[genre].astype(str)

    cols = list(data.columns)
    cols.pop(2)
    cols.insert(0, 'rating')
    data = data[cols]
    data['rating'] = data['rating'].apply(lambda r: r > 4 and 1 or 0)

    print(','.join(data.columns))

    output = open(path.join(data_dir, 'my.csv'), 'w')
    output.write(','.join(data.columns) + '\n')
    for tp in data.itertuples():
        output.write(','.join(str(x) for x in tp[1:]) + '\n')


def convert_csv():
    call(['./cmake-build-release/src/utility/csv2ffm',
          path.join(data_dir, 'my.csv'),
          path.join(data_dir, 'my.ffm'),
          '--header', '--per-column',
          '--encode', 'c2n2c3n18',
          '--group', '1;2;3;4;5;6;7;8-25',
          ])


def train():
    cmd_args = ['./cmake-build-release/src/zlearn',
                '--threads', '1',
                '--binary',
                'train', 'FFM',
                '--opt', 'momentum',
                '--metric', 'auc',
                '-n', '100',
                '--window', '4',
                '-k', '16',
                '-m', '5',
                '-r', '0.001',
                '--lr', '0.0002',
                # '--input', path.join(data_dir, 'my.ffm'),
                # '--split', '4:1',
                # '--dump-train', path.join(data_dir, 'train.ffm'),
                # '--dump-test', path.join(data_dir, 'test.ffm'),
                '--input', path.join(data_dir, 'train.ffm'),
                '--test', path.join(data_dir, 'test.ffm'),
                ]
    print(' '.join(cmd_args))
    call(cmd_args)


def predict():
    cmd_args = ['./cmake-build-release/src/zlearn',
                'predict',
                '--model', path.join(data_dir, 'train.ffm.bin'),
                '--input', path.join(data_dir, 'test.ffm'),
                '--output', path.join(data_dir, 'test.out'),
                ]
    print(' '.join(cmd_args))
    call(cmd_args)


def measure():
    truth = open(path.join(data_dir, 'test.ffm'))
    predicted = open(path.join(data_dir, 'test.out'))
    T = []
    P = []
    i = 0
    while True:
        t = truth.readline()
        p = predicted.readline()
        i += 1
        if len(t) == 0:
            break
        assert len(p) != 0
        t = t.split(' ')[0]
        T.append(float(t))
        P.append(float(p[:-1]))
    print(mean_squared_error(T, P))


if __name__ == '__main__':
    prepare_csv()
    convert_csv()
    # train()
    # predict()
    # measure()
