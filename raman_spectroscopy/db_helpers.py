import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_full_set(cnx, set_id, raw=False):
    cur = cnx.cursor()
    filenumbers = cur.execute(
        "SELECT DISTINCT(filenumber) FROM RawData WHERE set_id=:set_id",
        {'set_id': set_id}
    )
    plot_partial_set(cnx, set_id, [i[0] for i in filenumbers], raw)


def plot_partial_set(cnx, set_id, filenumbers, raw=False):
    sns.set_context('paper', font_scale=2)
    plt.figure(figsize=(8, 6))
    plt.title('Set {}'.format(set_id))
    plt.xlabel('wavenumber (1 / cm)')
    plt.ylabel('signal (a.u.)')

    for i in filenumbers:
        if raw:
            x, y = load_raw_data(cnx, set_id, i)
        else:
            x, y = load_aligned_data(cnx, set_id, i)
        plt.plot(x, y, label='file {}'.format(i))

    plt.legend(loc='best')
    plt.show()


def plot_spectral_matrix(
        Y, X=None, names=None, x_title='wavenumbers (1 / cm)', y_title='signal (a.u.)', as_series=False
):
    plt.clf()
    sns.set_context('paper', font_scale=2)
    plt.figure(figsize=(8, 6))
    plt.ylabel(y_title)
    plt.xlabel(x_title)

    if names is None:
        names = range(1, len(Y) + 1)
    if X is None:
        X = [np.arange(len(i)) for i in Y]

    if not as_series:
        for i in range(len(Y)):
            plt.plot(X[i], Y[i], label=names[i], lw=1.75)
    else:
        pallette = sns.light_palette('dark blue', input='xkcd', n_colors=len(Y) + 3)
        for i in range(len(Y)):
            plt.plot(X[i], Y[i] + i * 0.01, label=names[i], color=pallette[i + 3], lw=1.75)

    plt.legend(loc='best')
    plt.show()


def load_aligned_data(cnx, set_id, filenumber):
    sql = """
        SELECT signal, wavenumber
        FROM AlignData
        WHERE set_id={} AND filenumber={}
        """.format(
        set_id,
        filenumber
    )

    data = pandas.read_sql(sql, con=cnx)
    return data.wavenumber, data.signal


def load_raw_data(cnx, set_id, filenumber):
    sql = """
        SELECT signal, wavenumber
        FROM RawData
        WHERE set_id={} AND filenumber={}
        """.format(
        set_id,
        filenumber
    )

    data = pandas.read_sql(sql, con=cnx)
    return data.wavenumber, data.signal


def load_data_matrices(cnx, set_id, filenumbers=None, table=None):
    if table is None:
        table = "RawData"
    else:
        table is "AlignData"

    if filenumbers is None:
        cur = cnx.cursor()
        nums = cur.execute(
            "SELECT DISTINCT(filenumber) FROM RawData WHERE set_id=:set_id",
            {'set_id': set_id}
        )
        filenumbers = [i[0] for i in nums]

    # Assemble the data
    X, Y = [], []
    for i in filenumbers:
        if table == 'RawData':
            x, y = load_raw_data(cnx, set_id, i)
        elif table == 'AlignData':
            x, y = load_raw_data(cnx, set_id, i)
        X.append(x)
        Y.append(y / np.sum(y))

    return X, Y
