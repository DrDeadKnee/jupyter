import numpy as np


def highlow(fn):
    def wrapper(*args, **kwargs):
        if 'high' in kwargs and not kwargs['high']:
            key1, key2 = 'low_far', 'low_near'
        else:
            key1, key2 = 'high_far', 'high_near'

        return fn(*args, key1=key1, key2=key2)

    return wrapper


# computes limits from Lawton and Sylvester, technometrics, 1971
# returns three matrices of the form A = TV
# where V is the rank-2 eigenvecotr matrix, T is the amount, and A is the matrix of possible components
def compute_hard_limits(Y):

    # Factorize the covariance matrix
    C = np.transpose(np.asmatrix(Y)) * np.asmatrix(Y)
    val, vec = np.linalg.eigh(C)
    correct_vect = np.asarray(np.transpose(vec))
    if np.sum(correct_vect[-1]) < 0:
        correct_vect *= -1

    # Initialize the outputs
    V = [correct_vect[-1], correct_vect[-2]]
    Tslopes = {'high_near': None, 'low_near': None, 'high_far': None, 'low_far': None}
    T = {'high_near': None, 'low_near': None, 'high_far': None, 'low_far': None}
    A = {'high_near': None, 'low_near': None, 'high_far': None, 'low_far': None}

    xi_1 = [np.dot(V[0], i) for i in Y]
    xi_2 = [np.dot(V[1], i) for i in Y]

    # Get the four lines from the origin
    Tslopes['high_near'] = xi_2[np.argmax(xi_2)] / xi_1[np.argmax(xi_2)]
    Tslopes['low_near'] = xi_2[np.argmin(xi_2)] / xi_1[np.argmin(xi_2)]
    Tslopes['high_far'] = -min([np.abs(V[0][k] / V[1][k]) for k in range(len(V[1])) if V[1][k] >= 0])
    Tslopes['low_far'] = min([np.abs(V[0][k] / V[1][k]) for k in range(len(V[1])) if V[1][k] < 0])

    # Find the intersections of lines with unitary line, use to get A
    for key in Tslopes:
        T[key] = [
            1 / (np.sum(V[0]) + np.sum(V[1]) * Tslopes[key]),
            Tslopes[key] / (np.sum(V[0]) + np.sum(V[1]) * Tslopes[key])
        ]
        A[key] = T[key][0] * V[0] + T[key][1] * V[1]

    return A, T, V


def scaled_random_gaussian(vector, power=1):
    g = np.asarray([np.abs(np.random.normal()) for i in range(len(vector))])
    multiplier = np.mean(vector) * power / max(g)
    return g * multiplier


def add_noise(datamat, power=1):
    noisymat = []

    if isinstance(datamat, np.matrix):
        datamat = np.asarray(datamat.tolist())

    for i in datamat:
        g = scaled_random_gaussian(i, power=power)
        noisymat.append(i + g)
    return noisymat


def get_Cstar(c1, c2, b):
    return b * c1 + (1 - b) * c2


@highlow
def get_Cstar_matrix(A, betavec, **kwargs):
    matrix = []
    for b in betavec:
        matrix.append(get_Cstar(A[kwargs['key1']], A[kwargs['key2']], b))
    return matrix


def get_inf_ent(x, eps=10e-12):
    xmod = x / np.sum(x) + eps
    return -np.sum(xmod * np.log(xmod))


def get_autodiv(x, eps=10e-12):
    xmod = x + eps
    return -np.sum(xmod * np.log(xmod / np.mean(xmod)))


def get_entmax(A, entropies=None, betavec=None, highlow='high'):
    if betavec is None:
        betavec = np.arange(0, 1, 0.01)
    if entropies is None:
        entropies = gridsearch_entropies(betavec, A, highlow)

    beta = betavec[np.argmax(entropies)]
    if highlow == 'high':
        Cstar = get_Cstar(A['high_far'], A['high_near'], beta)
    else:
        Cstar = get_Cstar(A['low_far'], A['low_near'], beta)

    return Cstar


def get_entmin(A, entropies=None, betavec=None, highlow='high'):
    if betavec is None:
        betavec = np.arange(0, 1, 0.01)
    if entropies is None:
        entropies = gridsearch_entropies(betavec, A, highlow)

    beta = betavec[np.argmin(entropies)]
    if highlow == 'high':
        Cstar = get_Cstar(A['high_far'], A['high_near'], beta)
    else:
        Cstar = get_Cstar(A['low_far'], A['low_near'], beta)

    return Cstar


@highlow
def gridsearch_entropies(betavec, A, allpositive=False, **kwargs):
    entropies = []
    for b in betavec:
        Cstar = np.abs(get_Cstar(A[kwargs['key1']], A[kwargs['key2']], b))
        if allpositive:
            Cstar = np.abs(Cstar)
        H = get_inf_ent(Cstar)
        entropies.append(H)
    entropies = np.asarray(entropies)
    return entropies


@highlow
def gridsearch_autodivergences(betavec, A, **kwargs):
    autodivergences = []
    for b in betavec:
        Cstar = get_Cstar(A[kwargs['key2']], A[kwargs['key2']], b)
        autodivergences.append(get_autodiv(Cstar))
    return autodivergences
