import numpy as np
from pandas import read_csv
import csv
from numpy import array
import os
from math import floor
from scipy import stats

# split a multivariate sequence into samples
from sklearn.preprocessing import MinMaxScaler


def split_sequences(sequences, n_steps_in, n_steps_out, remove_CP):
    X, y, cp = list(), list(), list()
    STEP= 10
    for i in range(0,len(sequences), STEP):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        if (not remove_CP or 1 not in sequences[i:out_end_ix, 0]):
            X.append(seq_x[:, 2:])
            y.append(seq_y[:, 2:])
            if 1 in seq_x[:, 0]:
                cp.append(1)
            elif 1 in seq_y[:,0]:
                cp.append(1)
            else:
                cp.append(0)
    return array(X), array(y), array(cp)

def load_EYE_dataset(prefix, mode):
    n_steps = 20
    batch_size = 256
    future = 1  # Number of steps to forecast
    # read input file
    data = read_csv(os.path.join(prefix,"EYEEEG","EEGEYE_features_raw.csv"), header=0)

    values = np.array(data.values)
    labels = np.array(values[:, 0])
    series = values

    # ensure all data is float
    series = series.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    series[:, 2:] = scaler.fit_transform(series[:, 2:])
    time = range(1, data.shape[0])
    split_time = floor(0.8 * data.shape[0])

    return series, split_time, n_steps, future, batch_size, labels

def shuffle(a, b=None, c=None):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    a = a[indices]
    if b != None:
        b = b[indices]
    if c != None:
        c = c[indices]
    return a, b, c

def norm_data(x,y):
    z= np.concatenate((x,y), axis=2)
    zmin = np.min(np.min(z, axis=2), axis=0)
    zmax = np.max(np.max(z, axis=2), axis=0)
    for i in range(0, x.shape[1]):
        x[:,i,:] = (x[:,i,:] - zmin[i])/(zmax[i] - zmin[i])
        y[:, i, :] = (y[:, i, :] - zmin[i]) / (zmax[i] - zmin[i])
    return x,y

def load_HAR_dataset(prefix, mode):
    data = read_csv(os.path.join(prefix,"HAR_" + mode + ".csv"), header=None)
    values = np.array(data.values)
    print(values.shape)
    #steps_in, steps_out, batch_size = 80, 1, 124
    return values, values.shape[1] - 2

def load_Aug_HAR(prefix, mode1):
    mode = mode1
    if mode1 == "validation":
        mode = "train"
    X = read_csv(os.path.join(prefix , "Aug_HAR_" + mode + "_x.csv"), header=None).values
    X = X.reshape(X.shape[0],9,int(X.shape[1]/9))


    Y = read_csv(os.path.join(prefix , "Aug_HAR_" + mode + "_y.csv"), header=None).values
    Y = Y.reshape(Y.shape[0], 9, int(Y.shape[1] / 9))
    X,Y = norm_data(X,Y)

    lbl = np.loadtxt(os.path.join(prefix, "Aug_HAR_"+ mode + "_lbl.txt"))


    if mode1 =="train":
        len = int(X.shape[0] * 0.9)
        X = X[:len, ]
        Y = Y[:len, ]
        lbl = lbl[:len, ]
    elif mode1 == "validation":
        len = int(X.shape[0] * 0.9)
        X = X[len:, ]
        Y = Y[len:, ]
        lbl = lbl[len:, ]
    #X,Y,lbl = shuffle(X, Y, lbl)
    #steps_in, steps_out, batch_size = 80, 1, 124
    print(X.shape, Y.shape, lbl.shape)
    return X,Y,lbl

def load_WISDM_dataset(path, mode, step_in=60, step_out=10, overlap=0.9, part=-1):
    path = os.path.join(path,"raw", "watch")
    files = [f for f in os.listdir(os.path.join(path)) if os.path.isfile(os.path.join(path, f))]
    # print(files)
    sample_x = []
    sample_y = []
    sample_c = []
    if mode == 'train':
        flist = range((part-1)*10, (part)*10)
    if mode == 'valid':
        flist =  range((part)*10, (part)*10+2)
    elif mode == 'test':
        flist = range(41, 50)

    step = floor((1 - overlap) * step_in)
    print(step)
    for f in flist:

        data = read_csv(os.path.join(path, files[f]), header=None).values
        #data = data[0:1000,:]
        #user = data[:, 0]
        clss = data[:, 1]
        #data = stats.zscore(data[:, 3:].astype('float32'), axis=0)
        sample = data[:,3:].astype('float32')
        data = (sample - np.min(sample, axis=0)) / (np.max(sample, axis=0) - np.min(sample, axis=0))
        # sampling
        for i in range(0, data.shape[0] - (step_in + step_out), step):
            if (np.unique(clss[i: i + step_in + step_out]).shape[0] == 1) or (mode=='test'):
                sample_x.append(data[i: i + step_in, :].T)
                sample_y.append(data[i + step_in: i + step_in + step_out, :].T)
                #sample_c.append(clss[i: i + step_in + step_out])
                lbl = np.unique(clss[i: i + step_in + step_out])
                C=lbl[0]
                for c in range(1,lbl.shape[0]):
                    C = C+lbl[c]
                sample_c.append(C)
    return np.array(sample_x), np.array(sample_y), np.array(sample_c)

#return series, split_time, n_steps, future, batch_size, labels
def load_dataset(path, dataset, mode, step_in=60, step_out=10):
    prefix = path
    if dataset == "HAR":
        return load_HAR_dataset(path, mode)
    elif dataset == "Aug_HAR":
        return load_Aug_HAR(path, mode)
    elif dataset == "EYE":
        return load_EYE_dataset(path, mode)
    elif dataset == "WISDM":
        return load_WISDM_dataset(path, mode, step_in, step_out)

def save_data(path, data, title):
    writer = csv.writer(open(os.path.join(path,title),'w'))
    writer.writerows(data)

def create_pairs(data, batch_size, n_steps_in, n_steps_out):
    X, label = [],[]

    return X, label