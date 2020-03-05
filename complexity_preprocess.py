#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import loadmat

def create_inducedfields(root, subject_name, sf, scouts):
    kernel = load_kernel(root, subject_name)
    trials = glob.glob(os.path.join(root, subject_name, 'data_{}_*.mat'.format(sf)))
    vif_scout1, vif_scout2 = Ox_trial(trials[0], scouts, kernel)
    n_trials, n_timesamples = len(trials), len(vif_scout1)
    data = np.zeros((n_trials, n_timesamples, 2))
    data[0, :, :] = np.array([vif_scout1, vif_scout2]).T
    for q in range(1, n_trials):
        vif_scout1, vif_scout2 = Ox_trial(trials[q], scouts, kernel)
        data[q, :, :] = np.array([vif_scout1, vif_scout2]).T
    return data

def Ox_trial(trial, Ox, K):

    nch = 306
    d = loadmat(trial)
    X = d['F'][:nch, :]

    G1, G2 = K[Ox[0], :nch], K[Ox[1], :nch]
    y1, y2 = np.mean(np.matmul(G1, X), 0) * 10 ** 12, np.mean(np.matmul(G2, X), 0) * 10 ** 12
    vif1, vif2 = butter_bandpass_filter(y1, 10, 15, 1000,
    3), butter_bandpass_filter(y2, 10, 15, 1000, 3)
    return vif1, vif2

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def load_kernel(root, subject_name):
    kernel_file = glob.glob(os.path.join(root, subject_name, 'Kernel', '*.mat'))
    kernel_dict = loadmat(kernel_file[0])
    K = kernel_dict['ImagingKernel']
    return K

def load_scouts(fscouts):
    Scouts = loadmat(fscouts)
    nscouts = len(Scouts['Scouts'][0])
    V_left, V_right = [], []
    for iv in range(0, nscouts):
        v = Scouts['Scouts'][0][iv][0][0]
        name = Scouts['Scouts'][0][iv][3][0]
        if 'L' in name:
            V_left.extend(v)
        elif 'R' in name:
            V_right.extend(v)
        else:
            raise ValueError('Both L and R not in loaded scout names')
    return V_left, V_right

def delay_filt(x, k):
    z = [j - i for i, j in zip(x[:-k], x[k:])]
    return z
