import os
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import config
from utils import extract_features

# set seed for reproducibility
np.random.seed(0)

label2int = {
    "male": 1,
    "female": 0
}



def filter_dataset(datapath):
    filtered_path = datapath[:-4] + '_filtered.csv'
    if(os.path.isfile(filtered_path)):
        return pd.read_csv(filtered_path)
    df = pd.read_csv(datapath)
    df = df[(df.gender == 'male') | (df.gender == 'female')]
    df = df.reset_index()
    for i in tqdm.tqdm(range(df.shape[0]), "Filtering data"):
        df.at[i, 'duration'] = librosa.get_duration(filename='../dataset/' + df.filename[i])
    df.to_csv(filtered_path)
    return df


def load_data(data_path=config.DATA_PATH, vector_length=187):
    """A function to load gender recognition dataset from `dataset` folder
    After the second run, this will load from results/features.npy and results/labels.npy files
    as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them from there instead
    if os.path.isfile("results/features.npy") and os.path.isfile("results/labels.npy"):
        X = np.load("results/features.npy")
        y = np.load("results/labels.npy")
        return X, y
    # read filtered dataframe
    df = filter_dataset(data_path)
    # get total samples
    n_samples = len(df)
    # get total male samples
    n_male_samples = len(df[df['gender'] == 'male'])
    # get total female samples
    n_female_samples = len(df[df['gender'] == 'female'])
    print("Total samples:", n_samples)
    print("Total male samples:", n_male_samples)
    print("Total female samples:", n_female_samples)
    # initialize an empty array for all audio features
    X = np.zeros((n_samples, vector_length))
    # initialize an empty array for all audio labels (1 for male and 0 for female)
    y = np.zeros((n_samples, 1))
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data",
                                           total=n_samples):
        filename = '../dataset/' + filename
        X[i] = extract_features(
            filename, mel=True, mfcc=True, chroma=True,
            contrast=True)  # mfcc=True, chroma=True, contrast=True,  tonnetz=True
        y[i] = label2int[gender]
    # save the audio features and labels into files
    # so we won't load each one of them next run
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y



def load_test_data(data_path=config.TEST_DATA_PATH, vector_length=187):
    """A function to load test gender recognition dataset from `dataset` folder
    After the second run, this will load from results/features_test.npy and results/labels_test.npy files
    as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them from there instead
    if os.path.isfile("results/features_test.npy") and os.path.isfile("results/labels_test.npy"):
        X = np.load("results/features_test.npy")
        y = np.load("results/labels_test.npy")
        return X, y
    # read filtered dataframe
    df = filter_dataset(data_path)
    # get total samples
    n_samples = len(df)
    # get total male samples
    n_male_samples = len(df[df['gender'] == 'male'])
    # get total female samples
    n_female_samples = len(df[df['gender'] == 'female'])
    print("Total samples:", n_samples)
    print("Total male samples:", n_male_samples)
    print("Total female samples:", n_female_samples)
    # initialize an empty array for all audio features
    X = np.zeros((n_samples, vector_length))
    # initialize an empty array for all audio labels (1 for male and 0 for female)
    y = np.zeros((n_samples, 1))
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data",
                                           total=n_samples):
        filename = '../dataset/' + filename
        X[i] = extract_features(
            filename, mel=True, mfcc=True, chroma=True,
            contrast=True)  # mfcc=True, chroma=True, contrast=True,  tonnetz=True
        y[i] = label2int[gender]
    # save the audio features and labels into files
    # so we won't load each one of them next run
    np.save("results/features_test", X)
    np.save("results/labels_test", y)
    return X, y


def balance_dataset(X, y):
    # get min length
    n = min([(y == l).sum() for l in np.unique(y)])
    # select randomly
    mask = np.hstack([np.random.choice(np.where(y == l)[0], n, replace=False)
                      for l in np.unique(y)])
    # resample the arrays
    return X[mask], y[mask]


def split_dataset(X, y, test_size=0.1):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=0)
    print('Training set samples shape:', X_train.shape)
    print('Validation set samples shape:', X_val.shape)
    return X_train, X_val, y_train, y_val


def get_dataloader(X, y, dtype='train'):
    # create your datset
    dataset = TensorDataset(torch.Tensor(
        X), torch.Tensor(y))
    # create your dataloader
    if dtype == 'train':
        dataloader = DataLoader(
            dataset, batch_size=config.TRAIN_BATCH_SIZE)
    elif dtype == 'val':
        dataloader = DataLoader(
            dataset, batch_size=config.VALID_BATCH_SIZE)
    return dataloader


def normalize(X):
    if os.path.isfile(config.SCALAR_PATH):
        scaler = pickle.load(open(config.SCALAR_PATH, 'rb'))
        return scaler.transform(X)
    
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    pickle.dump(scaler, open(config.SCALAR_PATH))
    return Xn
