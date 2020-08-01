
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import pandas as pd
import torch
import tqdm
import config
import pickle
from torch.utils.data import TensorDataset, DataLoader
from utils import extract_features, extract_2d_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# set seed for reproducibility
np.random.seed(0)

label2int = {
    "male": 1,
    "female": 0
}

scaler = StandardScaler()


def plot_raw_audio(audio_file):
    # plot the raw audio signal
    raw_audio, _ = librosa.load(audio_file)
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    steps = len(raw_audio)
    ax.plot(np.linspace(1, steps, steps), raw_audio)
    plt.title('Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def load_data(data_path='../dataset/cv-valid-train_filtered.csv', vector_length=187):
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
    # read dataframe
    df = pd.read_csv(data_path)
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
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
        filename = '../dataset/' + filename
        X[i] = extract_features(
            filename, mel=True, mfcc=True, chroma=True, contrast=True)  # mfcc=True, chroma=True, contrast=True,  tonnetz=True
        y[i] = label2int[gender]
    # save the audio features and labels into files
    # so we won't load each one of them next run
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y


def load_data_2d(data_path='../dataset/cv-valid-train_filtered.csv'):
    """A function to load gender recognition dataset from `dataset` folder
    After the second run, this will load from results/features2d.pkl and results/labels.pkl files
    as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them from there instead
    if os.path.isfile("results/features2d.pkl") and os.path.isfile("results/labels2d.pkl"):
        X = pickle.load("results/features2d.pkl", allow_pickle=True)
        y = pickle.load("results/labels2d.pkl")
        return X, y
    # read dataframe
    df = pd.read_csv(data_path)
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
    X = [np.array([]) for _ in range(n_samples)]
    # initialize an empty array for all audio labels (1 for male and 0 for female)
    y = np.zeros((n_samples, 1))
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
        filename = '../dataset/' + filename
        X[i] = extract_2d_features(filename, mel=True)
        y[i] = label2int[gender]
    # save the audio features and labels into files
    # so we won't load each one of them next run
    np.save("results/features2d", X)
    np.save("results/labels2d", y)
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


def normalize(X, train=True):
    if train:
        Xn = scaler.fit_transform(X)
        pickle.dump(scaler, open('results/scaler.pkl','wb'))
        return Xn
    return scaler.transform(X)
