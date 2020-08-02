import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import pandas as pd
import torch
import tqdm
import config
import pickle
from torch.utils.data import TensorDataset, DataLoader, Dataset
from utils import extract_features
from torchaudio.transforms import MFCC, Resample, MelSpectrogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras_preprocessing.sequence import pad_sequences

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
        pickle.dump(scaler, open('results/scaler.pkl', 'wb'))
        return Xn
    return scaler.transform(X)


class VoiceInstance:
    def __init__(self, file, gender):

        self.file = file
        self.gender = gender
        self._transform_audio()

    def _transform_audio(self):
        waveform, rate = librosa.load(self.file)
        new_rate = rate/100
        resampled = Resample(rate, new_rate)(torch.Tensor(waveform))
        self.stft = self._get_stft(resampled, new_rate)
        self.mfcc = self._get_mfcc(resampled, new_rate)
        self.spectrogram = self._get_spectrogram(resampled, new_rate)

    def _get_spectrogram(self, arr, sample_rate=22000):

        spectrogram_tensor = MelSpectrogram(
            sample_rate, n_mels=config.WINDOW_SIZE)
        return spectrogram_tensor.forward(arr)

    def _get_mfcc(self, arr, sample_rate=22000):

        mfcc_tensor = MFCC(sample_rate, n_mfcc=config.WINDOW_SIZE)
        return mfcc_tensor.forward(arr)

    def _get_stft(self, waveform, rate):
        return torch.stft(waveform, int(rate))


def load_2d_data(data_path=config.DATA_PATH, vector_length=187):
    """A function to load gender recognition dataset from `dataset` folder
    After the second run, this will load from results/features_2d.pkl file
    as it is much faster!"""
    # make sure results folder exists
    if not os.path.isdir("results"):
        os.mkdir("results")
    # if features & labels already loaded individually and bundled, load them from there instead
    if os.path.isfile("results/features_2d.pkl"):
        # load dataset
        dataset = pickle.load(open('results/features_2d.pkl', 'rb'))
        # use mfcc features
        mfcc = [x.mfcc.reshape(-1, 128) for x in dataset]
        X = pad_sequences(mfcc, maxlen=config.MAX_LENGTH, padding='pre', value=0)
        y = np.array([label2int[x.gender] for x in dataset])
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
    dataset = []
    for i in tqdm.tqdm(df.index, "Loading data", total=len(df)):
        audio = VoiceInstance(os.path.join(
            '../dataset/', df.loc[i, 'filename']), df.loc[i, 'gender'])
        dataset.append(audio)
    pickle.dump(dataset, open("results/features_2d.pkl", "wb"))
    X = np.empty(len(dataset), object)
    y = np.zeros((n_samples, 1))
    for i in range(len(dataset)):
        X[i] = np.array(dataset[i].mfcc)
        y[i] = label2int[dataset[i].gender]
    return X, y
