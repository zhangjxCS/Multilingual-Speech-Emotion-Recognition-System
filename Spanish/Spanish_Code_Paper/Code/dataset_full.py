from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
import torchaudio
import os


class spanish_dataset(Dataset):
    def __init__(self, dataset_location, data_splits, model_name):
        self.dataset_location = dataset_location
        self.data_splits = data_splits

        #initialize label mapping 
        self.emotion_mapper = {
            'Neutral': 'neutral',
            'Anger': 'angry', 
            'Happiness': 'happy', 
            'Sadness': 'sad', 
            'Fear': 'fear',
            'Disgust': 'disgust'}
        self.emotion_to_label = {
                            'neutral':0,
                            'angry':1,
                            'happy':2,
                            'sad':3,
                            'fear':4,
                            'disgust':5}
        self.label_to_emotion = {
                            0:'neutral',
                            1:'angry',
                            2:'happy',
                            3:'sad',
                            4:'fear',
                            5:'disgust'}

        #get all audiofile locations 
        self._get_data_locations()

        #load model
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        if model_name == 'WAV2VEC2_BASE':
            self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        elif model_name == 'WAV2VEC2_LARGE':
            self.bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        elif model_name == 'WAV2VEC2_BASE_XLSR':
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        elif model_name == 'WAV2VEC2_LARGE_XLSR':
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        elif model_name == 'HUBERT_BASE':
            self.bundle = torchaudio.pipelines.HUBERT_BASE
        elif model_name == 'HUBERT_LARGE':
            self.bundle = torchaudio.pipelines.HUBERT_LARGE
        elif model_name == 'WAVLM_BASE':
            self.bundle = torchaudio.pipelines.WAVLM_BASE
        elif model_name == 'WAVLM_LARGE':
            self.bundle = torchaudio.pipelines.WAVLM_LARGE

        self.model = self.bundle.get_model().to(self.device)

    def _get_data_locations(self):
        #gatherting all audio file locations
        self.all_data = []
        for split in self.data_splits:
            split_location = self.dataset_location + 'split_' + str(split) + '/'
            for audiofile in os.listdir(split_location):
                filename = split_location + audiofile 
                self.all_data.append(filename)

        random.shuffle(self.all_data)
        self.len = len(self.all_data)

    def __len__(self):
        return self.len
         

    def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        audiofile = self.all_data[index]
        emotion_id = audiofile.split('/')[-1].split("_")[0]
        emotion = self.emotion_mapper[emotion_id]
        emotion_label = self.emotion_to_label[emotion]

        # step 2: load audio and features
        wave, sr = torchaudio.load(audiofile)
        wave = wave.to(self.device)
        if sr != self.bundle.sample_rate:
            wave = torchaudio.functional.resample(wave, sr, self.bundle.sample_rate)

        with torch.inference_mode():
            features, _ = self.model.extract_features(wave)

        #concatenate all features for the 12 layers
        features_pt = torch.tensor([])
        for layer in range(len(features)):
            features_pt = torch.cat((features_pt, features[layer].detach().cpu()), dim = 0)
        seq_length = features_pt.shape[1]

        return features_pt, emotion_label, seq_length


def my_collate_function(data):

    features, labels, seq_lengths = zip(*data)
    batch_size = len(features)
    max_seq_sen = max(seq_lengths)

    #FEATURES HAS DIMENSIONS 12 * Seq_Len * feature_dim
    features_collated = torch.zeros((batch_size, features[0].shape[0], max_seq_sen, features[0].shape[2]))

    for i in range(batch_size):
        features_collated[i,:,:seq_lengths[i], :] = features[i]

    labels = torch.tensor(labels)
    seq_lengths = torch.tensor(seq_lengths)

    return features_collated, labels, seq_lengths


def initialize_data(dataset_location, train_splits, test_splits, model_name):
    training_set = spanish_dataset(dataset_location, train_splits, model_name)
    testing_set = spanish_dataset(dataset_location, test_splits, model_name)

    train_params = {'batch_size': 32,
              'shuffle': True,
              }

    test_params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 4
              }

    train_loader = DataLoader(training_set, **train_params, collate_fn = my_collate_function)
    test_loader = DataLoader(testing_set, **test_params, collate_fn = my_collate_function)

    return train_loader, test_loader
