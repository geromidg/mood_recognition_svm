from os.path import exists
from copy import copy
import numpy as np

from pathos.multiprocessing import ProcessingPool as Pool

from mir3.modules.tool import wav2spectrogram
from mir3.modules.features import flatness, energy, flux, centroid, rolloff, low_energy, mfcc, join, stats
from mir3.modules.tool import to_texture_window
from mir3.data.feature_track import FeatureTrack

class ExtractFeatures:
    def __init__(self, dataset, num_bands=50, num_processes=8, enable_load=True, enable_save=True):
        self.dataset = dataset
        self.num_bands = num_bands
        self.num_processes = num_processes

        self.enable_load = enable_load
        self.enable_save = enable_save

    def run(self):
        print 'Running feature extraction...'

        print 'Extracting feature tracks...'
        feature_tracks = Pool(nodes=self.num_processes).map(self.extract_feature_track, self.dataset)

        print 'Extracting texture windows...'
        texture_windows = Pool(nodes=self.num_processes).map(self.extract_texture_window, feature_tracks)

        print 'Extracting the feature matrix...'
        feature_matrix = stats.Stats().stats(texture_windows, mean=True, variance=True, normalize=True)

        return feature_matrix

    def extract_feature_track(self, filename):
        print 'Extracting feature track from %s...' % (filename)

        feature_track = None
        ft_filename = filename[:-4] + '.ft'

        if self.enable_load and exists(ft_filename):
            with open(ft_filename, 'rb') as handler:
                feature_track = FeatureTrack().load(handler)
        else:
            feature_track = self.extract_bandwise_features(filename, self.num_bands)
            
            if self.enable_save:
                with open(ft_filename, 'wb') as handler:
                    feature_track.save(handler, restore_state=True)

        return feature_track

    def extract_texture_window(self, feature_track):
        texture_window = to_texture_window.ToTextureWindow().to_texture(feature_track, 100)

        return texture_window

    @staticmethod
    def get_mel_bands(spectrogram, num_bands):
        def mel_to_hz(mel):
            return 700 * (10**(float(mel) / 2595) - 1)
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + (float(hz) / 700))

        high_hz = int(spectrogram.metadata.max_freq)
        high_mel = hz_to_mel(high_hz)

        length = int(high_mel / float(num_bands))

        bands_mel = [(i * length, (i * length) + (length - 1)) for i in range(num_bands -1)]    
        bands_mel.append(((i + 1) * length, high_mel))
        
        bands_hz = [(int(np.floor(mel_to_hz(band_mel[0]))), int(np.floor(mel_to_hz(band_mel[1])))) \
            for band_mel in bands_mel]

        return bands_hz

    @staticmethod
    def calculate_features_per_band(spectrogram, bands):
        band_features = np.array([])
        
        for band in bands:
            lowbin = spectrogram.freq_bin(band[0])
            highbin = spectrogram.freq_bin(band[1])

            if lowbin == 0:
                lowbin = 1

            features = []

            features.append(flatness.Flatness().calc_track_band(spectrogram, lowbin, highbin))
            features.append(energy.Energy().calc_track_band(spectrogram, lowbin, highbin))
            features.append(flux.Flux().calc_track_band(spectrogram, lowbin, highbin))
            features.append(centroid.Centroid().calc_track_band(spectrogram, lowbin, highbin))
            features.append(rolloff.Rolloff().calc_track_band(spectrogram, lowbin, highbin))
            features.append(low_energy.LowEnergy().calc_track_band(spectrogram, 10, lowbin, highbin))

            band_features = np.hstack((band_features, features))

        band_features = np.hstack((band_features, mfcc.Mfcc().calc_track(spectrogram)))

        return band_features

    @staticmethod
    def join_bands_features(band_features):
        min_dim = np.min([x.data.shape for x in band_features])
        features = []
        for feature in band_features:
            features.append(copy(feature))
            if features[-1].data.ndim > 1:
                features[-1].data = np.resize(features[-1].data, (min_dim[0], features[-1].data.shape[1]))
            else:
                features[-1].data = np.resize(features[-1].data, min_dim)

            if features[-1].data.ndim > 2:
                print "ERROR: unexpected number of dimensions."

        joined_features = join.Join().join(features)

        return joined_features

    @classmethod
    def extract_bandwise_features(cls, filename, num_bands):
        wav_file = open(filename, 'rb')
        spectrogram = wav2spectrogram.Wav2Spectrogram().convert(wav_file, \
            dft_length=2048, window_step=1024, window_length=2048)
        spectrogram.data = 20 * np.log10(spectrogram.data + np.finfo(np.float).eps)  # spec to db
        wav_file.close()

        band_features = cls.calculate_features_per_band(spectrogram, cls.get_mel_bands(spectrogram, num_bands))
        joined_features = cls.join_bands_features(band_features)

        return joined_features
