"""
.. module:: extract_features
   :platform: Unix, Windows
   :synopsis: Contains methods to extract spectral bandwise features from an audio track.

.. moduleauthor: Dimitris Geromichalos <geromidg@gmail.com>
"""

from os.path import exists
from copy import copy
import numpy as np

from pathos.multiprocessing import ProcessingPool as Pool

from mir3.modules.tool import wav2spectrogram
from mir3.modules.features import flatness, energy, flux, centroid, rolloff, low_energy, mfcc, join, stats
from mir3.modules.tool import to_texture_window
from mir3.data.feature_track import FeatureTrack

class ExtractFeatures:
    """
    A class for processing a dataset and extracting audio features.

    It holds the input dataset and contains methods for processing it,
    by interfacing with the mir3 package of the pymi3 library.

    Args:
        dataset (List[str]): The input dataset from which the featues will be extracted.
        num_bands (Optional[int]): The number of spectral bands to operate on. Defaults to 50.
        num_processes (Optional[int]): The number of processes to be spawned. Defaults to 8.
        enable_load (Optional[bool]): Whether to load serialized features. Defaults to True.
        enable_save (Optional[bool]): Whether to serialized features after extracting them. Defaults to True.

    Attributes:
        dataset (List[str]): The input dataset from which the featues will be extracted.
        num_bands (int): The number of spectral bands to operate on.
        num_processes (int): The number of processes to be spawned.
        enable_load (bool): Whether to load serialized features.
        enable_save (bool): Whether to serialized features after extracting them.
    """

    def __init__(self, dataset, num_bands=50, num_processes=8, enable_load=True, enable_save=True):
        self.dataset = dataset
        self.num_bands = num_bands
        self.num_processes = num_processes

        self.enable_load = enable_load
        self.enable_save = enable_save

    def run(self):
        """
        Runs the feature extraction on the input dataset.

        The extraction consists of a 3-stage pipeline. The first 2 steps are computed in parallel.
        First, the bandwise features are extracted from all the audio tracks of the dataset.
        Then, all the features are converted to texture windows that contain statistics of groups frames.
        Finally, a feature matrix that holds all the texture windows and their features is created.
        """

        print 'Running feature extraction...'

        print 'Extracting feature tracks...'
        feature_tracks = Pool(nodes=self.num_processes).map(self.extract_feature_track, self.dataset)

        print 'Extracting texture windows...'
        texture_windows = Pool(nodes=self.num_processes).map(self.extract_texture_window, feature_tracks)

        print 'Extracting the feature matrix...'
        feature_matrix = stats.Stats().stats(texture_windows, mean=True, variance=True, normalize=True)

        return feature_matrix

    def extract_feature_track(self, filename):
        """
        Extracts a feature track from an audio file.

        Args:
            filename (str): The path of the audio file.

        Returns:
            FeatureTrack: The extracted feature track.
        """

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
        """
        Extract a texture window from a feature track.

        Args:
            feature_track (FeatureTrack): The input feature track.

        Returns:
            TextureWindow: The extracted texture window.
        """
        
        texture_window = to_texture_window.ToTextureWindow().to_texture(feature_track, 100)

        return texture_window

    @staticmethod
    def get_mel_bands(spectrogram, num_bands):
        """
        Splits a spectrogram to a number of bands.

        Args:
            spectrogram (Spectrogram): The input spectrogram.
            num_bands (int): The number of the bands to be split.

        Returns:
            List[Tuple(int, int)]: The frequencies of the split bands.
        """
        
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
        """
        Calculates the features for every band of a spectrogram.

        Args:
            spectrogram (Spectrogram): The input spectrogram.
            bands (List[Tuple(int, int)]): The split bands of the spectrogram.

        Returns:
            NumPy Array: An array containing all the frames and their features.
        """
        
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
        """
        Joins all the features from all the bands to a feature track.

        Args:
            band_features (NumPy Array): An array containing all the frames and their features.

        Returns:
            FeatureTrack: The feature track containing the joined features for each frame.
        """
        
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
        """
        Extract the bandwise featues of an audio track.

        First, the audio track (wav file) is converted to a spectrogram.
        Then, from the spectrogram the features for each band are calculated.
        Finally, the features are nicely joined into a feature track.

        Args:
            filename (str): The input audio track.
            num_bands (int): The number of bands to split the spectrogram.

        Returns:
            FeatureTrack: The feature track containing the joined features for each frame.
        """
        
        wav_file = open(filename, 'rb')
        spectrogram = wav2spectrogram.Wav2Spectrogram().convert(wav_file, \
            dft_length=2048, window_step=1024, window_length=2048)
        spectrogram.data = 20 * np.log10(spectrogram.data + np.finfo(np.float).eps)  # spec to db
        wav_file.close()

        band_features = cls.calculate_features_per_band(spectrogram, cls.get_mel_bands(spectrogram, num_bands))
        joined_features = cls.join_bands_features(band_features)

        return joined_features