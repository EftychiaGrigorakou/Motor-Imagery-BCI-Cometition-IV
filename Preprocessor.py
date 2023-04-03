# -*- coding: utf-8 -*-

import numpy as np
import mne
from asrpy import ASR
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

    
class Preprocessor:
    """ Contains the tools to perform preprocessing to the data
    
    Args:
        raw_dict (dictionairy): keys: Subjects
        subjects = list of integers defining the subjects to analyse 
    """

    def __init__(self, raw_dict, subjects):
        self.subjects = subjects
        self.n_sessions = len(raw_dict[1].keys())
        self.n_channels = 22
        
    def freq_filtering(self, raw_array, low_f, high_f):
        
        """
        Args:
            raw_array (_type_): mne.RawArray
            low_f (int): low threshold of filter
            high_f (int): high threshold of filter
        """
    
        # the application of filters causes ringing artifacts in the signal,
        # meaning intense artifacts at the beginning and at the end of the signal.
        # therefore we first extend the signal
        extention_length = 250
        raw_array = self._extend(raw_array, extention_length)
    
        # actual filtering
        raw_array = raw_array.load_data().\
            filter(l_freq=low_f, h_freq=high_f, h_trans_bandwidth=1, l_trans_bandwidth=1, verbose=0)
    
        # removing extention
        raw_array = self._remove_extension(raw_array, extention_length)
        return raw_array

    def _extend(self, raw_array, length):
        
        """the application of filters causes ringing artifacts in the signal,
        meaning intense artifacts at the beginning and at the end of the signal.
        This function extends the signal by adding 250 points in the 2 edges.
        Then the extended signal will be filtered and using the remove_extension function,
        we will drop the extended parts, which contain the artifacts.
        :param
            raw_array: mne.RawArray
        :return:
            extended_raw: mne.RawArray
        """
    
        data = raw_array.get_data()
        prepended_signal = data[:, :length]
        appended_signal = data[:, -length:]
        extended_signal = np.concatenate((prepended_signal, data, appended_signal), axis=1)
    
        extended_raw = mne.io.RawArray(data=extended_signal, info=raw_array.info, verbose=0, copy="both")
        return extended_raw
    
    def _remove_extension(self, raw_array, length):
        """
        Removes the extension which contain the ringing artifacts.
        :param
            raw_array: mne.RawArray
        :return:
            extensions_removed_raw: mne.RawArray
        """
        n_ch = len(raw_array.info["ch_names"])
        no_extended_data = []
    
        for channel in range(raw_array['data'][0].shape[0]):
            no_extension_signal = np.array(raw_array["data"][0][channel, length:-length])
            no_extended_data.append(no_extension_signal)
        no_extended_data = np.reshape(np.array(no_extended_data), (n_ch, -1))
    
        extensions_removed_raw = mne.io.RawArray(data=no_extended_data, info=raw_array.info, verbose=0, copy="both")
        return extensions_removed_raw
    
    def asr_(self, raw_array):
        
        """The function that detects and removes ocular and muscle artifacts, using ASR method.
    
            Args:
                raw_array (mne.RawArray): mne object with data
    
            Returns:
                raw_art_removed (mne.RawArray): mne object with data without the artifacts
            """
            
        thr = 50 
            
        raw_data = raw_array["data"][0]
        s_freq = raw_array.info['sfreq']
        raw_art_removed = raw_array.copy()
    
        raw_art_removed["data"] = raw_data
        asr = ASR(sfreq = s_freq, cutoff = thr)
        asr.fit(raw_array)
        raw_art_removed = asr.transform(raw_array)
        
        return raw_art_removed    
    
    
    def process(self, trial):
        """
            Filters data (Bandpass filter) and plots the PSD before and after preprocessing
      
            :return:
                clean_data (np.array): array of size (n_datapoints, n_channels) containing the clean data 
            """


        # Transform data to mne RawArray 
        ch = ['Ch-0', 'Ch-1', 'Ch-2', 'Ch-3', 'Ch-4', 'Ch-5', 'Ch-6', 'Ch-7', 'Ch-8', 'Ch-9', 'Ch-10', 'Ch-11', 'Ch-12', 'Ch-13', 'Ch-14', 'Ch-15', 'Ch-16', 'Ch-17', 'Ch-18', 'Ch-19', 'Ch-20', 'Ch-21']
        mne_info = mne.create_info(ch_names = ch, sfreq = 250, ch_types=['eeg']*self.n_channels)
        #print(trial.shape)
        raw_data = mne.io.RawArray(data = trial, info = mne_info, verbose=0, copy="both")
        
        # ---Plot PSD of raw data---
        #psd = raw_data.plot_psd()
        
        # Filter data
        filtered_data = self.freq_filtering(raw_data, low_f=1, high_f=50)
        
        # ---Plot PSD of filtered data---
        #psd = filtered_data.plot_psd()
    
        # Get the clean data 
        clean_data = filtered_data["data"][0]
    
        return clean_data
    

    def get_clean(self, dictionairy):
        """
            Preprocess data of each subject and structure it in a dictionairy. 
      
            :return:
                clean_dict (dictionairy): keys: subjects, containing clean data.
            """
        
        #loop over subjects
        for i in tqdm(self.subjects):
            #loop over sessions
            for s in dictionairy[i].keys():
                raw = dictionairy[i][s]['Data']
                clean_data = []
                # loop over trials
                for j in range(raw.shape[0]):
                    trial = raw[j,:,:]
                    clean_trial = self.process(trial)
                    clean_data.append(clean_trial)
                clean_data = np.stack((clean_data), axis=0)
                dictionairy[i][s]['Data'] = clean_data
        
        clean_dict = dictionairy
        return clean_dict
    
    
def get_raw(paradigm, dataset, subjects):
    """
        Loads data from moabb library
        :return:
            data_dict (dictionairy): keys: Subjects, containing the raw data. 
        """
    
    data_dict = {}

    for i in tqdm(subjects):
        # split the data into subjects
        subject_data = {}
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[i], return_epochs=1)
        y = LabelEncoder().fit_transform(y)
    
        # split the data of each subject into 2 sessions 
        for session in np.unique(metadata.session):
            session_data = {}
            ix = metadata.session == session
    
            session_data['Data'] = X[ix].get_data()
            session_data['Label'] = y[ix]
    
            subject_data[session] = session_data
    
        data_dict[i] = subject_data
    return data_dict
