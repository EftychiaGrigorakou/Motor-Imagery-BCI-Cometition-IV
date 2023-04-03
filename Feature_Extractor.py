# -*- coding: utf-8 -*-

import numpy as np
from pyriemann.tangentspace import TangentSpace
from tqdm import tqdm
 
class Feature_Extractor:
    """ Contains the tools to extract features from the data
    
    Args:
        data (dictionairy): keys: Subjects
    """
    
    def __init__(self, data):
        self.data = data
        
        
    def get_features(self):
        
        """
        Computes covariance matrix of each trial of each participant. Then extracts tangent space 
        features and combines all in a dictionairy. 
        :return:
            features_dict (dictionairy): keys: Subjects, containing the tangent features (n_trials, n_channels*(n_channels+1)/2, n_channels*(n_channels+1)/2 )
        """
        
        features_dict = {}
        #loop over subjects
        for i in tqdm(self.data.keys()):
            features_dict[i] = {}
            #loop over sessions
            for s in self.data[i].keys():
                features_dict[i][s] = {}
                eeg = self.data[i][s]['Data']
                
                cov_matrices = []
                # loop over trials
                for j in range(eeg.shape[0]):
                    trial = eeg[j,:,:]

                    # compute the covariance matrix
                    cov = np.cov(trial)
                    cov_matrices.append(cov)
                cov_matrices = np.stack((cov_matrices), axis=0)
                    
                # get tangent space features 
                f = self.tangent_space_features(cov_matrices)
                
                # create the final dictionairy
                features_dict[i][s]['Features'] = f
                
        return features_dict
                            
        

    def tangent_space_features(self, cov_matrices):
        """
        Extracts tangent space features using riemann geometry.
        
        Args: cov_matrices (np.array): (n_channels, n_channels)
        :return:
            tangent_space_features (np.array): keys: Subjects, containing the tangent features. 
                                        Size( n_channels*(n_channels+1)/2, n_channels*(n_channels+1)/2 )
        """
        
        # project the covariance matrix onto the tangent space
        ts = TangentSpace(metric='riemann')
        tangent_space_features = ts.fit_transform(cov_matrices)
        
        return tangent_space_features
        
    
