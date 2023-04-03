# -*- coding: utf-8 -*-

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np


class Classification :
    
    """ Classifies data employing different classifiers, evaluates the performance between the  sessions
    and finally plots the accuracy scores for each participant and each classifer. 
    
    Args:
        features_dict (dictionairy): keys: Subjects, containing the extracted features 
    """
    
    def __init__(self, features_dict):
        self.features_dict = features_dict
        
        
    
    def classify(self, clean_dict):
        """
        Performs classification, evaluation and plotting 
        Args: clean_dict (dictionairy): keys: Subjects
        Returns:
            acc_scores (dictionairy): keys: Classifiers, containing the accuracy score for each subject.
        """
        
        classifiers = {"LDA": LDA(solver="lsqr", shrinkage="auto"),
                       "SVM": SVC( kernel="rbf"),
                       "kNN": KNeighborsClassifier(n_neighbors = 8)}

        acc_scores = {clf_name: {subj: [] for subj in self.features_dict.keys()} for clf_name in classifiers}
       
        #loop over classifiers
        for clf_name, clf in classifiers.items():
            #loop over subjects
            for i in (self.features_dict.keys()):
                for key, value in self.features_dict[i].items():
                    # split data into training and test set
                    if key == "session_T":
                        X_train = self.features_dict[i][key]['Features']
                        y_train = clean_dict[i][key]['Label']
                    else:
                        X_test = self.features_dict[i][key]['Features']
                        y_test = clean_dict[i][key]['Label']
                        
                # train and test classifier 
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                # get accuracy score
                acc = accuracy_score(y_test, y_pred)
                acc_scores[clf_name][i].append(acc)
                
        # Plot the accuracy scores for each classifier and subject
        colors = {"LDA": 'red', "SVM": 'blue', "kNN": 'orange'}
        
        plt.figure(figsize=(8,5))
        #loop over classifiers
        for clf_name, clf in classifiers.items():
            #loop over subjects
            for i in (acc_scores[clf_name].keys()):
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.scatter(i, acc_scores[clf_name][i], label=f"{clf_name}" if clf_name not in labels else None, c = colors[clf_name], alpha = 0.8)
        plt.title('Accuracy scores'); plt.ylim(0.3,1)
        plt.legend(loc='lower right'); plt.grid()
        plt.xlabel('Subjects'); plt.ylabel('Accuracy')
        plt.show()
        
        return acc_scores
    
    def get_mean_acc(self, acc_scores):
        
        """
        Calculates the mean accuracy of each classifier among the subjects. 
        Args: 
            acc_scores (dictionairy): keys: classifiers.
        Returns:
            mean_accuracy (list): contains mean accuracy of each classifer. Length: n_classifiers. 
        """
        
        mean_accuracy = []

        for classifier in acc_scores.keys():
            for subject in acc_scores[classifier].keys():
                scores = np.array(list(acc_scores[classifier].values()))
            mean_acc = np.mean(scores)
            print(f"Mean accuracy of {classifier}: {100* mean_acc:.2f} %")
            
            mean_accuracy.append(mean_acc)
        return mean_accuracy
            
        

                            
              
            
        
