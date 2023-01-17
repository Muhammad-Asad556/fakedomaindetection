import pandas as pd
import numpy as np
from imblearn.under_sampling import NearMiss

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: Muhammad Asad Majeed
        Version: 1.0
        Revisions: None

        """
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    
    def Remove_Duplicates(self, data):
        """
                                                Method Name: Remove_Duplicates
                                                Description: This method used to remove duplicates from the dataset
                                                Output: Return Dataset Without any duplicate rows
                                                On Failure: Raise Exception

                                                Written By: Muhammad Asad Majeed
                                                Version: 1.0
                                                Revisions: None
        """
        self.logger_object.log(self.file_object, 'Entered the Remove_Duplicates method of the Preprocessor class')
        
        try:
            df = data.drop_duplicates()
            self.logger_object.log(self.file_object, 'Removing The Duplicates From the Data Successful. Exited the Remove_Duplicates method of the Preprocessor class')
            return df
        
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in Remove_Duplicates method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()

    def Separate_Label_Feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: Muhammad Asad Majeed
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

   

    def Remove_Columns_For_Zero_Std_Deviation(self, data):
        """
                                                Method Name: Remove_Columns_For_Zero_Std_Deviation
                                                Description: This method removes the columns which have a standard deviation zero.
                                                Output: Return Dataset With Columns without std deviation=0
                                                On Failure: Raise Exception

                                                Written By: Muhammad Asad Majeed
                                                Version: 1.0
                                                Revisions: None
                             """
        self.logger_object.log(self.file_object, 'Entered the remove_columns_for_zero_std_deviation method of the Preprocessor class')
        try:
            self.data = data
            new_data = self.data.drop([
                                        'qty_slash_domain','qty_questionmark_domain','qty_equal_domain',
                                        'qty_and_domain','qty_exclamation_domain','qty_space_domain',
                                        'qty_tilde_domain','qty_comma_domain','qty_plus_domain',
                                        'qty_asterisk_domain','qty_hashtag_domain','qty_dollar_domain',
                                        'qty_percent_domain'
                                        ], axis=1
                                    ) 
            self.logger_object.log(self.file_object, 'Column Removes for Standard Deviation of Zero Successful. Exited the remove_columns_for_zero_std_deviation method of the Preprocessor class')
            return new_data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in remove_columns_for_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()
    
    def Handle_Imbalance_Dataset(self, X, Y):
        """
                                                Method Name: Handle_Imbalance_Dataset
                                                Description: This method used to down sampling the dataset
                                                Output: Return Balanced Dataset
                                                On Failure: Raise Exception

                                                Written By: Muhammad Asad Majeed
                                                Version: 1.0
                                                Revisions: None
        """
        self.logger_object.log(self.file_object, 'Entered the Handle_Imbalance_Dataset method of the Preprocessor class')
        
        try:
            sample = NearMiss()
            X_bal,y_bal = sample.fit_resample(X, Y)
            self.logger_object.log(self.file_object, 'Down Sampling the Data Successful. Exited the Handle_Imbalance_Dataset method of the Preprocessor class')
            return X_bal,y_bal
        
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in Handle_Imbalance_Dataset method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()