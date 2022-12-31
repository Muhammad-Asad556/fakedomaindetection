"""
This is the Entry point for Training the Machine Learning Model.

Written By: Muhammad Asad Majeed
Version: 1.0
Revisions: None

"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

#Creating the common Logging object


class TrainModel:

    def __init__(self):
        self.log_writer = logger.AppLogger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def Training_Model(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.DataGetter(self.file_object, self.log_writer)
            data=data_getter.Get_Data()


            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object, self.log_writer)

            print("data before removing duplicate rows: ", data.shape)
            df = preprocessor.Remove_Duplicates(data)
            print("data after removing duplicate rows: ", df.shape)

            # create separate features and labels
            X,Y = preprocessor.Separate_Label_Feature(df, label_column_name='phishing')



            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output
            # prepare the list of such columns to drop
            cols_to_drop = preprocessor.Get_Columns_With_Zero_Std_Deviation(X)

            # drop the columns obtained above
            X = preprocessor.Remove_Columns(X, cols_to_drop)


            # Handle The Imbalanced Dataset Using Nearmiss
            print("data before handling imbalanced dataset: ", X.shape)
            X_bal, y_bal = preprocessor.Handle_Imbalance_Dataset(X ,Y)
            print("data after handling imbalanced dataset: ", X_bal.shape)
           

            """ Applying the clustering approach"""

            kmeans = clustering.KMeansClustering(self.file_object, self.log_writer) # object initialization.
            number_of_clusters = kmeans.Elbow_Plot(X_bal)  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X = kmeans.Create_Clusters(X_bal, number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = y_bal

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['Cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels','Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder = tuner.ModelFinder(self.file_object, self.log_writer) # object initialization

                #getting the best model for each of the clusters
                best_model_name, best_model = model_finder.Get_Best_Model(x_train, y_train, x_test, y_test)

                #saving the best model to the directory.
                file_op = file_methods.FileOperation(self.file_object, self.log_writer)
                save_model=file_op.Save_Model(best_model, best_model_name+str(i))

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception