import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import PredictionDataValidation


class Prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.AppLogger()
        if path is not None:
            self.pred_data_val = PredictionDataValidation(path)

    def Prediction_From_Model(self):

        try:
            self.pred_data_val.Delete_Prediction_File() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object, 'Start of Prediction')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
            data=data_getter.Get_Data()


            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            print("data before removing duplicate rows: ", data.shape)
            df = preprocessor.Remove_Duplicates(data)
            print("data after removing duplicate rows: ", df.shape)            


            cols_to_drop = preprocessor.Get_Columns_With_Zero_Std_Deviation(df)
            data = preprocessor.Remove_Columns(df,cols_to_drop)

            file_loader = file_methods.FileOperation(self.file_object, self.log_writer)
            kmeans = file_loader.Load_Model('KMeans')


            clusters = kmeans.predict(data)
            data['clusters'] = clusters
            clusters = data['clusters'].unique()
            result=[]
            for i in clusters:
                cluster_data = data[data['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.Find_Correct_Model_File(i)
                model = file_loader.Load_Model(model_name)
                for val in (model.predict(cluster_data)):
                    result.append(val)
            result = pandas.DataFrame(result, columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv", header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path, result.head().to_json(orient="records")




