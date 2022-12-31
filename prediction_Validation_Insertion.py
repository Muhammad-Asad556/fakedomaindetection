from datetime import datetime
from Prediction_Raw_Data_Validation.predictionDataValidation import PredictionDataValidation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import DBOperation
from application_logging import logger

class PredValidation:
    def __init__(self,path):
        self.raw_data = PredictionDataValidation(path)
        self.dBOperation = DBOperation()
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.AppLogger()

    def Prediction_Validation(self):

        try:

            self.log_writer.log(self.file_object,'Start of Validation on files for prediction!!')
            #extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.Values_From_Schema()
            #getting the regex defined to validate filename
            regex = self.raw_data.Manual_Regex_Creation()
            #validating filename of prediction files
            self.raw_data.Validation_File_Name_Raw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            #validating column length in the file
            self.raw_data.Validate_Column_Length(noofcolumns)
            #validating if any column has all values missing
            self.raw_data.Validate_Missing_Values_In_Whole_Column()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object, "Creating Prediction_Database and tables on the basis of given schema!!!")
            #create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.Create_Table_Db('Prediction', column_names)
            self.log_writer.log(self.file_object, "Table creation Completed!!")
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")
            #insert csv files in the table
            self.dBOperation.Insert_Into_Table_Good_Data('Prediction')
            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")
            #Delete the good data folder after loading files in table
            self.raw_data.Delete_Existing_Good_Data_Training_Folder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            #Move the bad files to archive folder
            self.raw_data.Move_Bad_Files_To_Archive_Bad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            #export data in table to csvfile
            self.dBOperation.Selecting_Data_from_table_into_csv('Prediction')

        except Exception as e:
            raise e









