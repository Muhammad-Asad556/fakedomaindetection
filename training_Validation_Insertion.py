from datetime import datetime
from Training_Raw_data_validation.rawValidation import RawDataValidation
from DataTypeValidation_Insertion_Training.DataTypeValidation import DBOperation
from application_logging import logger

class TrainValidation:
    def __init__(self,path):
        self.raw_data = RawDataValidation(path)
        self.dBOperation = DBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.AppLogger()

    def Train_Validation(self):
        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files!!')
            # extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.Values_From_Schema()
            # getting the regex defined to validate filename
            regex = self.raw_data.Manual_Regex_Creation()
            # validating filename of prediction files
            self.raw_data.Validation_File_Name_Raw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            # validating column length in the file
            self.raw_data.Validate_Column_Length(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.Validate_Missing_Values_In_Whole_Column()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object,
                                "Creating Training_Database and tables on the basis of given schema!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.Create_Table_Db("Training", column_names)
            self.log_writer.log(self.file_object, "Table creation Completed!!")
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")
            # insert csv files in the table
            self.dBOperation.Insert_Into_Table_Good_Data('Training')
            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")
            # Delete the good data folder after loading files in table
            self.raw_data.Delete_Existing_Good_Data_Training_Folder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # Move the bad files to archive folder
            self.raw_data.Move_Bad_Files_To_Archive_Bad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            # export data in table to csvfile
            self.dBOperation.Selecting_Data_from_table_into_csv('Training')
            self.file_object.close()

        except Exception as e:
            raise e









