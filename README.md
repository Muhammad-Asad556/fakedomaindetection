
![Capture](https://user-images.githubusercontent.com/121654705/210150539-9a6f4d60-6140-4b1c-846a-396551ae1297.PNG)


# **_Training Pipeline_**


## **Data Batches For Training**

It is actually a location from where we pull the data for training our machine learning model.

---

## **Data Validation**

After pull the data, here we do some data sharing agreement with the client that what will be the file name, file extension, length of time and date stamp, what will be the column names and length of columns. and based on all, we prepare a master data management. good files will be moved into good raw directory for further processing and bad files will be moved into bad directory. these bad files will be used for notification/alarm to client.

---

## **Data Transformation**

In this step, generally, we will replace missing values with null,add new column or convert data type of column if required.

---

## **Data Insertion Into Database**

finally here we will load all batch files one by one into our database.

---

## **Export CSV file from database**

After loading all the files we will export one master csv file from that database for further processing.

---

## **Data Preprocessing**

In this step, we will check if there is any null values in our dataset, if present we will do imputation. secondly, we will check and remove all the columns whose std=0.

---

## **Data Clustering**

After preprocessing is done, we will divide data into various clustors with the help of kmeans. To choose optimum number of cluster we will use knee locator method. 

---

## **Get Best Model Of Each Cluster**

We will train separate model for each cluster and after that we will compare the score and based on that we will finalize them for each cluster. And hence in this way our customized ml approach will be satisfied.

---

## **Hyperparameter Tuning**

Here we will tune all the parameters that are important to finalize our model to acheive maximum accuracy.

---

## **Model Saving**

in this step, we will save our models into their sav file for prediction.

---

## **Cloud Setup And Pushing App To The Cloud**

we will choose one cloud to host our solution in the form of API So that client will be able to send data and do predictions.

---
---

# **_Prediction Pipeline_**

## **Data From Client To Be Predicted**

The client will send data through API for predictions and makes decision.

---

## **Data Validation**

in this step we will do same operation as we have already done in the training pipeline except the length of columns will be change because the output column will not be present.

---

## **Data Transformation**

here we will replace missing values with null,add new column or convert data type of column if required.

---

## **Data Insertion Into Database**

we will insert data into database for further processing.

---

## **Export Data To csv For Prediction**

we will export data from database into one csv file.

---

## **Data Preprocessing**

For prediction data, We will perform same operations as we had already performed over the training data.

---

## **Data Clustering**

we will divide data into various clustors with the help of kmeans. To choose optimum number of cluster we will use knee locator method. 

---

## **Model Call For Specific Cluster and Prediction**

we will call each model for each cluster and do predictions.

---

## **Export Prediction To CSV**

finally, we will save predictions into one csv file and use this file and make final decision.

---
---
