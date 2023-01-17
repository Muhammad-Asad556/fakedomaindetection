from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import PredValidation
from trainingModel import TrainModel
from training_Validation_Insertion import TrainValidation
import flask_monitoringdashboard as dashboard
from predictFromModel import Prediction
import json

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def Predict_Route_Client():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = PredValidation(path) #object initialization

            pred_val.Prediction_Validation() #calling the prediction_validation function

            pred = Prediction(path) #object initialization

            # predicting for dataset present in database
            path,json_predictions = pred.Prediction_From_Model()
            return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        elif request.form is not None:
            path = request.form['filepath']

            pred_val = PredValidation(path) #object initialization

            pred_val.Prediction_Validation() #calling the prediction_validation function

            pred = Prediction(path) #object initialization

            # predicting for dataset present in database
            path,json_predictions = pred.Prediction_From_Model()
            return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)



@app.route("/train", methods=['GET'])
@cross_origin()
def Train_Route_Client():

    try:
            path = "Training_Batch_Files"

            train_valObj = TrainValidation(path) #object initialization

            train_valObj.Train_Validation()#calling the training_validation function


            trainModelObj = TrainModel() #object initialization
            trainModelObj.Training_Model() #training the model for the files in the table


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

#port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    app.run(debug=True)
    # host = '0.0.0.0'
    # port = 5000
    # httpd = simple_server.make_server(host, port, app)
    # # # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
