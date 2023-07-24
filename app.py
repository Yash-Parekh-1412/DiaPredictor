import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)

#load the pickle model
rfModel = pickle.load(open("rfmodel.pkl", "rb"))
dtModel = pickle.load(open("dtmodel.pkl", "rb"))

TEMPLATES_AUTO_RELOAD = True

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/rf")
def rf():
    return render_template("rf.html")

@app.route("/dt")
def dt():
    return render_template("dt.html")



@app.route("/predict", methods = ["POST", "GET"])
def rfpredict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    #Random Forest model 
    rfPrediction = rfModel.predict(features)
    rfOutput= round(rfPrediction[0], 2)
    
    # #Decision Tree Model
    # dtPrediction = dtModel.predict(features)
    # dtOutput= round(dtPrediction[0], 2)

    if rfOutput> 0.5:
        return render_template("rf.html", rfPred = "You are Diabetic")
    else:
        return render_template("rf.html", rfPred = "You are not Diabetic")


@app.route("/predict", methods = ["POST", "GET"])
def dfpredict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    # #Random Forest model 
    # rfPrediction = rfModel.predict(features)
    # rfOutput= round(rfPrediction[0], 2)
    
    #Decision Tree Model
    dtPrediction = dtModel.predict(features)
    dtOutput= round(dtPrediction[0], 2)

    if dtOutput> 0.5:
        return render_template("dt.html", dtPred = "You are Diabetic {}".format(dtOutput))
    else:
        return render_template("dt.html", dtPred = "You are not Diabetic {}".format(dtOutput))



if __name__ == "__main__":
    app.run(debug = True)