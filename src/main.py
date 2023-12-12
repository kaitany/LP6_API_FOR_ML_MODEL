from fastapi import FastAPI, HTTPException
import uvicorn
#from typing import List, Literal
from pydantic import BaseModel, Field
import pandas as pd
import pickle, os

 #setup
# Get the directory of the current file (FastAPI application file)
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Construct the path to ml.pkl relative to the current file using forward slashes
ml_core_fp = os.path.join(DIRPATH, "../model/ml.pkl")


#useful functions
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    

# Loading: Execute and instantiate ml components
ml_components_dict = load_ml_components(fp = ml_core_fp)

pipeline = ml_components_dict["pipeline"]

encoder =  ml_components_dict["encoder"]


# API
app = FastAPI(
    title= "Sepsis classification API"
)

# Input for Modelling 
class Sepsis_Pred(BaseModel):

    PRG: int = Field(..., description='Plasma glucose')
    PL: int = Field(..., description='Blood Work Result-1 (mu U/ml)')
    PR: int = Field(..., description='Blood Pressure (mm Hg)')
    SK: int = Field(..., description='Blood Work Result-2 (mm)')
    TS: int = Field(..., description='Blood Work Result-3 (mu U/ml)')
    M11: float = Field(..., description='Body mass index (weight in kg/(height in m)^2)')
    BD2: float = Field(..., description='Blood Work Result-4 (mu U/ml)')
    Age: int = Field(..., description='Patient age (years)')
    Insurance: int = Field(..., description='If a patient holds a valid insurance card')


@app.get("/")
def root():
    return {
        "Info": "Sepsis classification API : This API classifies whether a patient will develop sepsis based on various test results"
    }


@app.post("/classify_patient")
def sepsis_classification(sepsis_pred: Sepsis_Pred):

    try:

        #Dataframe creation
        df = pd.DataFrame([sepsis_pred.model_dump()])

        print(f'df: {df}')
       
        # ML prediction
        prediction = pipeline.predict(df)

        # Get the index of the predicted class (0 or 1 in binary classification)
        predicted_class_index = prediction[0]

        confidence_score = pipeline.predict_proba(df)

        # Retrieve the confidence score for the predicted class

        confidence_score_predicted_class = confidence_score[0][predicted_class_index]


        print(f"confidence_score: {confidence_score}")

        execution_message = "Execution successful"


        # encoded prediction
        decoded_prediction = encoder.inverse_transform([prediction])[0]

    
        return {"execution message": execution_message, "patient_diagnosis": decoded_prediction, "confidence_score": confidence_score_predicted_class}
    
    
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)