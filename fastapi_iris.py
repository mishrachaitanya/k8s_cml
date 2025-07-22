#FASTAPI Code
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define input data model using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# GET endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI iris classification endpoint"}

# POST endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    try:
        # Convert input features into numpy array
        input_data = np.array([[features.sepal_length, features.sepal_width,
                                features.petal_length, features.petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return predicted class
        return {"predicted_species": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

