from typing import Dict, Any
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle


app = FastAPI(title="ML Model API", description="Machine Learning Model with ngrok tunnel")

class CarFeatures(BaseModel):
    cylinders: int
    displacement: int
    horsepower: int
    weight: int
    acceleration: int
    modelYear: int
    origin: int
    
@app.post("/predict_car_mph")
def pred(features: CarFeatures) -> Dict[str, Any]:
    try:
        model = pickle.load(open('./data/model.pkl','rb'))
        prediction = model.predict([[
            features.cylinders,
            features.displacement,
            features.horsepower,
            features.weight,
            features.acceleration,
            features.modelYear,
            features.origin
        ]])
        if prediction[0] < 0:
            prediction[0] = 0
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    
    # Start FastAPI server
    print("Starting FastAPI server on http://localhost:5000")
    print("API documentation available at: http://localhost:5000/docs")

    #http://localhost:5000/
    uvicorn.run(app, host="0.0.0.0", port=5000)
