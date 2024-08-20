from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Initialize the FastAPI app
app = FastAPI()

# Define the BaseModel for input validation
class PredictionInput(BaseModel):
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: int
    Seller_Type: int
    Transmission: int
    Owner: int

# Load the saved models
models_folder = "./models"
linear_model_path = os.path.join(models_folder, "linear_regression_model.pkl")
lasso_model_path = os.path.join(models_folder, "lasso_regression_model.pkl")

try:
    linear_model = joblib.load(linear_model_path)
    lasso_model = joblib.load(lasso_model_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading models: {e}")

# Endpoint for Linear Regression Prediction
@app.post("/car_price_prediction/linear")
async def predict_linear(input_data: PredictionInput):
    try:
        features = [[input_data.Year, input_data.Present_Price, input_data.Kms_Driven,input_data.Fuel_Type, input_data.Seller_Type, input_data.Transmission, input_data.Owner]]
        # Add more features as required
        prediction = linear_model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

# Endpoint for Lasso Regression Prediction
@app.post("/car_price_prediction/lasso")
async def predict_lasso(input_data: PredictionInput):
    try:
        features = [[input_data.Year, input_data.Present_Price, input_data.Kms_Driven,input_data.Fuel_Type, input_data.Seller_Type, input_data.Transmission, input_data.Owner]]
        # Add more features as required
        prediction = lasso_model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
