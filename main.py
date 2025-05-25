from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Command to run the server: uvicorn main:app --reload

app = FastAPI()

# Add CORS middleware IMMEDIATELY after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-predict-cdf1v4bei-henasivenoms-projects.vercel.app"],  # Only allow your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load("model/insurance_model.pkl")
print("Model expects columns:", getattr(model, 'feature_names_in_', 'Unknown'))
print("Model type:", type(model))

# Define the input schema
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# Preprocess input data
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data['age'] = data['age'].astype(int)
    data['bmi'] = data['bmi'].astype(float)
    data['children'] = data['children'].astype(int)
    # Do NOT touch sex, smoker, region
    data = data.fillna('')
    print("Preprocessed data:\n", data)
    print("Data types after preprocessing:\n", data.dtypes)
    return data

# Prediction endpoint
@app.post("/predict")
def predict_cost(input: InsuranceInput):
    data = pd.DataFrame([input.dict()])
    print("Raw input data:\n", data)
    print("Raw data types:\n", data.dtypes)
    data = preprocess(data)
    expected_cols = getattr(model, 'feature_names_in_', None)
    if expected_cols is not None:
        data = data[list(expected_cols)]
    print("Final data for prediction:\n", data)
    print("Final data types:\n", data.dtypes)
    try:
        prediction = model.predict(data)
        return {"prediction": round(float(prediction[0]), 2)}
    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}


