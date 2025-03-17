from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from weather_chat import get_weather_analysis

app = FastAPI()


class WeatherConditions(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    precipitation: float


@app.post("/predict")
async def predict_weather(conditions: WeatherConditions):
    try:
        current_conditions = conditions.dict()
        result = get_weather_analysis(current_conditions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn weather_api:app --reload
