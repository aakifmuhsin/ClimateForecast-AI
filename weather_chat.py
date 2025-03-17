from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from feedmodel import WeatherPredictor
import json

# Initialize the models
llama2_chat = ChatOllama(
    model="deepseek-r1:1.5b",
    num_ctx=512,
    repeat_penalty=1.1
)

# Initialize weather predictor
weather_predictor = WeatherPredictor()

# Load the trained model
try:
    weather_predictor.load_model('weather_model.joblib')
except:
    print("No pre-trained model found. Training new model...")
    X, y = weather_predictor.prepare_data('weather_history.csv')
    weather_predictor.train_model(X, y)
    weather_predictor.save_model()

# Define the prompt template for weather analysis
template = """You are a weather analysis expert. Based on the current conditions and predicted temperature, provide a detailed weather analysis.

Current Conditions:
- Temperature: {temperature}°C
- Dewpoint: {dewpoint}°C
- Humidity: {humidity}%

Predicted Next Temperature: {predicted_temp}°C

Please provide:
1. A summary of current conditions
2. Analysis of the prediction
3. Comfort level assessment
4. Recommendations based on the conditions

Response:"""

prompt = ChatPromptTemplate.from_messages([("human", template)])


def get_weather_analysis(current_conditions):
    """Get weather prediction and analysis"""
    try:
        # Get prediction from weather model
        predicted_temp = weather_predictor.predict_next(current_conditions)
        
        # Get analysis from LLM
        formatted_prompt = prompt.format(
            temperature=current_conditions['temperature'],
            dewpoint=current_conditions['dewpoint'],
            humidity=current_conditions['humidity'],
            predicted_temp=f"{predicted_temp:.2f}"
        )
        
        # Get response from the model (assuming it directly returns a string)
        response = llama2_chat(formatted_prompt)
        
        if isinstance(response, str):
            return {
                "current_conditions": current_conditions,
                "predicted_temperature": f"{predicted_temp:.2f}°C",
                "analysis": response
            }
        else:
            return {
                "current_conditions": {},
                "predicted_temperature": f"{predicted_temp:.2f}°C",
                "analysis": "Error in generating analysis."
            }

    except Exception as e:
        return {
            "current_conditions": {},
            "predicted_temperature": "N/A",
            "analysis": f"Error in weather analysis: {str(e)}"
        }



# Example usage
if __name__ == "__main__":
    current_conditions = {
        'temperature': 25.0,
        'dewpoint': 15.0,
        'humidity': 65.0
    }
    
    result = get_weather_analysis(current_conditions)
    
    print("\n=== Weather Analysis Report ===")
    print("\nCurrent Conditions:")
    if isinstance(result["current_conditions"], dict):
        for key, value in result["current_conditions"].items():
            print(f"{key.title()}: {value}")
    else:
        print("Error: current_conditions is not a dictionary")

    
    print(f"\nPredicted Temperature: {result['predicted_temperature']}")
    print("\nAnalysis:")
    print(result["analysis"])
