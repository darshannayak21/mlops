from fastapi import FastAPI, HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import pickle
from datetime import datetime
import logging
import time

# ==========================================
# Expt 3: Structured Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("football_api")

# Global state for model and data
ml_models: Dict[str, Any] = {}
app_data: Dict[str, pd.DataFrame] = {}
mappings: Dict[str, dict] = {}

# ==========================================
# Lifespan Context Manager (replaces app.on_event("startup"))
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    try:
        logger.info("Starting up: Loading model and historical data...")
        with open('model.pkl', 'rb') as f:
            ml_models['football_model'] = pickle.load(f)
        
        # Load necessary historical state for calculating rolling average input features
        all_matches = pd.read_csv('final_matches.csv')
        all_matches['date'] = pd.to_datetime(all_matches['date'])
        app_data['matches'] = all_matches
        
        # Build category mappings to match training state
        all_matches['team_code'] = all_matches['team'].astype('category').cat.codes
        all_matches['opp_code'] = all_matches['opponent'].astype('category').cat.codes
        all_matches['venue_code'] = all_matches['venue'].astype('category').cat.codes

        mappings['team_codes'] = dict(zip(all_matches['team'], all_matches['team_code']))
        mappings['opp_codes'] = dict(zip(all_matches['opponent'], all_matches['opp_code']))
        mappings['venue_codes'] = dict(zip(all_matches['venue'], all_matches['venue_code']))
        
        logger.info("Successfully loaded model and data.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        ml_models['football_model'] = None
    
    yield # API runs safely here
    
    # Shutdown logic
    logger.info("Shutting down API: Cleaning up resources...")
    ml_models.clear()
    app_data.clear()
    mappings.clear()

app = FastAPI(
    title="Football Match Prediction API",
    description="REST API to predict football match outcomes utilizing a scikit-learn model.",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# Expt 4: Basic Authentication
# ==========================================
API_KEY = "super_secret_api_key_123"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        logger.warning(f"Unauthorized access attempt with API Key: {api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key credentials",
        )

# ==========================================
# Data Validation (Schemas)
# ==========================================
class PredictionRequest(BaseModel):
    team_1: str = Field(..., description="Name of the team (e.g. 'Manchester City')")
    team_2: str = Field(..., description="Name of the opponent (e.g. 'Arsenal')")
    match_venue: str = Field(..., description="Venue: 'Home' or 'Away' from perspective of team_1")
    match_date: str = Field(..., description="Date of the match (YYYY-MM-DD)")

class PredictionResponse(BaseModel):
    team_1: str
    team_2: str
    match_venue: str
    match_date: str
    predicted_outcome: str
    win_probability: float

# ==========================================
# Expt 3: Request/Response Logging Middleware
# ==========================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Completed request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response

# ==========================================
# Expt 3: Exception Handlers
# ==========================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Invalid input data format."},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "An unexpected server error occurred. Please try again later."},
    )

# ==========================================
# Prediction Endpoint
# ==========================================
@app.post("/predict", response_model=PredictionResponse, dependencies=[Security(get_api_key)])
async def predict_match(request: PredictionRequest):
    logger.info(f"Processing prediction request: {request.model_dump()}")
    
    model = ml_models.get('football_model')
    all_matches = app_data.get('matches')
    
    if model is None or all_matches is None:
        logger.error("Model or historical data is missing or failed to load on startup.")
        raise HTTPException(status_code=500, detail="Prediction model is currently unavailable.")
    
    try:
        prediction_date = datetime.strptime(request.match_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format received: {request.match_date}")
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    
    team_codes = mappings.get('team_codes', {})
    opp_codes = mappings.get('opp_codes', {})
    venue_codes = mappings.get('venue_codes', {})
    
    if request.team_1 not in team_codes or request.team_2 not in opp_codes:
        logger.warning(f"Unknown teams requested: {request.team_1} vs {request.team_2}")
        raise HTTPException(status_code=404, detail="One or both teams not found in historical training data.")
        
    if request.match_venue not in venue_codes:
        logger.warning(f"Unknown venue requested: {request.match_venue}")
        raise HTTPException(status_code=400, detail="Venue must exactly be 'Home' or 'Away'.")

    # --- Feature Engineering ---
    cols = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot', 'pk']
    
    team_matches = all_matches[all_matches['team'] == request.team_1].sort_values('date')
    past_matches = team_matches[team_matches['date'] < prediction_date]
    
    if len(past_matches) < 5:
        logger.warning(f"Not enough historical data for {request.team_1}")
        raise HTTPException(status_code=400, detail=f"Not enough historical data (fewer than 5 matches) for '{request.team_1}' before {request.match_date}.")
        
    rolling_stats = past_matches.tail(5)[cols].mean()
    rolling_averages_team1 = {f'{c}_rolling': val for c, val in rolling_stats.items()}
    
    # Feature ordering enforced via `.feature_names_in_`
    predictors = model.feature_names_in_
    
    feature_dict = {
        'venue_code': venue_codes[request.match_venue],
        'opp_code': opp_codes[request.team_2],
        'team_code': team_codes[request.team_1],
        'day_code': prediction_date.weekday()
    }
    feature_dict.update(rolling_averages_team1)
    
    # Format directly for Scikit-Learn (batch size x features array format managed by pandas DataFrame context)
    input_df = pd.DataFrame([feature_dict])[predictors]
    
    try:
        # --- Make Prediction ---
        prediction_raw = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        outcome = "Win" if prediction_raw[0] == 1 else "Loss or Draw"
        win_probability = float(prediction_proba[0][1])
        
        logger.info(f"Prediction successful for {request.team_1}: {outcome} ({win_probability:.2%})")
        
        return PredictionResponse(
            team_1=request.team_1,
            team_2=request.team_2,
            match_venue=request.match_venue,
            match_date=request.match_date,
            predicted_outcome=outcome,
            win_probability=win_probability
        )
    except Exception as e:
        logger.error(f"Error during model inference calculation: {e}")
        raise HTTPException(status_code=500, detail="Model computation failed during prediction phase.")

if __name__ == "__main__":
    import uvicorn
    # Make sure we start via uvicorn so lifespan contexts execute
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
