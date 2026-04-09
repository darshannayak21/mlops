from fastapi import FastAPI, HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Any, Dict
from contextlib import asynccontextmanager
from pathlib import Path
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

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "index.html"
ROLLING_STAT_COLUMNS = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot', 'pk']
WINNER_GAP_THRESHOLD = 0.05
OPPONENT_NAME_OVERRIDES = {
    "Brighton And Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Nottingham Forest": "Nott'ham Forest",
    "Sheffield United": "Sheffield Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# Global state for model and data
ml_models: Dict[str, Any] = {}
app_data: Dict[str, Any] = {}
mappings: Dict[str, dict] = {}

def _normalize_name_key(value: str) -> str:
    return ''.join(character for character in value.lower() if character.isalnum())

def _build_team_aliases(team_names):
    aliases = {}

    for team_name in team_names:
        aliases[_normalize_name_key(team_name)] = team_name
        opponent_name = OPPONENT_NAME_OVERRIDES.get(team_name, team_name)
        aliases[_normalize_name_key(opponent_name)] = team_name

    return aliases

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

        canonical_teams = sorted(all_matches['team'].dropna().unique())

        mappings['team_codes'] = dict(zip(all_matches['team'], all_matches['team_code']))
        mappings['opp_codes'] = dict(zip(all_matches['opponent'], all_matches['opp_code']))
        mappings['venue_codes'] = dict(zip(all_matches['venue'], all_matches['venue_code']))
        mappings['canonical_to_opponent'] = {
            team_name: OPPONENT_NAME_OVERRIDES.get(team_name, team_name)
            for team_name in canonical_teams
        }
        mappings['team_aliases'] = _build_team_aliases(canonical_teams)
        app_data['teams'] = canonical_teams
        
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

class WinnerPredictionRequest(BaseModel):
    home_team: str = Field(..., description="Name of the home team")
    away_team: str = Field(..., description="Name of the away team")
    match_date: str = Field(..., description="Date of the match (YYYY-MM-DD)")

class WinnerPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    match_date: str
    predicted_winner: str
    result_summary: str
    home_win_probability: float
    away_win_probability: float

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
# Helper Functions
# ==========================================
def _normalize_text(value: str) -> str:
    return value.strip()

def _canonicalize_team_name(value: str) -> str:
    normalized_value = _normalize_text(value)
    team_aliases = mappings.get('team_aliases', {})
    canonical_name = team_aliases.get(_normalize_name_key(normalized_value))

    if canonical_name is None:
        logger.warning(f"Unknown team requested: {value}")
        raise HTTPException(status_code=404, detail=f"Team '{value}' not found in historical training data.")

    return canonical_name

def _get_model_and_matches():
    model = ml_models.get('football_model')
    all_matches = app_data.get('matches')

    if model is None or all_matches is None:
        logger.error("Model or historical data is missing or failed to load on startup.")
        raise HTTPException(status_code=500, detail="Prediction model is currently unavailable.")

    return model, all_matches

def _parse_match_date(match_date: str) -> datetime:
    try:
        return datetime.strptime(match_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format received: {match_date}")
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

def _build_feature_frame(team_1: str, team_2: str, match_venue: str, match_date: str):
    model, all_matches = _get_model_and_matches()

    team_1 = _canonicalize_team_name(team_1)
    team_2 = _canonicalize_team_name(team_2)
    match_venue = _normalize_text(match_venue)
    prediction_date = _parse_match_date(match_date)

    team_codes = mappings.get('team_codes', {})
    opp_codes = mappings.get('opp_codes', {})
    canonical_to_opponent = mappings.get('canonical_to_opponent', {})
    venue_codes = mappings.get('venue_codes', {})
    opponent_name = canonical_to_opponent.get(team_2, team_2)

    if team_1 not in team_codes or opponent_name not in opp_codes:
        logger.warning(f"Unknown teams requested: {team_1} vs {team_2}")
        raise HTTPException(status_code=404, detail="One or both teams not found in historical training data.")

    if match_venue not in venue_codes:
        logger.warning(f"Unknown venue requested: {match_venue}")
        raise HTTPException(status_code=400, detail="Venue must exactly be 'Home' or 'Away'.")

    team_matches = all_matches[all_matches['team'] == team_1].sort_values('date')
    past_matches = team_matches[team_matches['date'] < prediction_date]

    if len(past_matches) < 5:
        logger.warning(f"Not enough historical data for {team_1}")
        raise HTTPException(
            status_code=400,
            detail=f"Not enough historical data (fewer than 5 matches) for '{team_1}' before {match_date}."
        )

    rolling_stats = past_matches.tail(5)[ROLLING_STAT_COLUMNS].mean()
    rolling_averages = {f'{column}_rolling': value for column, value in rolling_stats.items()}

    feature_dict = {
        'venue_code': venue_codes[match_venue],
        'opp_code': opp_codes[opponent_name],
        'team_code': team_codes[team_1],
        'day_code': prediction_date.weekday()
    }
    feature_dict.update(rolling_averages)

    input_df = pd.DataFrame([feature_dict])[model.feature_names_in_]
    return model, input_df, team_1, team_2, match_venue

def _predict_team_result(team_1: str, team_2: str, match_venue: str, match_date: str):
    model, input_df, team_1, team_2, match_venue = _build_feature_frame(
        team_1=team_1,
        team_2=team_2,
        match_venue=match_venue,
        match_date=match_date
    )

    try:
        prediction_raw = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
    except Exception as e:
        logger.error(f"Error during model inference calculation: {e}")
        raise HTTPException(status_code=500, detail="Model computation failed during prediction phase.")

    outcome = "Win" if prediction_raw[0] == 1 else "Loss or Draw"
    win_probability = float(prediction_proba[0][1])

    logger.info(f"Prediction successful for {team_1}: {outcome} ({win_probability:.2%})")
    return outcome, win_probability, team_1, team_2, match_venue

def _build_prediction_response(request: PredictionRequest) -> PredictionResponse:
    outcome, win_probability, team_1, team_2, match_venue = _predict_team_result(
        team_1=request.team_1,
        team_2=request.team_2,
        match_venue=request.match_venue,
        match_date=request.match_date
    )

    return PredictionResponse(
        team_1=team_1,
        team_2=team_2,
        match_venue=match_venue,
        match_date=request.match_date,
        predicted_outcome=outcome,
        win_probability=win_probability
    )

def _build_winner_response(request: WinnerPredictionRequest) -> WinnerPredictionResponse:
    home_team = _normalize_text(request.home_team)
    away_team = _normalize_text(request.away_team)

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="Home team and away team must be different.")

    _, home_win_probability, normalized_home_team, normalized_away_team, _ = _predict_team_result(
        team_1=home_team,
        team_2=away_team,
        match_venue="Home",
        match_date=request.match_date
    )
    _, away_win_probability, _, _, _ = _predict_team_result(
        team_1=away_team,
        team_2=home_team,
        match_venue="Away",
        match_date=request.match_date
    )

    probability_gap = abs(home_win_probability - away_win_probability)

    if probability_gap < WINNER_GAP_THRESHOLD:
        predicted_winner = "Too close to call"
        result_summary = "The model sees this as a very even match, so a draw or tight result is possible."
    elif home_win_probability > away_win_probability:
        predicted_winner = normalized_home_team
        result_summary = f"{normalized_home_team} has the stronger win signal from the model."
    else:
        predicted_winner = normalized_away_team
        result_summary = f"{normalized_away_team} has the stronger win signal from the model."

    return WinnerPredictionResponse(
        home_team=normalized_home_team,
        away_team=normalized_away_team,
        match_date=request.match_date,
        predicted_winner=predicted_winner,
        result_summary=result_summary,
        home_win_probability=home_win_probability,
        away_win_probability=away_win_probability
    )

# ==========================================
# Frontend Routes
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"Frontend file not found at {FRONTEND_PATH}")
        raise HTTPException(status_code=500, detail="Frontend file is missing.")

@app.get("/ui/teams")
async def list_teams():
    teams = app_data.get('teams')

    if not teams:
        raise HTTPException(status_code=500, detail="Teams are not available right now.")

    return {"teams": teams}

@app.post("/ui/predict-winner", response_model=WinnerPredictionResponse)
async def predict_winner_ui(request: WinnerPredictionRequest):
    logger.info(f"Processing winner prediction request: {request.model_dump()}")
    return _build_winner_response(request)

# ==========================================
# Prediction Endpoint
# ==========================================
@app.post("/predict", response_model=PredictionResponse, dependencies=[Security(get_api_key)])
async def predict_match(request: PredictionRequest):
    logger.info(f"Processing prediction request: {request.model_dump()}")
    return _build_prediction_response(request)

if __name__ == "__main__":
    import uvicorn
    # Make sure we start via uvicorn so lifespan contexts execute
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
