from fastapi import FastAPI, Request
from pydantic import BaseModel
from core_logic import process_query_realtime, load_argo_index
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# --- CORS Middleware Configuration ---
origins = ["*"]  # For development, allow all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    On startup, load the ARGO index data into the app's state.
    """
    app.state.argo_df = await load_argo_index()


class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: Request, query_request: QueryRequest):
    """
    Handles the natural language query from the frontend.
    It now uses the cached dataframe from the application state.
    """
    # Access the cached dataframe from the app state
    argo_df = request.app.state.argo_df
    
    # Pass both the user's query and the dataframe to the processing function
    response_data = await process_query_realtime(query_request.query, argo_df)
    return response_data

