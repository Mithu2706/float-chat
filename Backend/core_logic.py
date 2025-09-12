import pandas as pd
import xarray as xr
import httpx
import io
import re
from dotenv import load_dotenv
# --- Add back AI-related imports, now for Perplexity ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_perplexity import ChatPerplexity # Import Perplexity model

# --- Data Loading (to be called once at startup) ---
async def load_argo_index():
    """
    Downloads and parses the main ARGO index file.
    This should be run once when the application starts.
    """
    print("Downloading ARGO index file... (This may take a moment)")
    index_url = "https://data-argo.ifremer.fr/ar_index_global_meta.txt"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(index_url)
        response.raise_for_status()
    
    content = response.text
    data_io = io.StringIO("\n".join(line for line in content.splitlines() if not line.startswith('#')))
    
    column_names = [
        "file", "date", "latitude", "longitude", "ocean", "profiler_type",
        "institution", "date_update"
    ]
    df = pd.read_csv(data_io, header=None, names=column_names)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S', errors='coerce')
    df['date_update'] = pd.to_datetime(df['date_update'], format='%Y%m%d%H%M%S', errors='coerce')
    
    def get_wmo_id(path):
        parts = str(path).split('/')
        return parts[1] if len(parts) > 1 else None

    df['wmo_id'] = df['file'].apply(get_wmo_id)
    df.dropna(subset=['wmo_id', 'date', 'date_update'], inplace=True)
    
    print("✅ ARGO index file loaded and parsed successfully.")
    return df


# --- Pydantic model for structured LLM output ---
class ArgoQueryResponse(BaseModel):
    """Defines the structured JSON output for the AI model."""
    text_response: str = Field(description="A user-friendly text summary of the findings.")
    wmo_ids: list[str] = Field(description="A list of relevant ARGO float WMO IDs based on the user's query.")
    variables: list[str] = Field(description="A list of variables to plot (e.g., 'TEMP', 'PSAL').")

# --- AI-Powered Query Processing with Perplexity ---
async def process_query_realtime(query: str, argo_df: pd.DataFrame):
    """
    Processes a user query using Perplexity AI to understand natural language
    and extract structured information.
    """
    load_dotenv()
    
    # 1. Initialize the Perplexity model and the JSON output parser
    # Using a Llama 3 Sonar model, which is fast and capable.
    llm = ChatPerplexity(model="llama-3-sonar-large-32k-online", temperature=0)
    parser = JsonOutputParser(pydantic_object=ArgoQueryResponse)
    
    # 2. Create a prompt template to guide the AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert oceanographic data assistant. Analyze the user's query and the provided data context to extract key information. Respond ONLY with a valid JSON object based on the following format instructions:\n{format_instructions}"),
        ("human", "User Query: '{query}'.\n\nSample of available data:\n{data_context}'.\n\nBased on the query and data, provide the structured JSON response.")
    ])
    
    # 3. Create the processing chain
    chain = prompt | llm | parser
    
    # 4. Invoke the chain with the query and a small sample of the data as context
    data_context = argo_df.head().to_string()
    response_json = await chain.ainvoke({
        "query": query,
        "data_context": data_context,
        "format_instructions": parser.get_format_instructions()
    })
    
    # 5. Fetch data based on the AI's structured response
    wmo_ids_to_fetch = response_json.get('wmo_ids', [])
    variables_to_plot = response_json.get('variables', [])
    
    map_data = {'lat': [], 'lon': [], 'wmo_ids': []}
    profile_data = {}
    base_url = "https://data-argo.ifremer.fr/dac"
    
    for wmo_id in wmo_ids_to_fetch[:5]: # Limit to 5 floats
        try:
            latest_entry = argo_df[argo_df['wmo_id'] == wmo_id].sort_values('date_update', ascending=False).iloc[0]
            file_path = latest_entry['file']
            
            map_data['lat'].append(latest_entry['latitude'])
            map_data['lon'].append(latest_entry['longitude'])
            map_data['wmo_ids'].append(wmo_id)

            if variables_to_plot:
                profile_url = f"{base_url}/{file_path.replace('meta.nc', 'prof.nc')}"
                async with httpx.AsyncClient(timeout=30.0) as client:
                    profile_res = await client.get(profile_url)
                    profile_res.raise_for_status()

                with xr.open_dataset(io.BytesIO(profile_res.content), engine="h5netcdf") as ds:
                    if 'TEMP' in variables_to_plot and 'TEMP' in ds:
                        profile_data.setdefault(wmo_id, {})['temp'] = {
                            'pressure': ds['PRES'].values.flatten().tolist(),
                            'values': ds['TEMP'].values.flatten().tolist()
                        }
                    if 'PSAL' in variables_to_plot and 'PSAL' in ds:
                         profile_data.setdefault(wmo_id, {})['sal'] = {
                            'pressure': ds['PRES'].values.flatten().tolist(),
                            'values': ds['PSAL'].values.flatten().tolist()
                        }
        except Exception as e:
            print(f"Could not fetch or process profile for {wmo_id}: {e}")
            
    # 6. Return the final structured response for the frontend
    return {
        "text_response": response_json.get('text_response', 'Could not generate a text response.'),
        "map_data": map_data,
        "profile_data": profile_data
    }

