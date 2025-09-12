import pandas as pd
import xarray as xr
import httpx
import io
import re
from dotenv import load_dotenv
# --- Update AI-related imports for Google Gemini ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI # Import Google Gemini model
# --- New imports for local file system access ---
import os
import numpy as np

# --- Helper function for robustly opening NetCDF files ---
def open_argo_file(file_path):
    """
    Tries to open an ARGO NetCDF file using different engines.
    This handles variations in the NetCDF format.
    """
    ds = None
    try:
        # Try the h5netcdf engine first
        ds = xr.open_dataset(file_path, engine="h5netcdf")
    except (OSError, IOError) as e:
        # If there's a file signature error, it's likely a different NetCDF format
        if 'signature' in str(e).lower():
            print(f"Info: h5netcdf failed for {os.path.basename(file_path)}. Trying netcdf4 engine.")
            try:
                # Fallback to the netcdf4 engine
                ds = xr.open_dataset(file_path, engine="netcdf4")
            except Exception as e2:
                print(f"Warning: Could not open file {os.path.basename(file_path)} with netcdf4 engine either. Error: {e2}")
                return None
        else:
            # Re-raise other IOErrors that are not signature-related
            print(f"Warning: Could not open file {os.path.basename(file_path)}. Error: {e}")
            return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred with {os.path.basename(file_path)}. Error: {e}")
        return None
    return ds

# --- Data Loading (to be called once at startup) ---
async def load_argo_index():
    """
    Scans a local 'data' directory for .nc files and builds an in-memory
    index of the available ARGO float data.
    This should be run once when the application starts.
    """
    print("Scanning for local ARGO .nc files...")
    data_dir = "data"
    metadata_list = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The '{data_dir}' directory was not found. Please create it and add your .nc files.")

    # Walk through the data directory and find all NetCDF files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.nc'):
                file_path = os.path.join(root, file)
                
                ds = open_argo_file(file_path)
                
                if ds:
                    with ds:
                        # Extract metadata from the local file
                        lat = ds.LATITUDE.values[0]
                        lon = ds.LONGITUDE.values[0]
                        
                        # Handle cases where JULD might be a datetime object already
                        juld_val = ds.JULD.values[0]
                        if np.issubdtype(ds.JULD.dtype, np.datetime64):
                             date = pd.to_datetime(juld_val)
                        else: # Otherwise, it's a Julian day relative to 1950
                             date = pd.to_datetime(juld_val, origin='1950-01-01', unit='D')
                        
                        wmo_id = str(ds.PLATFORM_NUMBER.values.astype(str)[0].strip())
                        
                        metadata_list.append({
                            "file": file, # Store just the filename
                            "date": date,
                            "latitude": lat,
                            "longitude": lon,
                            "ocean": ds.attrs.get('ocean', 'UNKNOWN'),
                            "wmo_id": wmo_id,
                            "date_update": date,
                        })

    if not metadata_list:
        raise ValueError("No valid ARGO .nc files found or processed in the 'data' directory. Check the files in the 'data' folder.")

    df = pd.DataFrame.from_records(metadata_list)
    print(f"✅ Local data index created successfully with {len(df)} entries.")
    return df


# --- Pydantic model for structured LLM output ---
class ArgoQueryResponse(BaseModel):
    """Defines the structured JSON output for the AI model."""
    text_response: str = Field(description="A user-friendly text summary of the findings.")
    wmo_ids: list[str] = Field(description="A list of relevant ARGO float WMO IDs based on the user's query.")
    variables: list[str] = Field(description="A list of variables to plot (e.g., 'TEMP', 'PSAL').")
    visualization_type: str = Field(description="The type of visualization requested. Must be one of: 'plot_profile', 'plot_map', 'bar', 'pie', or 'text_only'.")


def clean_profile_data(pressure_raw, values_raw):
    """
    Removes non-compliant JSON values (NaN, infinity) from profile data.
    """
    # Flatten the arrays in case they are multidimensional
    pressure = np.array(pressure_raw, dtype=float).flatten()
    values = np.array(values_raw, dtype=float).flatten()
    
    # Create a mask for valid, finite numbers in both arrays
    valid_mask = np.isfinite(values) & np.isfinite(pressure)
    
    # Apply the mask to both pressure and values
    pressure_clean = pressure[valid_mask].tolist()
    values_clean = values[valid_mask].tolist()
    
    return pressure_clean, values_clean

# --- AI-Powered Query Processing with Google Gemini ---
async def process_query_realtime(query: str, argo_df: pd.DataFrame):
    """
    Processes a user query using Google Gemini AI to understand natural language
    and extract structured information for visualization or textual response.
    """
    load_dotenv()
    
    # 1. Initialize the Google Gemini model and the JSON output parser
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    parser = JsonOutputParser(pydantic_object=ArgoQueryResponse)
    
    # 2. Create a more detailed prompt template to guide the AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert oceanographic data assistant. Your task is to analyze a user's query and a provided data sample to extract key information. You must respond ONLY with a valid JSON object based on the format instructions provided.\n{format_instructions}"),
        ("human", """Analyze the user query below to determine the user's intent based on the sample data context.
- **Intent Detection is Critical**: First, determine if the user wants a visual (graph/map) or a text answer.
- If the user explicitly asks for a 'map', 'locations', or 'positions', set 'visualization_type' to 'plot_map'.
- If the user asks to 'plot', 'graph', 'chart', or 'show a graph' of a variable (like temperature or salinity), set 'visualization_type' to 'plot_profile'. If they ask for a 'bar chart', use 'bar'. If they ask for a 'pie chart', use 'pie'.
- For ALL other questions asking for information (e.g., 'what is the temperature?', 'tell me the salinity', 'how many floats are there?', 'temperature for 13857'), you MUST set 'visualization_type' to 'text_only'.
- For simple greetings (e.g., 'hi', 'hello'), provide a friendly 'text_response' and set 'visualization_type' to 'text_only', leaving other fields empty.

- **Data Extraction is Key**:
- If the query contains a specific float ID (e.g., a number like '13857' or '1901345'), extract that ID and place it in the 'wmo_ids' list. This is the most important rule.
- If the query mentions 'temperature', 'temp', or 'heat', include 'TEMP' in the 'variables' list.
- If the query mentions 'salinity' or 'salt', include 'PSAL' in the 'variables' list.

- **Response Generation**:
- For 'text_only' requests that include a variable and a float ID, formulate a response like: "Here is the data for float 13857:". The program will add the specific values.
- If you cannot find the requested float ID in the data context, the text_response should state that, for example: "I couldn't find a float with the ID 12345 in the available data."

User Query: '{query}'

Sample of available data:
{data_context}

Now, provide the structured JSON response.""")
    ])
    
    # 3. Create the processing chain
    chain = prompt | llm | parser
    
    if argo_df.empty:
        return {
            "text_response": "No Argo float data is currently available in the system. Please add .nc files to the 'data' directory and restart.",
            "map_data": {}, "profile_data": {}, "visualization_type": "text_only"
        }
    
    # 4. Invoke the chain with the query and data context
    data_context = argo_df.to_string()
    try:
        response_json = await chain.ainvoke({
            "query": query, "data_context": data_context, "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        print(f"Error invoking LLM chain: {e}")
        return { "text_response": "Sorry, I had trouble understanding that. Could you please rephrase?", "map_data": {}, "profile_data": {}, "visualization_type": "text_only" }

    
    # 5. Fetch data based on the AI's structured response
    wmo_ids_to_fetch = response_json.get('wmo_ids', [])
    variables_to_plot = response_json.get('variables', [])
    viz_type = response_json.get('visualization_type', 'text_only')

    map_data = {'lat': [], 'lon': [], 'wmo_ids': []}
    profile_data = {}
    data_dir = "data"
    
    additional_text = ""

    for wmo_id in wmo_ids_to_fetch:
        try:
            wmo_id_str = str(wmo_id)
            # Find the specific entry for the float
            matching_rows = argo_df[argo_df['wmo_id'] == wmo_id_str]
            if matching_rows.empty:
                print(f"Warning: WMO ID {wmo_id_str} requested by AI not in index. Skipping.")
                continue
            
            latest_entry = matching_rows.sort_values('date_update', ascending=False).iloc[0]
            file_name = latest_entry['file']
            
            # Populate map data if requested
            if viz_type == 'plot_map':
                map_data['lat'].append(latest_entry['latitude'])
                map_data['lon'].append(latest_entry['longitude'])
                map_data['wmo_ids'].append(wmo_id_str)

            # Process variables for text or plot
            if variables_to_plot:
                profile_path = os.path.join(data_dir, file_name)
                ds = open_argo_file(profile_path)
                
                if ds:
                    with ds:
                        # Handle Temperature
                        if 'TEMP' in variables_to_plot and 'TEMP' in ds:
                            pressure, temp = clean_profile_data(ds['PRES'].values, ds['TEMP'].values)
                            if viz_type == 'text_only' and temp:
                                avg_temp = np.mean(temp)
                                additional_text += f"\n- Average temperature for float {wmo_id}: {avg_temp:.2f}°C"
                            elif viz_type != 'text_only':
                                profile_data.setdefault(wmo_id_str, {})['temp'] = {'pressure': pressure, 'values': temp}
                        
                        # Handle Salinity
                        if 'PSAL' in variables_to_plot and 'PSAL' in ds:
                            pressure, sal = clean_profile_data(ds['PRES'].values, ds['PSAL'].values)
                            if viz_type == 'text_only' and sal:
                                avg_sal = np.mean(sal)
                                additional_text += f"\n- Average salinity for float {wmo_id}: {avg_sal:.2f} PSU"
                            elif viz_type != 'text_only':
                                profile_data.setdefault(wmo_id_str, {})['sal'] = {'pressure': pressure, 'values': sal}
        except Exception as e:
            print(f"Could not fetch or process profile for {wmo_id}: {e}")

    final_text_response = (response_json.get('text_response') or "") + additional_text

    data_files_loaded = len([f for f in os.listdir(data_dir) if f.endswith(".nc")])
    fleet_status = {
        "total_floats": argo_df['wmo_id'].nunique(),  # unique floats = active floats
        "files_loaded": data_files_loaded
    }

            
    # 6. Return the final structured response for the frontend
    return {
        "text_response": final_text_response,
        "map_data": map_data,
        "profile_data": profile_data,
        "visualization_type": viz_type,
        "fleet_status": fleet_status
    }

