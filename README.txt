🌊 FloatChat - AI Conversational Interface for ARGO Data
This project is the MVP implementation of FloatChat, an AI-powered conversational interface designed to democratize access to complex ARGO oceanographic data. It allows users to query oceanographic data using natural language and receive answers in the form of text, interactive maps, and various charts.

This version uses Google's Gemini AI for natural language understanding and loads data directly from local .nc (NetCDF) files.

✨ Features
Conversational AI: Ask questions in plain English (e.g., "show me the temperature for float 13857").

Local Data Processing: Reads ARGO .nc files directly from your computer for fast and offline access.

Intelligent Visualization: The AI determines the best way to display your data.

Text-Based Answers: For direct questions like "what is the average salinity?".

Interactive Maps: For location-based queries like "where are the floats?".

Profile Plots: For depth-based data like "plot temperature vs pressure".

Time-Series Graphs: For time-based queries like "show temperature over time for float 13857".

Toggleable Views: Switch between different visualizations (Map, Profile Plot, Time-Series) for the same query result.

🛠️ Tech Stack & Setup
Backend
Language: Python 3.9+

Framework: FastAPI

AI/ML: LangChain, Google Gemini

Data Libraries: Pandas, Xarray, NetCDF4

Frontend
Technology: HTML, CSS, JavaScript

Visualization: Plotly.js

Setup Steps
Clone the Repository & Set Up Folders:

Create a main project folder (e.g., FloatChat_WebApp).

Inside it, create two folders: backend and frontend.

Inside backend, create an empty folder named data.

Place Your Data:

Copy all of your ARGO .nc files into the backend/data/ folder.

Backend Dependencies & API Key:

Open a terminal in the backend directory.

Create a file named .env and add your Google API key:

GOOGLE_API_KEY="Your-Google-AI-Studio-API-Key"

Install the required Python packages:

pip install "fastapi[all]" uvicorn python-dotenv httpx xarray netcdf4 h5netcdf langchain langchain-google-genai

Frontend Setup:

Save the index.html file into the frontend folder.

Run the Application:

Terminal 1 (Backend): Navigate to the backend folder and run the server.

uvicorn main:app --reload

The server will start, scan your data folder, and become ready at http://localhost:8000.

Terminal 2 (Frontend): You don't need a separate server. Simply open the frontend/index.html file in your web browser.

💬 Example Queries
Simple Greeting: hello

Text-Based Query: what is the average temperature for float 13857?

Map Query: show me the locations of all floats

Profile Plot Query: plot the salinity profile for float 13857

Time-Series Query: show the temperature history for float 13857 over time