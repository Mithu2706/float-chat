🌊 FloatChat - AI Conversational Interface for ARGO Data
FloatChat is a web-based, AI-powered conversational interface designed to make exploring complex ARGO oceanographic data simple and intuitive. Users can ask questions in natural language (e.g., "show temperature profiles for floats in the Indian Ocean") and receive interactive maps and graphs in response.

This application fetches data in real-time and uses a powerful Large Language Model from Perplexity AI to understand user queries, eliminating the need for technical expertise in data analysis.

✨ Features
Natural Language Queries: Ask complex questions in plain English.

AI-Powered: Utilizes Perplexity AI (llama-3-sonar-large-32k-online) for state-of-the-art language understanding.

Real-Time Data: Fetches the latest ARGO float index and profile data directly from the official repository.

Interactive Visualizations: Generates dynamic world maps and depth-profile graphs using Plotly.js.

No API Key Needed (Optional): Can be configured to run in a rule-based mode without an AI provider.

Lightweight Frontend: Built with standard HTML, CSS, and JavaScript for maximum compatibility and speed.

Efficient Backend: Uses FastAPI with an intelligent caching mechanism to ensure fast response times after the initial data load.

🛠️ Tech Stack
Backend:

Python

FastAPI (for the web server)

LangChain (for interacting with the AI model)

Perplexity AI (as the language model provider)

Pandas & Xarray (for data manipulation)

Frontend:

HTML5

CSS3

Vanilla JavaScript

Plotly.js (for charting)

🚀 Getting Started
Follow these instructions to set up and run the FloatChat application on your local machine.

1. Prerequisites
Python 3.9+

Perplexity AI API Key (Required for the AI-powered version)

Sign up and generate a key at Perplexity Labs.

2. Backend Setup
Clone the Repository:

git clone <your-repository-url>
cd <repository-folder>/backend

Create an Environment File:

In the backend directory, create a file named .env.

Add your Perplexity API key to this file:

PERPLEXITY_API_KEY="pplx-YourSecretAPIKeyHere"

Install Python Dependencies:

pip install "fastapi[all]" uvicorn python-dotenv httpx pandas xarray netcdf4 h5netcdf langchain langchain-perplexity

Run the Backend Server:

uvicorn main:app --reload

The backend will start, download the ARGO index file (this might take a minute on the first run), and then be ready at http://localhost:8000. Keep this terminal running.

3. Frontend Setup
Save the Frontend File:

Navigate to the frontend directory.

Ensure you have the index.html file saved there.

Open in Browser:

Simply open the index.html file directly in your web browser (e.g., by right-clicking and selecting "Open with Chrome").

The application is now running! You can start typing queries into the chat interface.

💬 Example Queries
Here are some examples of what you can ask FloatChat:

hi

locate 5 floats position

show me floats in the indian ocean

show temperature and salinity for float 1901345

can you find the position of floats in the atlantic?