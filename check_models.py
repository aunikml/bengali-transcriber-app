# check_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load and configure API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("--- Checking for available models for your API key ---")

# List all available models
for m in genai.list_models():
  # Check if the model supports the 'generateContent' method
  if 'generateContent' in m.supported_generation_methods:
    print(f"✅ {m.name}")

print("\n--- End of list ---")
print("Your app should use one of the models marked with ✅ above.")
print("The model 'models/gemini-1.5-pro-latest' should be on this list.")