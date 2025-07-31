# test_api.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

print("--- Starting API Test ---")

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Check if the API key was found
if not api_key:
    print("🔴 ERROR: GEMINI_API_KEY not found in .env file.")
    print("--- Test Failed ---")
    exit()

print("✅ API Key found in .env file.")

try:
    # 3. Configure the Gemini client
    print("⏳ Configuring Gemini API...")
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured successfully.")

    # 4. Make a simple API call to a text-only model
    print("⏳ Generating content with gemini-pro...")
    model = genai.GenerativeModel('models/gemini-1.5-pro') # Using the simple text model for a quick test
    
    response = model.generate_content("Give a one-sentence, friendly hello to a new user.")
    
    # 5. Check and print the response
    if response.text:
        print("✅ API call successful!")
        print("\n--- Gemini's Response ---")
        print(response.text)
        print("-------------------------\n")
        print("🎉 Your API key and connection are working correctly!")
        print("--- Test Passed ---")
    else:
        print("🔴 ERROR: API call succeeded but returned no text.")
        print("--- Test Failed ---")

except Exception as e:
    print(f"🔴 An error occurred during the API test: {e}")
    print("\n--- Common Causes ---")
    print("1. Invalid API Key: Double-check the key in your .env file.")
    print("2. API Not Enabled: Ensure the 'Generative Language API' is enabled in your Google Cloud project.")
    print("3. Billing Issues: If you've used the API before, check your project's billing status.")
    print("--------------------")
    print("--- Test Failed ---")