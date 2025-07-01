

import requests
import json

# The URL of the running agent API
API_URL = "http://localhost:8001/chat"

# The data to send in the request
payload = {
    "message": "Hay hueco en el gimnasio para el viernes?",
    "thread_id": "python-test-01"
}

# Set the headers to indicate we are sending JSON
headers = {
    "Content-Type": "application/json"
}

def test_agent_api():
    """
    Sends a request to the agent API and prints the response.
    """
    print(f"▶️  Sending request to: {API_URL}")
    print(f"▶️  Payload: {json.dumps(payload, indent=2)}")

    try:
        # Make the POST request
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        # Raise an exception if the request returned an error status code
        response.raise_for_status()

        print("\n✅ Request successful!")
        print(f"Status Code: {response.status_code}")
        
        # Print the JSON response from the server
        print("Response JSON:")
        # Use ensure_ascii=False to correctly print Spanish characters
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    except requests.exceptions.HTTPError as http_err:
        print(f"\n❌ HTTP error occurred: {http_err}")
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.text)
    except requests.exceptions.RequestException as req_err:
        print(f"\n❌ An error occurred while making the request: {req_err}")
    except Exception as err:
        print(f"\n❌ An unexpected error occurred: {err}")

if __name__ == "__main__":
    test_agent_api()

