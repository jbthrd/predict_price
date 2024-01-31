import requests

url = 'http://127.0.0.1:5000/'  # Adjust if your Flask app is running on a different port
data = {
    "name": "Déguisement tigre '",
    "description": "Déguisement de tigre à porter en toute occasion ! Peut aussi servir de Pyjama. (Attention, les fermetures éclaires sont cassées).",
    "etat": "État moyen",
    "category": "Loisirs",
    "subcategoryName": "Déguisements"
}

# Sending a POST request to the Flask application
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Convert the response to JSON format
    response_data = response.json()
    # Print the entire response
    print("Full Response:", response_data)
    # Assuming the response is a list of records, print the translation results for the first record
    if response_data and isinstance(response_data, list):
        first_record = response_data[0]
        print("Translated Name:", first_record.get('en_name', 'N/A'))
        print("Translated Description:", first_record.get('en_description', 'N/A'))
else:
    print(f"Failed to get a valid response, status code: {response.status_code}")