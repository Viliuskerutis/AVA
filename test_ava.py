import requests

url = "http://51.21.224.254:5000/evaluate"
file_path = "C:/Users/kerut/OneDrive/Dokumentai/AVA/AVA/data/images/test/175e9250451079.58d10d89ef851.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())