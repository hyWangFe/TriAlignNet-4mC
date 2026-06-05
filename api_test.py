import requests

url = "http://127.0.0.1:12345/predict"
payload = {
    "sequence": "CCACGGTGTCTAGAACCGGACGGATTCTTGGTGTTACGGTT"
}

resp = requests.post(url, json=payload)
print(resp.json())