import requests

url = 'http://localhost:3000/generate'

request = {
    "prompt": "Once upon a time, lily found",
    "max_tokens": 30
}

result = requests.post(url, json=request)

elapsed_time = result.elapsed
response_time_seconds = elapsed_time.total_seconds()

print(f"Response time: {response_time_seconds} seconds")
print(f"Result: {result.json()}")