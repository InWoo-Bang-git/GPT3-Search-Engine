import requests

def upload_dataset(api_key, dataset_path):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with open(dataset_path, "rb") as file:
        response = requests.post(
            "https://api.openai.com/v1/datasets",
            headers=headers,
            data=file
        )
    return response.json()

api_key = "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"
dataset_path = "data/swinburne_faq_dataset.jsonl"
upload_response = upload_dataset(api_key, dataset_path)
print(upload_response)
