import sys
import openai
import json

openai.api_key = "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"

def generate_response(prompt, model_name):
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def load_dataset(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    return data

def convert_to_jsonl(dataset, output_file):
    with open(output_file, "w") as file:
        for item in dataset:
            jsonl_line = {
                "prompt": f"Swinburne Online FAQ Chatbot: Answer the following question: {item['question']}",
                "completion": item['answer']
            }
            file.write(json.dumps(jsonl_line) + "\n")

dataset = load_dataset("test_dataset.json")
convert_to_jsonl(dataset, "test_dataset.jsonl")

#

def chatbot(finetuned_model):
    print("Swinburne Online FAQ Chatbot (type 'exit' to quit)")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            sys.exit(0)
        
        prompt = f"Swinburne Online FAQ Chatbot: Answer the following question: {user_input}"
        chatbot_response = generate_response(prompt, finetuned_model)
        print("Chatbot: " + chatbot_response)

if __name__ == "__main__":
    finetuned_model = "finetuned_model_v1"
    chatbot(finetuned_model)
