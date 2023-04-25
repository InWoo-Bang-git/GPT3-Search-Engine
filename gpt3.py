import sys
import openai

openai.api_key = "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"

def response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


def chatbot():
    print("Swinburne Online FAQ Chatbot (type 'exit' to quit)")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            sys.exit(0)
        
        prompt = f"Swinburne Online FAQ Chatbot: Answer the following question: {user_input}"
        chatbot_response = response(prompt)
        print("Chatbot: " + chatbot_response)

if __name__ == "__main__":
    chatbot()
