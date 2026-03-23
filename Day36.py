from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output_text
 

def evaluate_response(prompt, response):

    evaluation = {}

    evaluation["response_length"] = len(response.split())


    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())

    overlap = prompt_words.intersection(response_words)

    evaluation["keyword_overlap"] = len(overlap)

    if evaluation["response_length"] > 30:
        evaluation["usefulness"] = "Detailed"
    else:
        evaluation["usefulness"] = "Too short"

    return evaluation
  
prompt = "Explain Artificial Intelligence in simple terms."

print("\nPrompt:\n", prompt)

response = generate_response(prompt)

print("\nAI Response:\n")
print(response)

result = evaluate_response(prompt, response)

print("\nEvaluation Results:\n")
for key, value in result.items():
    print(f"{key}: {value}")