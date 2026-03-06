import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_ai(question):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=question
        )

        return response.output_text

    except Exception as e:
        return f"Error occurred: {e}"


def main():

    print("====================================")
    print("   AI Prompt Response Generator")
    print("====================================\n")

    while True:

        question = input("Ask something about AI (type 'exit' to quit):\n> ")

        if question.lower() == "exit":
            print("\nExiting program...")
            break

        print("\nGenerating response...\n")

        answer = ask_ai(question)

        print("AI Response:\n")
        print(answer)
        print("\n------------------------------------\n")


if __name__ == "__main__":
    main()