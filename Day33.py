from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# Create memory object
memory = ConversationBufferMemory()

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print("AI Chatbot (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    response = conversation.predict(input=user_input)

    print("AI:", response)