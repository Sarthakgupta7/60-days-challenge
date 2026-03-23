from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


llm = ChatOpenAI(model="gpt-4.1-mini")


memory = ConversationBufferMemory()

chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print("\n🤖 AI Chatbot with Memory")
print("Type 'exit' to end the conversation.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nChatbot: Goodbye! 👋")
        break

    response = chatbot.predict(input=user_input)

    print("Bot:", response)