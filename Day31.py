from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# Create Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain the concept of {topic} in simple terms for a beginner."
)
parser = StrOutputParser()

chain = prompt | llm | parser

topic = "Artificial Intelligence"

response = chain.invoke({"topic": topic})

print("\n=== AI Response ===\n")
print(response)