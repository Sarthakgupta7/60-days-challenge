from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOpenAI(model="gpt-4.1-mini")

document = """
Artificial Intelligence (AI) is transforming industries by enabling
machines to analyze large amounts of data, recognize patterns,
and make decisions. Applications of AI include healthcare
diagnostics, fraud detection in finance, recommendation systems
in e-commerce, and autonomous vehicles. As AI systems become
more powerful, ethical concerns such as fairness, transparency,
and responsible use are becoming increasingly important.
"""

prompt = ChatPromptTemplate.from_template(
    """
Summarize the following document in 3 concise bullet points:

Document:
{document}
"""
)

parser = StrOutputParser()

summarization_chain = prompt | llm | parser

summary = summarization_chain.invoke({"document": document})

print("\n=== Original Document ===\n")
print(document)

print("\n=== Generated Summary ===\n")
print(summary)