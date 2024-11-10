from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=200,
    google_api_key=os.getenv("GEMINI_KEY")
)

from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | llm | output_parser

x = chain.invoke({"subject": "popular Korean cusine"})
print(x) #['Kimchi', 'Bibimbap', 'Bulgogi', 'Korean Fried Chicken', 'Japchae']

