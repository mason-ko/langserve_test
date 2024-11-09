from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

# 한글을 영어로
prompt1 = ChatPromptTemplate.from_template("translates {topic} to English.")
prompt2 = ChatPromptTemplate.from_template(
    "explain {english_word} using oxford dictionary to me in Korean."
)

chain1 = prompt1 | llm | StrOutputParser()

chain2 = (
    {"english_word": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)