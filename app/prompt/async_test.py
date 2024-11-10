from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import asyncio
# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

# 한글을 영어로
prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

chain = prompt | llm | StrOutputParser()

async def run_async():
    result = await chain.ainvoke({"topic": "해류"})
    print("ainvoke 결과:", result[:50], "...")

async def run_async_stream():
    stream = chain.stream({"topic": "지진"})
    print("stream 결과:")
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()

async def run_async_batch():
    topics = ["지구 공전", "화산 활동", "대륙 이동"]
    results = chain.batch([{"topic": t} for t in topics])
    for topic, result in zip(topics, results):
        print(f"{topic} 설명: {result[:50]}...")  # 결과의 처음 50자만 출력

#asyncio.run(run_async())
#asyncio.run(run_async_stream())
asyncio.run(run_async_batch())