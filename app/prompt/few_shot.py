from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings


llm = ChatOllama(model="EEVE-Korean-10.8B:latest")
example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")
examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]

prompt = FewShotPromptTemplate(
    examples=examples,              # 사용할 예제들
    example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
    suffix="질문: {input}",          # 예제 뒤에 추가될 접미사
    input_variables=["input"],      # 입력 변수 지정
)

#print(prompt.invoke({"input": "화성의 표면이 붉은 이유는 무엇인가요?"}).to_string())

#chain = prompt | llm | StrOutputParser()
#print(chain.invoke({"input": "원주율(π)의 정의는 무엇인가요?"}))

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,            # 사용할 예제들
    OpenAIEmbeddings(),  # 임베딩 모델
    Chroma,              # 벡터 저장소
    k=1,                 # 선택할 예제 수
)
question = "화성의 표면이 붉은 이유는 무엇인가요?"
selected_examples = example_selector.select_examples({"question": question})
print(f"입력과 가장 유사한 예제: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
