import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import traceback

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
# upstage embedding model

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    client = OpenAI(
    api_key="up_OZFOByunbTwITSBgJGCeW64CesvMo",
    base_url="https://api.upstage.ai/v1/solar"
    )
    batch_embeddings = []
    query_result = client.embeddings.create(
    model = "embedding-query",
    input = sentences
        )
    for query_embedding in query_result.data:
        batch_embeddings.append(query_embedding.embedding)
    return np.array(batch_embeddings).astype('float32')


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)


# Sparse + Dense 앙상블 검색
def ensemble_retrieve(query_str, sparse_size=20, dense_size=3):
    # Sparse 방식으로 20개의 문서 검색
    sparse_results = sparse_retrieve(query_str, sparse_size)

    # Sparse 결과에서 문서 추출
    candidate_docs = [hit["_source"] for hit in sparse_results["hits"]["hits"]]

    # 각 문서의 content에 대한 임베딩 생성
    candidate_contents = [doc["content"] for doc in candidate_docs]
    candidate_embeddings = get_embedding(candidate_contents)

    # 쿼리 임베딩 생성
    query_embedding = get_embedding([query_str])[0]

    # 각 문서와 쿼리 간의 유사도를 계산하여 리랭킹
    similarities = np.dot(candidate_embeddings, query_embedding)
    ranked_indices = np.argsort(similarities)[::-1]  # 유사도 내림차순 정렬

    # 상위 dense_size개의 문서 선택
    top_docs = [candidate_docs[i] for i in ranked_indices[:dense_size]]

    return top_docs


es_username = "elastic"
es_password = "a_vQ1EC_x9an2dIAtoRZ"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.15.2/config/certs/http_ca.crt")

# # Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 4096,
            "index": True,
            "similarity": "l2_norm",
            # brute-force 검색을 위한 인덱싱 구조 변경
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("/home/data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 앙상블 검색 예제
ensemble_results = ensemble_retrieve(test_query, sparse_size=20, dense_size=3)

# 결과 출력 테스트
for rst in ensemble_results:
    print('source:', rst["content"])


# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
llm_model = "gpt-4o"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 인문, 사회, 과학, 공학 등 모든 분야의 지식에 통달한 전문가

## Instructions
- 사용자의 이전 메시지 정보와 주어진 참고 자료를 활용하여 간결한 답변을 생성한다.
- 주어진 검색 결과 정보로 답변할 수 없을 경우, 정보가 부족하다고 대답한다.
- 답변은 한국어로 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 인문, 사회, 과학, 공학 등 모든 분야의 지식에 통달한 전문가

## Instruction
- Classify the last query into question, advisal request or casual conversation

query: 연구의 과정과 결과를 잘 기록해야 하는 이유? // question

연구실에서 무엇인지 파악이 안된 가루를 저울로 옮기는 좋은 방법은? // advisal request

요새 너무 힘드네.. // casual conversation
- 사용자와 여러 번 대화를 나눈 경우, 대화 맥락을 바탕으로 마지막으로 나눈 대화의 의미를 추론한다.
- 사용자가 각종 지식을 주제로 질문을 하거나 설명을 요청하면, 시스템은 검색 API를 호출할 수 있어야 한다.
- 지식과 관련되지 않은 다른 일상적인 대화 메시지에 대해서는 적절한 응답을 생성한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search relevant documents. Call this whenever you got any kind of question or advisal request except casual conversation.",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query which is suitable for document search based on the user messages history. Always write it in Korean."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # 앙상블 검색을 사용하여 검색 결과 추출
        search_results = ensemble_retrieve(standalone_query, sparse_size=20, dense_size=3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i, rst in enumerate(search_results):
            retrieved_context.append(rst["content"])
            response["topk"].append(rst.get("docid", i))  # docid가 없는 경우 인덱스를 사용
            response["references"].append({"content": rst["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            # if idx>10:
            #     break
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("/home/data/no_eval.jsonl", "ensemble_submission_1.csv")