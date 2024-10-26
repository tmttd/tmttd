# %% [markdown]
# ### 1. 라이브러리 설치 및 import

# %%
import os
import json
import yaml
import openai
from tqdm import tqdm
import pandas as pd

# %% [markdown]
# ### 2. Configuration 사전 정의

# %%
config_data = {
    "general": {
        "data_path": "../data/", 
        "output_dir": "./",
        "api_key": "your_openai_api_key"  # OpenAI API 키 입력
    },
    "gpt": {
        "model": "gpt-3.5-turbo",  # 사용할 GPT 모델
        "max_tokens": 100,  # 생성할 텍스트의 최대 토큰 수
        "temperature": 0.7,  # 생성의 다양성 조절
        "top_p": 1.0,  # 누적 확률 분포
        "n": 1  # 생성할 응답 수
    },
}

# %%
# 모델의 구성 정보를 YAML 파일로 저장.
config_path = "./config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config_data, file, allow_unicode=True)

# %% [markdown]
# ### 3. Configuration 불러오기

# %%
# 저장된 config 파일을 불러옵니다.
config_path = "./config.yaml"

with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)

# %% [markdown]
# ### 4. 데이터 확인

# %%
data_path = loaded_config['general']['data_path']

# train data 확인
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
train_df.tail()

# %% [markdown]
# ### 5. 텍스트 생성 함수 정의

# %%
def generate_text(prompt, config):
    openai.api_key = config['general']['api_key']  # OpenAI API 키 설정

    response = openai.ChatCompletion.create(
        model=config['gpt']['model'],
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=config['gpt']['max_tokens'],
        temperature=config['gpt']['temperature'],
        top_p=config['gpt']['top_p'],
        n=config['gpt']['n']
    )
    
    return response['choices'][0]['message']['content']

# %% [markdown]
# ### 6. 텍스트 생성 및 결과 저장

# %%
def main(config):
    results = []
    
    # 각 입력에 대해 텍스트 생성
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        prompt = row['text']  # 'text' 컬럼에서 프롬프트 가져오기
        generated_text = generate_text(prompt, config)
        results.append({
            "input": prompt,
            "generated": generated_text
        })

    # 결과 DataFrame으로 변환
    output_df = pd.DataFrame(results)
    result_path = config['general']['output_dir']
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    output_df.to_csv(os.path.join(result_path, "generated_texts.csv"), index=False)

# %%
if __name__ == "__main__":
    main(loaded_config)
