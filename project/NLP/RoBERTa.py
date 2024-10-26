# %% [markdown]
# ### 1. 라이브러리 설치 및 import

# %%
import pandas as pd
import os
import json
import yaml
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import Trainer, TrainingArguments
import wandb

# %% [markdown]
# ### 2. Configuration 사전 정의

# %%
# tokenizer 정의
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# %%
config_data = {
    "general": {
        "data_path": "../data/", 
        "model_name": "roberta-base",
        "output_dir": "./" 
    },
    "tokenizer": {
        "max_len": 512,
        "pad_token": f"{tokenizer.pad_token}",
        "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#'],
    },
    "training": {
        "overwrite_output_dir": True,
        "num_train_epochs": 5,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "seed": 42,
        "logging_dir": "./logs",
        "report_to": "wandb"
    },
    "wandb": {
        "entity": "tmttd",
        "project": "Text Classification",
        "name": "test1"
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
train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
train_df.tail()

# %% [markdown]
# ### 5. 데이터 가공 및 데이터셋 클래스 구축

# %%
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# %% 
# 데이터셋 준비
def prepare_dataset(df, tokenizer, max_len):
    return TextDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

# %% [markdown]
# ### 6. Trainer 및 Trainingargs 구축하기

# %%
# 모델 성능 평가를 위한 메트릭
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'f1': f1
    }

# %% 
# 학습을 위한 trainer 클래스와 매개변수를 정의
def load_trainer_for_train(config, model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        logging_dir=config['training']['logging_dir'],
        report_to=config['training']['report_to']
    )

    wandb.init(
       entity=config['wandb']['entity'],
       project=config['wandb']['project'],
       name=config['wandb']['name'],
   )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer

# %% 
# 학습을 위한 tokenizer와 사전 학습된 모델 불러오기.
def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    model.to(device)
    return model, tokenizer

# %% [markdown]
# ### 7. 모델 학습하기

# %%
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 사용할 모델과 tokenizer 로드.
    model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # train dataset 로드.
    train_dataset = prepare_dataset(train_df, tokenizer, config['tokenizer']['max_len'])

    # Trainer 클래스 로드.
    trainer = load_trainer_for_train(config, model, train_dataset, None)  # validation dataset 추가 가능
    trainer.train()   # 모델 학습을 시작합니다.

    wandb.finish()

# %%
if __name__ == "__main__":
    main(loaded_config)
