___
**note-taker**: Duy Quang (Bombom21)
**Source**: 
**Time created**: 15:41 06-23-2024
**Tags**:
- #jobs, #jobs/intern
___
# 1. Install
Make changes in : 
- llava/train/train_xformer.py (add path)
- training_scripts/finetune.sh (change path)
- training_scripts/zero3.json (add file zero3.json from llava)
- dockerfile, docker-compose.yml
```bash
git clone https://github.com/Bombomls21/LLM_finetuning.git
```
# 2. Weights

`TakeNote`: sign in huggingface for permission
```bash
#save directory
mkdir ./checkpoints
#download weights
huggingface-cli download Viet-Mistral/Vistral-7B-Chat --local-dir ./checkpoints
huggingface-cli download chitb/LaVy-pretrain --local-dir ./checkpoints
huggingface-cli download chitb/LaVy-instruct --local-dir ./checkpoints
```

# 3. Data
Use data from huggingface: [vi-OCR_VQA]
## a) Sample
![[Pasted image 20240623155851.png]]
[ 
  { "from": "human", 
	"value": "Tổ chức nào đã giao cuốn sách này cho chúng ta?"},
  { "from": "gpt", 
	"value": "AZ Việt Nam" },
  
  { "from": "human", 
	"value": "Đâu là tên đúng của cuốn sách này?" }, 
  { "from": "gpt", 
    "value": "Có Một Ngày, Bố Mẹ Sẽ Già Đi" }, 
  
  { "from": "human", 
    "value": "Ai là người đã viết cuốn sách này?" }, 
  { "from": "gpt", 
    "value": "Nhiều Tác Giả" }, 
  
  { "from": "human", 
    "value": "Tên của nhà xuất bản đứng sau cuốn sách này là gì?" }, 
  { "from": "gpt", 
    "value": "NXB Thế Giới" } 
]

## b) Download
This code downloads images and configures annotations file similar to **LLAVA**
```python
# pip install datasets tqdm -q 
import json
import os
import requests
from datasets import load_dataset
from tqdm import tqdm  # Import tqdm for progress bar

dataset = load_dataset("LR-AI-Labs/vi-OCR_VQA")

image_dir = "vi-OCR_VQA"
os.makedirs(image_dir, exist_ok=True)
start = 20000
end = start
step = 2000

flag = True

while flag:
    anns = []
    counter = 1
    start = end
    end = start + step

    if end >= len(dataset['train']):
        end = len(dataset['train'])
        flag = False

    for i in range(start, end):
        image_url = dataset['train'][i]["image_url"]
        image_id = dataset['train'][i]['image_id'] + '.jpg'
        conservations = dataset['train'][i]["conversations"]
        image_path = os.path.join(image_dir, image_id)

        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            if os.path.exists(image_path):
                sample_info = {'id': dataset['train'][i]['image_id'],
                                'image': image_path,
                                'conservations': conservations}
                anns.append(sample_info)
    

    json_file_path = f"vi-OCR_VQA/vi-OCR_VQA_{start}_{end}.json"
    with open(json_file_path, "w", encoding='utf-8') as jsonfile:
        json.dump(anns, jsonfile, indent=2, ensure_ascii=False)

    print(f"{start}:{end} images downloaded successfully!")
```

Then **Merge**
```python
import os
import json

root = r'path/to/annotations/folder'
for i, file_ann in enumerate(os.listdir(root)):
    json_path = os.path.join(root, file_ann)
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    if i == 0:
        full_data = data
    else:
        full_data.extend(data)
json_file_path = f"annotations_name"
with open(json_file_path, "w", encoding="utf-8") as jsonfile:
    json.dump(full_data, jsonfile, indent=2, ensure_ascii=False)

#### 
root = r'path/of/final/annotations/file'
with open(root, encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
```

Final Annotations can download [here](https://drive.google.com/file/d/1an3zHmuYKsMEPSvdoVD9PJrAJKaOyPKV/view?usp=sharing)
Additional code to change path of image to adapt when using inside **docker container**
```python
import json

with open('path/to/original/annotations', encoding='utf-8') as f:
    data = json.load(f)

for sample in data:
    sample['image'] = 'change/path' + sample['id'] + '.jpg'

with open('path/to/another/annotations', "w", encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, indent=2, ensure_ascii=False)
```

# 4. FineTuning
## 1. Prepare
Please ensure that the following prerequisites are met:
- `docker` installed
- `CUDA container toolkit` installed ([Guide](https://stackoverflow.com/a/77269071))
- Nvidia driver version >= 450.80.02* ([compatibility with CUDA version 11.x](https://docs.nvidia.com/deploy/cuda-compatibility/))
- To make everything work properly, replace only source `data`/`checkpoints` when binding in `docker-compose.yml`

## 2. Start training
Run this command to build docker image
```bash
docker build -t lavy .
```

After creating image, initialize container
```bash
docker compose run src
```

Run the `finetune.sh` script to start finetune process
```bash
bash ./training_scripts/pretrain.sh
```