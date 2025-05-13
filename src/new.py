from google.colab import drive
drive.mount('/content/drive')

from datasets import load_dataset
import os
from datasets import concatenate_datasets, load_from_disk

dataset_path = "/content/CFIRSTNET2/dataset"
save_base = "/content/drive/MyDrive/saved_data/BeGAN_02_part"
split_name = "BeGAN_02"

batch_size = 100
max_size = 1000  # 예제 수 확인 후 조절

for i in range(0, max_size, batch_size):
    end = min(i + batch_size, max_size)
    print(f"Saving split {i}:{end}")
    part = load_dataset(dataset_path, split=f"{split_name}[{i}:{end}]")
    part.save_to_disk(f"{save_base}{i//batch_size+1}")
    del part

save_base = "/content/drive/MyDrive/saved_data/BeGAN_02_part"
num_parts = 10  # 조각 개수 맞춰서 수정

parts = []
for i in range(1, num_parts + 1):
    part = load_from_disk(f"{save_base}{i}")
    parts.append(part)

full_dataset = concatenate_datasets(parts)
full_dataset.save_to_disk("/content/drive/MyDrive/saved_data/BeGAN_02_full")
