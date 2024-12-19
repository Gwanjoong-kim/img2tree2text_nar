import os
from PIL import Image
import pickle
import json
from tqdm import tqdm
from multiprocessing import Process

class CustomDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset["image"][index]
        ground_truth = json.loads(self.dataset["ground_truth"][index])
        return image, ground_truth

# 데이터셋 로드
from datasets import load_dataset, concatenate_datasets
cord_1 = load_dataset("naver-clova-ix/cord-v1", split="train")
cord_2 = load_dataset("naver-clova-ix/cord-v2", split="train")
ds = concatenate_datasets([cord_1, cord_2])
print("data_loaded")

def process_batch(dataset, indices, output_path, cpu_core):
    """
    특정 배치의 데이터를 처리하고, 지정된 CPU에서 실행.
    :param dataset: 데이터셋
    :param indices: 처리할 데이터 인덱스 리스트
    :param output_path: 결과를 저장할 경로
    :param cpu_core: 프로세스가 고정될 CPU 코어 번호
    """
    os.sched_setaffinity(0, {cpu_core})  # 현재 프로세스를 지정된 CPU 코어에 고정
    print(f"Process running on CPU core {cpu_core}")

    img_paths = [
        f'/home1/kim03/myubai/IMG2TEXT_NAR/donut/synthdog/resources/paper/paper_{x}.jpg'
        for x in range(1, 7)
    ]

    for img_path in img_paths:
        background_image = Image.open(img_path).convert("RGB")

        for idx in tqdm(indices, desc=f"Processing indices on CPU {cpu_core}"):
            try:
                image, ground_truth = dataset[idx]
                bounding_boxes = []
                texts = []

                # 원본 이미지와 배경 이미지 크기 조정
                width, height = image.size
                background_image_resized = background_image.resize((width, height)).copy()
                print("image_resized")
                # Ground Truth 파싱
                for line in ground_truth["valid_line"]:
                    for word in line["words"]:
                        bounding_boxes.append(word["quad"])
                print("bounding_boxes_parsed")
                # 바운딩 박스 처리
                for box in bounding_boxes:
                    x1, y1, x3, y3 = int(box["x1"]), int(box["y1"]), int(box["x3"]), int(box["y3"])
                    cropped = image.crop((x1, y1, x3, y3))
                    background_image_resized.paste(cropped, (x1, y1))
                print("image_pasted")
                # 텍스트 저장
                texts.append(ground_truth["gt_parse"])

                # 결과 저장
                os.makedirs(os.path.join(output_path, f"image_{str(img_path)[-5]}"), exist_ok=True)
                os.makedirs(os.path.join(output_path, f"text_{str(img_path)[-5]}"), exist_ok=True)
                background_image_resized.save(os.path.join(output_path, f"image_{str(img_path)[-5]}", f"output_image_{idx}.png"))
                image.save(os.path.join(output_path, f"image_{str(img_path)[-5]}", f"input_image_{idx}.png"))

                with open(os.path.join(output_path, f"text_{str(img_path)[-5]}", f"gt_list_{idx}.pkl"), "wb") as file:
                    pickle.dump(texts, file)

            except Exception as e:
                print(f"Error processing index {idx}: {e}")

def main():
    dataset = CustomDataset(ds)
    output_path = "/home1/kim03/myubai/IMG2TEXT_NAR/clean_narit/sample"

    # 총 데이터 크기와 프로세스 분할
    num_processes = 64
    indices = list(range(len(dataset)))
    split_indices = [
        indices[i::num_processes] for i in range(num_processes)
    ]

    # 멀티프로세싱 실행
    processes = []
    for i, split in enumerate(split_indices):
        p = Process(target=process_batch, args=(dataset, split, output_path, i))
        processes.append(p)
        p.start()

    # 모든 프로세스 종료 대기
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()