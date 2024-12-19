from torch.utils.data import Dataset
import json
import torch
import re
from argparse import Namespace
from PIL import ImageOps
import io
from torchvision.transforms.functional import resize, pil_to_tensor
from transformers import XLMRobertaTokenizer
import sys
sys.path.append("/home1/kim03/myubai/IMG2TEXT_NAR/clean_narit")
from nat_decoder import NATransformerDecoder
from PIL import Image

my_tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")

def prepare_input(img) -> torch.Tensor:
    # imgs: list of PIL Images
    input_size = [2560, 1920]

    img = img.convert("RGB")
    img = resize(img, input_size)
    img.thumbnail((input_size[1], input_size[0]))
    delta_width = input_size[1] - img.width
    delta_height = input_size[0] - img.height
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    tensor = pil_to_tensor(ImageOps.expand(img, padding)).float()

    # Stack tensors to create a batch
    return tensor

class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data
        args = Namespace()
        NATransformerDecoder.base_architecture(args)
        self.max_target_length = args.max_target_length

    def __getitem__(self, index):
        item = self.data[index]
        gt_raw = str(json.loads(item["ground_truth"])["gt_parse"]["text_sequence"])
        gt_cleaned = re.sub(r'[^\w\s]', '', gt_raw)
        gt_temp = my_tokenizer.encode(gt_cleaned)

        gt = gt_temp[:self.max_target_length]  

        return {
            'pixel_values': prepare_input(item['image']),
            'gt': torch.tensor(gt + [my_tokenizer.pad_token_id for _ in range(self.max_target_length - len(gt))])
            }

    def __len__(self):
        return len(self.data)

class MyTreeStructureDataset(Dataset):

    def __init__(self, data):
        self.data = data
        args = Namespace()
        NATransformerDecoder.base_architecture(args)
        self.max_target_length = args.max_target_length 
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")

    def __getitem__(self, index):
        EOS_NODE_TYPE, EOS_PARENT_IDX, EOS_TOKEN_LENGTH = 5, 101, 91

        item = self.data[index]
        # JSON 형식의 ground_truth에서 트리 정보를 추출
        gt_raw = json.loads(item["ground_truth"])["gt_parse"]
        
        # 트리 구조를 행렬로 변환하는 함수
        gt_matrix = traverse_tree(gt_raw, node_idx=[0])
        gt_matrix.append([EOS_NODE_TYPE, EOS_PARENT_IDX, EOS_TOKEN_LENGTH])
        # 패딩 없이 원래 크기의 행렬 반환
        return {'pixel_values': prepare_input(item['image']), 
                'gt_matrix': gt_matrix}

    def __len__(self):
        return len(self.data)

class MyTreeStructureDataset_AUG(Dataset):
    def __init__(self, data):
        self.data = data
        args = Namespace()
        NATransformerDecoder.base_architecture(args)
        self.max_target_length = args.max_target_length 
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
        
    def __getitem__(self, index):
        target, img = self.data["ground_truth"][index], self.data["image"][index]
        if type(self.data["image"][index])==dict:
            img = Image.open(io.BytesIO(self.data["image"][index]["bytes"]))
        EOS_NODE_TYPE, EOS_PARENT_IDX, EOS_TOKEN_LENGTH = 5, 101, 91
        # 트리 구조를 행렬로 변환하는 함수
        gt_matrix = traverse_tree(target, node_idx=[0])
        gt_matrix.append([EOS_NODE_TYPE, EOS_PARENT_IDX, EOS_TOKEN_LENGTH])
        # 패딩 없이 원래 크기의 행렬 반환
        return {'pixel_values': prepare_input(img), 
                'gt_matrix': gt_matrix}

    def __len__(self):
        return len(self.data["image"])

# 트리를 순회하여 matrix를 생성하는 함수
def traverse_tree(tree, parent_idx=None, node_idx=[0], matrix=None):
    if matrix is None:
        matrix = []

    # 루트 노드를 처리
    if parent_idx is None:  # 루트 노드는 parent_idx가 None
        node_type = determine_node_type(tree)
        token_length = get_token_length(tree)  # 루트의 토큰 길이 계산
        matrix.append([node_type, -1, token_length])  # 루트 노드 추가 (parent_idx=-1로 설정)
        current_node_idx = node_idx[-1]
        node_idx[-1] += 1  # 다음 노드를 위해 인덱스 증가

        # 자식 노드 처리
        if isinstance(tree, dict):
            for key, value in tree.items():
                traverse_tree(value, parent_idx=current_node_idx, node_idx=node_idx, matrix=matrix)
        elif isinstance(tree, list):
            for item in tree:
                traverse_tree(item, parent_idx=current_node_idx, node_idx=node_idx, matrix=matrix)
    else:
        # 자식 노드 처리
        node_type = determine_node_type(tree)
        token_length = get_token_length(tree)  # 자식의 토큰 길이 계산
        matrix.append([node_type, parent_idx, token_length])  # 자식 노드 추가
        current_node_idx = node_idx[-1]
        node_idx[-1] += 1  # 다음 노드를 위해 인덱스 증가

        # 자식 노드가 있으면 순회
        if isinstance(tree, dict):
            for key, value in tree.items():
                traverse_tree(value, parent_idx=current_node_idx, node_idx=node_idx, matrix=matrix)
        elif isinstance(tree, list):
            for item in tree:
                traverse_tree(item, parent_idx=current_node_idx, node_idx=node_idx, matrix=matrix)

    return matrix
    # Node의 type, parent index, token length를 저장하는 행렬

# 노드 타입을 결정하는 함수
def determine_node_type(node):
    if isinstance(node, dict):
        return 0  # ObjectNode
    elif isinstance(node, list):
        return 3  # ArrayNode
    elif isinstance(node, str):
        return 1  # KeyValueNode
    elif isinstance(node, (int, float)):
        return 2  # ValueNode
    return -1  # Unknown

# KeyValueNode와 ValueNode의 값을 기반으로 토큰 길이 계산
def get_token_length(node):
    if isinstance(node, str):
        return len(my_tokenizer.encode(node))
    elif isinstance(node, (int, float)):
        return len(my_tokenizer.encode(str(node)))
    return 0