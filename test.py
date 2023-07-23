""""
{
  "id": 2931434,
  "language": "vi",
  "num_pairs": 59,
  "qa_pairs": [
    {
      "question": "1 kg nghệ tây có giá bao nhiêu?",
      "answer": "Trên thị trường, nghệ tây tốt nhất chất lượng cao có giá khoảng 3000 đô la cho 2 pound (khoảng 1 kg) vì vậy với mức giá đó, một pound có thể có giá lên tới 1830 đô la.",
      "language": "vi"
    }
  ]
}
"""
import json
from datasets import Dataset

def load_mfag_from_file(file_path):
    # Đọc tệp JSONL và chuyển thành một danh sách các mẫu
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    samples = [json.loads(line.strip()) for line in lines]

    # Xử lý dữ liệu và chuyển đổi thành định dạng phù hợp với datasets.Dataset
    dataset_dict = {
        "id": [],
        "language": [],
        "num_pairs": [],
        "domain": [],
        "qa_pairs": []
    }

    for sample in samples:
        dataset_dict["id"].append(sample["id"])
        dataset_dict["language"].append(sample["language"])
        dataset_dict["num_pairs"].append(sample["num_pairs"])
        dataset_dict["domain"].append(sample["domain"])
        qa_pairs = [{"question": pair["question"], "answer": pair["answer"], "language": pair["language"]} for pair in sample["qa_pairs"]]
        dataset_dict["qa_pairs"].append(qa_pairs)

    dataset = Dataset.from_dict(dataset_dict)
    return dataset

# Đường dẫn tới tệp train.jsonl
train_file_path = r"C:\Users\DELL\Desktop\qs_aw_vi\train.jsonl"


# Tải dữ liệu từ tệp train.jsonl và chuyển thành dataset
ds_raw = load_mfag_from_file(train_file_path)

# Kiểm tra thông tin về dataset
print(ds_raw)

