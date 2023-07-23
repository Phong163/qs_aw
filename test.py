import json
ds_raw = []  # Khởi tạo biến ds_raw ở mức độ toàn cục
ds_r = []
with open("C:\\Users\\DELL\\Desktop\\qs_aw_vi\\datatrain.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        ds_r.append(json.loads(line.strip()))

def get_all_sentences(ds_r):
    
    for item in ds_r:
        ds_raw_item = item["qa_pairs"]  # Đặt tên biến khác với ds_raw để tránh ghi đè
        ds_raw.extend(ds_raw_item)  # Sử dụng extend để thêm các phần tử vào ds_raw, không ghi đè lên biến
        # Hoặc có thể sử dụng ds_raw += ds_raw_item

get_all_sentences(ds_r)
print(ds_raw)
print("len=", len(ds_raw)) 
 # In ra chiều dài của ds_raw
