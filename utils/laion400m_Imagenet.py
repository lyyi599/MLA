import json

# 读取原始JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 写入新的JSON到文件
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 用条目中的"name"字段值替换原键名
def replace_keys_with_names(data):
    new_data = {}
    for key, item in data.items():
        new_key = item.get("name")  # 获取"name"字段作为新的键名
        if new_key:  # 确保"name"字段存在
            new_data[new_key] = item
    return new_data

def process_json(original_file, new_file):
    # 读取数据
    data = read_json_file(original_file)

    # 修改数据结构
    modified_data = replace_keys_with_names(data)

    # 保存到新的文件
    write_json_file(modified_data, new_file)

    print(f"Data has been modified and saved to {new_file}.")

# 指定原始和新文件的路径
original_file_path = '/2T/PycharmProjects/lyy/reweighting/lift/datasets/ImageNet_LT/metrics-LAION400M.json'  # 替换为原始JSON文件的路径
new_file_path = './utils/laion2Imagenet.json'            # 替换为保存新JSON数据的路径

# 调用处理函数
process_json(original_file_path, new_file_path)
