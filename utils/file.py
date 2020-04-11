import json

# hàm ghi một mảng ra file txt
def write_file_txt(file_path,data):
    string = "\n".join(data)
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write(string)
    return True

# hàm đọc một file text và cho các line vào mảng 
def read_file_txt(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = f.readlines()
    return [line[0:-1] if line[-1]=="\n" else line for line in data]

# hàm ghi một mảng các dict hoặc 1 dict vào file json
def write_file_json(path,data):
    # data là một mảng các dict hoặc 1 dict
    with open(path, "w",encoding='UTF-8') as file_write:
        json.dump(data, file_write, sort_keys=True)
    return True

# hàm load dữ liệu từ file json
def read_file_json(path):
    # data là một mảng các dict hoặc 1 dict
    with open(path, "r",encoding='UTF-8') as file_read:
        data = json.load(file_read)
    return data