import numpy as np
from utils.normalize import nomarlize_teen_code, nomarlize_text, no_marks

# hàm đọc một file text và cho các line vào mảng 
def read_file_txt(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = f.readlines()
    return [line[0:-1] if line[-1]=="\n" else line for line in data]
    
neg = read_file_txt("sentiment_dicts/neg.txt")
pos = read_file_txt("sentiment_dicts/pos.txt")
nots = read_file_txt("sentiment_dicts/not.txt")

def get_mat_data(model, data, option=1, max_len=50, size_word_emd=300):
    """ 
    Hàm chuyển dữ liệu dạng text đầu vào sang matrận đặc trưng word2vec
    Với những từ không có trong word2vec từ vector của nó sẽ là np.zeros
    Những câu có chiều dài nhỏ hơn max_len thì sẽ được bổ sung thêm các vector np.zeros ở phía sau cho đủ
    
        Parameters: 
            model : model word2vec gensim 
            data (array): mảng 2 chiều, với mỗi hàng có dạng [["token1","token2"],"0"], phần tử thứ 2 của rows là nhãn
            option (int): option là 1 trong 3 số 1,2,3. Nếu là 1 thì không nối thêm vector cảm xúc,
                                                        Nếu là 2 thì nối thêm vector cảm xúc 2 chiều vào word_ebedding
                                                        Nếu là 3 thì nối thêm vector cảm xúc 4 chiều vào word_ebedding
            max_len (int): Số tokens tối đa cho mỗi câu
            size_word_emd (int): Số chiều của vector word embedding

        Returns: 
            X_train (ndarray): kích thước (num_of_sentences*max_len*(size_word_emd+0|2|4))
            Y_train (ndarray): kích thước (num_of_sentences)
    """

    X_train = []
    Y_train = []
    if option == 1:
        for row in data:
            Y_train.append(int(row[1]))
            sent = []
            for word in row[0]:
                ## lấy vector embedding của từ
                try:
                    sent.append(model[word])
                except:
                    sent.append(np.zeros(size_word_emd))
            for i in range(max_len-len(row[0])):
                sent.append(np.zeros(size_word_emd))
            X_train.append(np.array(sent))
    elif option == 2:
        for row in data:
            Y_train.append(int(row[1]))
            sent = []
            sentiment = []
            for word in row[0]:
                ## Lấy vector cảm xúc của từ
                if word in nots:
                    sentiment.append([1,0])
                elif word in pos:
                    sentiment.append([0,1])
                elif word in neg:
                    sentiment.append([0,-1])
                else:
                    sentiment.append([0,0])
                ## lấy vector embedding của từ
                try:
                    sent.append(model[word])
                except:
                    sent.append(np.zeros(size_word_emd))
            
            for i in range(max_len-len(row[0])):
                sent.append(np.zeros(size_word_emd))
                sentiment.append([0,0])
            # Nối vector embedding với 4 chiều cảm xúc
            sent = np.concatenate([sent,sentiment], axis=1)
            X_train.append(sent)
    else:
        for row in data:
            Y_train.append(int(row[1]))
            sent = []
            sentiment = []
            for word in row[0]:
                ## Lấy vector cảm xúc của từ
                if word in nots:
                    sentiment.append([1,0,0,0])
                elif word in pos:
                    sentiment.append([0,1,0,0])
                elif word in neg:
                    sentiment.append([0,0,1,0])
                else:
                    sentiment.append([0,0,0,1])
                ## lấy vector embedding của từ
                try:
                    sent.append(model[word])
                except:
                    sent.append(np.zeros(size_word_emd))
            
            for i in range(max_len-len(row[0])):
                sent.append(np.zeros(size_word_emd))
                sentiment.append([0,0,0,0])
            # Nối vector embedding với 4 chiều cảm xúc
            sent = np.concatenate([sent,sentiment], axis=1)
            X_train.append(sent)

    return np.array(X_train,dtype=np.float32), np.array(Y_train,dtype=np.int32)


def get_mat_data_test(model, data, option=1, max_len=50, size_word_emd=300):
    """ 
    Dành cho dữ liệu test
    Hàm chuyển dữ liệu dạng text đầu vào sang matrận đặc trưng word2vec
    Với những từ không có trong word2vec từ vector của nó sẽ là np.zeros
    Những câu có chiều dài nhỏ hơn max_len thì sẽ được bổ sung thêm các vector np.zeros ở phía sau cho đủ
    
        Parameters: 
            model : model word2vec gensim 
            data (array): mảng 2 chiều, với mỗi hàng có dạng [["token1","token2"],"0"], phần tử thứ 2 của rows là nhãn
            option (int): option là 1 trong 3 số 1,2,3. Nếu là 1 thì không nối thêm vector cảm xúc,
                                                        Nếu là 2 thì nối thêm vector cảm xúc 2 chiều vào word_ebedding
                                                        Nếu là 3 thì nối thêm vector cảm xúc 4 chiều vào word_ebedding
            max_len (int): Số tokens tối đa cho mỗi câu
            size_word_emd (int): Số chiều của vector word embedding

        Returns: 
            X_train (ndarray): kích thước (num_of_sentences*max_len*size_word_emd)
    """
    X_train = []
    if option == 1:
        for row in data:
            sent = []
            for word in row:
                ## lấy vector embedding của từ
                try:
                    sent.append(model[word])
                except:
                    sent.append(np.zeros(size_word_emd))
            for i in range(max_len-len(row)):
                sent.append(np.zeros(size_word_emd))
            X_train.append(np.array(sent))
    elif option == 2:
        for row in data:
            sent = []
            sentiment = []
            for word in row:
                ## Lấy vector cảm xúc của từ
                if word in nots:
                    sentiment.append([1,0])
                elif word in pos:
                    sentiment.append([0,1])
                elif word in neg:
                    sentiment.append([0,-1])
                else:
                    sentiment.append([0,0])
                ## lấy vector embedding của từ
                try:
                    sent.append(model[word])
                except:
                    sent.append(np.zeros(size_word_emd))
            
            for i in range(max_len-len(row)):
                sent.append(np.zeros(size_word_emd))
                sentiment.append([0,0])
            # Nối vector embedding với 4 chiều cảm xúc
            sent = np.concatenate([sent,sentiment], axis=1)
            X_train.append(sent)
    else:
        for row in data:
            sent = []
            sentiment = []
            for word in row:
                ## Lấy vector cảm xúc của từ
                if word in nots:
                    sentiment.append([1,0,0,0])
                elif word in pos:
                    sentiment.append([0,1,0,0])
                elif word in neg:
                    sentiment.append([0,0,1,0])
                else:
                    sentiment.append([0,0,0,1])
                ## lấy vector embedding của từ
                try:
                    sent.append(model[word])
                except:
                    sent.append(np.zeros(size_word_emd))
            
            for i in range(max_len-len(row)):
                sent.append(np.zeros(size_word_emd))
                sentiment.append([0,0,0,0])
            # Nối vector embedding với 4 chiều cảm xúc
            sent = np.concatenate([sent,sentiment], axis=1)
            X_train.append(sent)
    return np.array(X_train,dtype=np.float32)


def load_raw_data(filename):
    """ 
    Hàm đọc dữ liệu từ các file train.crash và test.crash
    
        Parameters: 
            filename (str): đường dẫn tương đối hoặc tuyệt đối đến file dữ liệu

        Returns: 
            data (array): mảng 2 chiều, mỗi hàng trong mảng có dạng ["đây là câu", "label 1 or 0"]
    """
    flag_file = filename.split("/")[-1].split(".")[0]
    if flag_file == "train":
        flag = "train_"
    elif flag_file == "test":
        flag = "test_"
    else:
        print("File dữ liệu phải là train.crash hoặc test.crash")
        return False
    data = []
    tmp_text = "  "
    with open(filename, 'r',encoding='utf-8') as file:
        for line in file:
            if flag in line:
                data.append([tmp_text[:-2],tmp_text[-2]])
                tmp_text = ""
            elif line != "\n":
                tmp_text += line
        data.append([tmp_text[:-2],tmp_text[-1]])
    return data[1:]

def clean_data(data):
    """ 
    Hàm làm sạch dữ liệu: chuẩn hoá dấu câu, teencode, emoj,..., sau đó tách từ
    Dữ liệu sau chuẩn hoá có dạng : [["đây","là","câu"], "label 1 or 0"]
    
        Parameters: 
            data (array): mảng 2 chiều, mỗi hàng trong mảng có dạng ["đây là câu", "label 1 or 0"]
        
    """
    for i, row in enumerate(data):
        data[i][0] = nomarlize_teen_code(row[0]).split()

def clean_data_test(data):
    """ 
    Dành cho dữ liệu test
    Hàm làm sạch dữ liệu test: chuẩn hoá dấu câu, teencode, emoj,..., sau đó tách từ
    Dữ liệu sau chuẩn hoá có dạng : [["đây","là","câu","1"],["đây","là","câu","2"]]
    
        Parameters: 
            data (array): mảng gồm danh sách các câu cần phân loại
        
    """
    result = []
    for i, row in enumerate(data):
        tokens = nomarlize_teen_code(row).split()
        if len(tokens) < 50:
            result.append(tokens)
        else :
            result.append(tokens[0:50])
    return result

def separate_data(data):
    """ 
    Chia dữ liệu thành 2 phần, một phần gồm các câu dài hơn 50 từ, và một phần gồm các câu nhỏ hơn 50 từ
    
        Parameters: 
            data (array): mảng 2 chiều, với mỗi hàng có dạng [["token1","token2"],"0"], phần tử thứ 2 của rows là nhãn

        Returns: 
            long_data (array): tương tự data, nhưng chỉ chứa các câu dài hơn 50 từ
            normal_data (array): tương tự data, nhưng chỉ chứa các câu ngắn hơn 50 từ
    """
    long_data = []
    normal_data = []
    for row in data:
        if len(row[0]) > 50:
            long_data.append(row)
        else:
            normal_data.append(row)
    return long_data, normal_data
