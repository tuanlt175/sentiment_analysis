# hàm tính độ dài vector
import numpy as np

def cosine(vector_a, vector_b):
    """
    Tính độ tương tự cosin giữa hai vectơ
    :param vector_a: mảng hoăc numpy array a
    :param vector_b: mảng hoăc numpy array b
    :return: cosin
    """
    # chuyển 2 vector thành ma trận, mục đích để tính toán trên 2 ma trận cho đơn giản
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    # tích hai vector là bằng vector1 nhân với ma trận chuyển vị của vector2
    num = float(vector_a * vector_b.T)
    # tính độ dài của một vector, hay tính norm2 của một mảng/vector
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        return cos

def cos_sim(vector_a, vector_b):
    """
    Tính độ tương tự cosin giữa hai vectơ
    :param vector_a: mảng hoăc numpy array a
    :param vector_b: mảng hoăc numpy array b
    :return: sim
    """
    # chuyển 2 vector thành ma trận, mục đích để tính toán trên 2 ma trận cho đơn giản
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    # tích hai vector là bằng vector1 nhân với ma trận chuyển vị của vector2
    num = float(vector_a * vector_b.T)
    # tính độ dài của một vector, hay tính norm2 của một mảng/vector
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim