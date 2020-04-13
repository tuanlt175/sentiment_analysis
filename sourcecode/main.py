import tensorflow as tf
from sourcecode.models.BiLSTM_SoftMax import BiLSTM_SoftMax
from sourcecode.models.LSTM_SoftMax import LSTM_SoftMax
from sourcecode.models.SoftMax import SoftMax
import gensim  

from sourcecode.preprocess import clean_data_test, get_mat_data_test

class Sentiment():
    def __init__(self):
        fileW2V = '../data/embedding/word2vec/baomoi.window2.vn.model.bin'
        self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(fileW2V, binary=True)
        self.classify = LSTM_SoftMax()
        self.classify.load_model()
        
    def predict(self,list_text):
        """
        Hàm phân loại cảm xúc câu.
            Parameters: 
                list_text (list): một danh sách các câu cần phân loại

            Returns: 
                result (list): danh sách các câu cùng nhãn dự đoán, nó có dạng như bên dưới
                                [['Cái áo này đẹp thật rất đẹp', 'posivite'],
                                 ['Cửa hàng phục vụ không được tốt lắm', 'negative'],
                                 .......]
        """
        data = clean_data_test(list_text)
        data = get_mat_data_test(self.model_word2vec, data)
        prediction = self.classify.predict(data)
        result = []
        for i,text in enumerate(list_text):
            label = "posivite" if prediction[i] == 0 else "negative"
            result.append([text,label])
        return result