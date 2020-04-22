import tensorflow as tf
# from model_tensorflow.models.BiLSTM_SoftMax import BiLSTM_SoftMax
from model_tensorflow.models.LSTM_SoftMax import LSTM_SoftMax
# from model_tensorflow.models.SoftMax import SoftMax
import gensim  

from model_tensorflow.preprocess import clean_data_test, get_mat_data_test

class Sentiment():
    def __init__(self,option=1):
        fileW2V = '../data/embedding/word2vec/baomoi.window2.vn.model.bin'
        self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(fileW2V, binary=True)

        self.option = option
        if option == 1:
            size_word_emd = 300
            num_unit_lstm = 128
            path = "models/LSTM-Softmax-300/LSTM-Softmax.ckpt"
        elif option == 2:
            size_word_emd = 302
            num_unit_lstm = 256
            path = "models/LSTM-Softmax-302-256/LSTM_SoftMax_302_256"
        else:
            size_word_emd = 304
            num_unit_lstm = 256
            path = "models/LSTM-Softmax-304-256/LSTM_SoftMax_304_256"
        self.classify = LSTM_SoftMax(num_unit_lstm=num_unit_lstm,size_word_emd=size_word_emd)
        self.classify.load_model(path)
        
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
        data = get_mat_data_test(self.model_word2vec, data, self.option)
        prediction = self.classify.predict(data)
        result = []
        for i,text in enumerate(list_text):
            label = "posivite" if prediction[i] == 0 else "negative"
            result.append([text,label])
        return result