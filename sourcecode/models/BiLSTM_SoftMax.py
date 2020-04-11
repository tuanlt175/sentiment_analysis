import tensorflow as tf
from sourcecode.evaluate import evaluate_model

class BiLSTM_SoftMax():
    """ 
    Model BiLSTM_SoftMax dùng cho bài toán phân loại câu
    Model dùng cross_entropy làm hàm mất mát và dùng Gradient Descent để tối ưu tham số
    Model code bằng TensofFlow v1
    """
    def __init__(self, sess, num_class=2,num_unit_lstm=128,learning_rate=0.01,weight_decay=0.00001,
                 max_len=50,size_word_emd=300):
        """ 
        Khởi tạo lớp model với các tham số cơ bản
        
            Parameters: 
                sess (tensor.session) : session tensorflow
                num_class (int): Số lớp cần phân loại
                num_unit_lstm (int): Số lượng unit trong mỗi cell LSTM
                max_len (int): Số tokens tối đa cho mỗi câu
                size_word_emd (int): Số chiều của vector word embedding
                learning_rate (float): tỷ lệ hay tốc độ học của model
                weight_decay (float) : trọng số cho thành phần regularization L2 trong loss function

            Returns: 
            Model_BiLSTM_SoftMax
        """
        self.sess = sess
        self.num_class = num_class
        self.num_unit_lstm = num_unit_lstm
        self.max_len = max_len
        self.size_word_emd = size_word_emd
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def modeling(self):
        """ 
        Hàm khởi tạo Graph cho model, hay có thể nói là khởi tạo mô hình
        """
        ## Placeholder
        self.pl_X_train = tf.placeholder(tf.float32, shape=[None, self.max_len, self.size_word_emd])
        self.pl_Y_train = tf.placeholder(tf.int32,shape=[None])
        # xác xuất cho dropout ở cả lớp tuyến tính đầu tiên và trong cell LSTM
        self.keep_prob = tf.placeholder(tf.float32)
        
        ## Variable
        
        # Trọng số và bias cho lớp tuyến tính chỉnh lưu đầu tiên: size_word_emd*num_unit_lstm
        self.ln_w = tf.Variable(tf.truncated_normal([self.size_word_emd, self.num_unit_lstm], stddev=1))
        self.ln_b = tf.Variable(tf.zeros([self.num_unit_lstm]))
        
        # trọng số và bias cho lớp tuyến tính phân loại đầu cuối
        self.classify_w = tf.Variable(tf.truncated_normal([self.num_unit_lstm, self.num_class], stddev=1))
        self.classify_b = tf.Variable(tf.zeros([self.num_class]))
        
        ## Chuyển đổi ma trận đầu vào thành (max_len*None*size_word_emd)
        X_train = tf.transpose(self.pl_X_train, [1, 0, 2])
        # nối tất cả các câu lại thành 1 chuỗi ((max_len*None)*size_word_emd))
        X_train = tf.reshape(X_train, [-1, self.size_word_emd]) 
        
        ## Lớp tuyến tính chỉnh lưu đầu tiên
        X_train = tf.nn.relu(tf.add(tf.matmul(X_train, self.ln_w), self.ln_b))  # X_train*ln_w + ln_b
        X_train = tf.nn.dropout(X_train, self.keep_prob)
        
        # Cắt chuỗi câu trong x_train thành 1 list gồm max_len Tensor([None*num_unit_lstm])
        # Mỗi Tensor([None*num_unit_lstm]) sẽ là đầu vào cho 1 cell LSTM tại 1 timestep, có max_len timestep
        X_train = tf.split(axis = 0, num_or_size_splits = self.max_len, value = X_train)
        
        ## Cell LSTM tiến và lùi
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_unit_lstm, forget_bias = 0.8)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_unit_lstm, forget_bias = 0.8)
        
        ## Dropout cho LSTM
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw_cell,state_keep_prob=self.keep_prob)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw_cell,state_keep_prob=self.keep_prob)
        
        # Lấy ma trận kết quả từ mạng BiLSTM: output_state_fw[1] có dạng Tensor([None*num_unit_lstm])
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn( lstm_fw_cell,
                                                                                             lstm_bw_cell,
                                                                                             X_train,
                                                                                             dtype='float32')
        # cộng kết quả của 2 lớp lại: output_state_fw + output_state_bw
        sum_fw_bw = tf.add(output_state_fw[1], output_state_bw[1]) #Tensor([None*num_unit_lstm])
        
        ## Lớp tuyến tính cho softmax
        classifier = tf.add(tf.matmul(sum_fw_bw, self.classify_w), self.classify_b) # sum_fw_bw*classify_w + classify_b
        
        ## Predict
        self.prediction = tf.argmax(tf.nn.softmax(classifier), 1)
        
        ## Chuyển đổi nhãn sang dạng vector onehot: tf_Y_train có dạng [0,2,1,1], với 0,1,2 là nhãn
        self.Y_train = tf.one_hot(self.pl_Y_train,
                              self.num_class,
                              on_value = 1.0,
                              off_value = 0.0,
                              axis = -1)
        ## SoftMax và CrossEntropy
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=classifier,labels=self.Y_train)
        self.cross_entropy = tf.reduce_mean(softmax_cross_entropy)
        
        regularization = self.weight_decay * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() 
                                    if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.cross_entropy = self.cross_entropy + regularization
        
        # Số vòng lặp
        self.global_step = tf.Variable(0, trainable = False)
        self.learning_rate_ = tf.train.exponential_decay(self.learning_rate, 
                                                        self.global_step, 1000, 0.8, staircase = True)
        ## AdagradOptimizer dùng thuật toán Gradient Descent để tối ưu  các tham số
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate_).minimize(self.cross_entropy,
                                                                                 global_step=self.global_step)
        
        # Tạo đối tượng saver để lưu model
        self.saver = tf.train.Saver()
        # khởi tạo các giá trị seed cho các node/biến Variable
        self.init = tf.global_variables_initializer()
    
    def load_model(self, path_file = "models/BiLSTM-Softmax/BiLSTM-Softmax.ckpt"):
        """ 
        Hàm load model từ file
            Parameters: 
                path_file (str) : Đường dẫn đến file ckpt
        """
        self.saver.restore(self.sess, path_file)

    def save_model(self, path_file = "models/BiLSTM-Softmax/BiLSTM-Softmax.ckpt"):
        """ 
        Hàm lưu model ra file
            Parameters: 
                path_file (str) : Đường dẫn đến file ckpt
        """
        self.saver.save(self.sess, path_file)
        
    def train(self, data, loop, batch_size):
        """ 
        Hàm train model
            Parameters: 
                data (ndarray) : dữ liệu train có dạng [X_train,Y_train]
                                X_train: một mảng 3 chiều có dạng num_docs*max_word_of_sent*word_embedding
                                Y_train: [1,2,1,2,3,1] với 0,1,2,3,.. là nhãn
                loop (int) : số vòng lặp khi train model
                batch_size (int) : số lượng dữ liệu đưa vào trong mỗi lần chạy
        """
        self.sess.run(self.init)
        #loss_list = []
        #accuracy_list = []
        
        num_data = len(data[1])
        num_batch = num_data//batch_size
        for step in range(loop):
            for i in range(num_batch+1):
                start = i*batch_size
                end = start + batch_size
                end = end if end < num_data else (num_data-1)
                if start >= end: 
                    continue
                o,loss = self.sess.run([self.optimizer,self.cross_entropy], 
                                          feed_dict={
                                              self.pl_X_train: data[0][start:end],
                                              self.pl_Y_train: data[1][start:end],
                                              self.keep_prob: 0.8,
                                          })
                #loss_list.append(float(loss))
            if step%1==0:
                print("Vòng lặp thứ {} và giá trị cross entropy là {}".format(step,loss))
                
        return True
    
    def evaluate(self, data):
        """ 
        Hàm dự đoán nhãn của dữ liệu

            Parameters: 
                data (ndarray) : dữ liệu train có dạng [X_test,Y_test]
                            X_test: một mảng 3 chiều có dạng num_docs*max_word_of_sent*word_embedding
                            Y_test: [1,2,1,2,3,1] với 0,1,2,3,.. là nhãn
            Returns: 
                Hàm in ra các thông số đánh giá precision, recall, F1
        """
        prediction = self.sess.run(self.prediction, 
                                    feed_dict={
                                      self.pl_X_train: data[0],
                                      self.keep_prob:1.0,
                                      })
        print("Model BiLSTM + SoftMax")
        evaluate_model(self.num_class,data[1],prediction)

    def predict(self, data):
        """ 
        Hàm dự đoán nhãn của dữ liệu

            Parameters: 
                data: một mảng 3 chiều có dạng num_docs*max_word_of_sent*word_embedding
            Returns: 
                prediction (ndarray): có dạng [1,2,3,0,1] với 0,1,2,3,.. là nhãn
        """
        prediction = self.sess.run(self.prediction, 
                                    feed_dict={
                                      self.pl_X_train: data,
                                      self.keep_prob:1.0,
                                      })
        return prediction
                