import tensorflow as tf
from model_tensorflow.evaluate import evaluate_model

class SoftMax():
    """ 
    Model SoftMax dùng cho bài toán phân loại câu gồm 2 lớp tuyến tính và 1 lớp SoftMax
    Model dùng cross_entropy làm hàm mất mát và dùng Gradient Descent để tối ưu tham số
    Model code bằng TensofFlow v1
    """
    def __init__(self, num_class=2, num_unit=100, learning_rate=0.01,weight_decay=0.000001,
                 size_doc_emd=300):
        """ 
        Khởi tạo lớp model với các tham số cơ bản
        
            Parameters: 
                num_class (int): Số lớp cần phân loại
                num_unit (int): Số lượng unit trong lớp tuyến tính đầu tiên
                size_doc_emd (int): Số chiều của vector docs embedding
                learning_rate (float): tỷ lệ hay tốc độ học của model
                weight_decay (float) : trọng số cho thành phần regularization L2 trong loss function

            Returns: 
            SoftMax
        """
        self.num_class = num_class
        self.num_unit = num_unit
        self.size_doc_emd = size_doc_emd
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sess = tf.compat.v1.Session(graph=self.modeling())
        
    def modeling(self):
        """ 
        Hàm khởi tạo Graph cho model, hay có thể nói là khởi tạo mô hình
        """
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            ## Placeholder
            self.pl_X_train = tf.compat.v1.placeholder(tf.float32, shape=[None, self.size_doc_emd])
            self.pl_Y_train = tf.compat.v1.placeholder(tf.int32,shape=[None])
            # xác xuất cho dropout ở cả lớp tuyến tính đầu tiên và trong cell LSTM
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)
            
            ## Variable
            
            # Trọng số và bias cho lớp tuyến tính đầu tiên: size_word_emd*num_unit
            self.ln_w = tf.Variable(tf.random.truncated_normal([self.size_doc_emd, self.num_unit], stddev=1))
            self.ln_b = tf.Variable(tf.zeros([self.num_unit]))
            
            # trọng số và bias cho lớp tuyến tính phân loại cuối
            self.classify_w = tf.Variable(tf.random.truncated_normal([self.num_unit, self.num_class], stddev=1))
            self.classify_b = tf.Variable(tf.zeros([self.num_class]))
            
            
            ## Lớp tuyến tính chỉnh lưu đầu tiên
            X_train = tf.nn.relu(tf.add(tf.matmul(self.pl_X_train, self.ln_w), self.ln_b))  # X_train*ln_w + ln_b
            X_train = tf.nn.dropout(X_train, rate=1-self.keep_prob)
            
            
            ## Lớp tuyến tính cho softmax
            classifier = tf.add(tf.matmul(X_train, self.classify_w), self.classify_b) # sum_fw_bw*classify_w + classify_b
            
            ## Predict
            self.prediction = tf.argmax(tf.nn.softmax(classifier), 1)
            
            ## Chuyển đổi nhãn sang dạng vector onehot: tf_Y_train có dạng [0,2,1,1], với 0,1,2 là nhãn
            self.Y_train = tf.one_hot(self.pl_Y_train,
                                self.num_class,
                                on_value = 1.0,
                                off_value = 0.0,
                                axis = -1)
            ## SoftMax và CrossEntropy
            softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=classifier,labels=self.Y_train)
            self.cross_entropy = tf.reduce_mean(softmax_cross_entropy)
            
            regularization = self.weight_decay * sum(
                tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables() 
                                        if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
            )
            self.cross_entropy = self.cross_entropy + regularization
            
            # Số vòng lặp
            self.global_step = tf.Variable(0, trainable = False)
            self.learning_rate_ = tf.compat.v1.train.exponential_decay(self.learning_rate, 
                                                            self.global_step, 1000, 0.8, staircase = True)
            ## AdagradOptimizer dùng thuật toán Gradient Descent để tối ưu  các tham số
            self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.learning_rate_).minimize(self.cross_entropy,
                                                                                    global_step=self.global_step)
            
            # Tạo đối tượng saver để lưu model
            self.saver = tf.compat.v1.train.Saver()
            # khởi tạo các giá trị seed cho các node/biến Variable
            self.init = tf.compat.v1.global_variables_initializer()
        return self.graph
    
    def load_model(self, path_file = "models/Softmax/Softmax.ckpt"):
        """ 
        Hàm load model từ file
            Parameters: 
                path_file (str) : Đường dẫn đến file ckpt
        """
        self.saver.restore(self.sess, path_file)

    def save_model(self, path_file = "models/Softmax/Softmax.ckpt"):
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
                                X_train: một mảng 2 chiều có dạng num_docs*doc_embedding
                                Y_train: [1,2,1,2,3,1] với 0,1,2,3,.. là nhãn
                loop (int) : số vòng lặp khi train model
                batch_size (int) : số lượng dữ liệu đưa vào trong mỗi lần chạy
        """
        self.sess.run(self.init)
        #loss_list = []
        #accuracy_list = []
        
        num_data = len(data[1])
        num_batch = num_data//batch_size
        min_loss = 1000000
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
                if float(loss) < min_loss:
                    min_loss = float(loss)
            if step%100==0:
                print("Vòng lặp thứ {} và giá trị cross entropy là {}   current_min_cross_entropy {}".format(step,loss,min_loss))
        print("Giá trị cross entropy nhỏ nhất là {}".format(min_loss)) 
        return True
    
    def evaluate(self, data):
        """ 
        Hàm dự đoán nhãn của dữ liệu

            Parameters: 
                data (ndarray) : dữ liệu train có dạng [X_test,Y_test]
                            X_test: một mảng 2 chiều có dạng num_docs*doc_embedding
                            Y_test: [1,2,1,2,3,1] với 0,1,2,3,.. là nhãn
            Returns: 
                Hàm in ra các thông số đánh giá precision, recall, F1
                Trả về index của các trường hợp bị sai
        """
        prediction = self.sess.run(self.prediction, 
                                    feed_dict={
                                      self.pl_X_train: data[0],
                                      self.keep_prob:1.0,
                                      })
        print("Model SoftMax")
        index_mistake = evaluate_model(self.num_class,data[1],prediction)
        # Trả về index của các trường hợp bị sai
        return index_mistake

    def predict(self, data):
        """ 
        Hàm dự đoán nhãn của dữ liệu

            Parameters: 
                data: một mảng 2 chiều có dạng num_docs*doc_embedding
            Returns: 
                prediction (ndarray): có dạng [1,2,3,0,1] với 0,1,2,3,.. là nhãn
        """
        prediction = self.sess.run(self.prediction, 
                                    feed_dict={
                                      self.pl_X_train: data,
                                      self.keep_prob:1.0,
                                      })
        return prediction
    def close(self):
        """
        Đóng tất cả các cài nguyên mà tensor.session của class sử dụng
        """
        self.sess.close()          