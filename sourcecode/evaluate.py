## author: ThanhTuan

def evaluate_model(num_label,y_label,prediction):
    """ 
        Hàm đưa ra các kết quả đánh giá Precision, Recall, F1 và ma trận nhầm lẫn ([TP,FP,TN,FN]) của từng lớp

            Parameters: 
                num_label (int): số lượng lớp
                y_label (array) : nhãn thực tế, có dạng [1,2,3,0,1] với 0,1,2,3,.. là nhãn
                prediction (ndarray): nhãn dự đoán, có dạng [1,2,3,0,1] với 0,1,2,3,.. là nhãn
            Returns: 
                Hàm in ra các kết quả đánh giá và ma trận nhầm lẫn của từng lớp
                text (str): Đoạn text kết quả đánh giá Precision, Recall, F1 và ma trận nhầm lẫn
    """
    assert num_label > max(prediction)
    assert num_label > max(y_label)
    assert len(y_label) == len(prediction)
    confusion_matrix = []
    for i in range(num_label):
        confusion_matrix.append([0,0,0,0]) # confusion_matrix[i] = [TP,FP,TN,FN]
    
    ## Tính toán ma trận nhầm lẫn
    for i,label in enumerate(y_label):
        if label == prediction[i]:
            # Khi dự đoán đúng nhãn thì True Positive của nhãn đó tăng lên 1
            confusion_matrix[label][0] += 1     # confusion_matrix[label] = [TP,FP,TN,FN]
            # Đồng thời, True Negative của các nhãn khác tăng lên 1
            for k in range(num_label):
                if k != label:
                    confusion_matrix[k][2] += 1
        else:
            # Khi nhãn thực tế label bị dự đoán sai thành prediction[i] thì
            # False Negative của nhãn label bị tăng lên 1
            confusion_matrix[label][3] += 1
            # False Positive của nhãn prediction[i] tăng lên 1
            confusion_matrix[prediction[i]][1] += 1
            # True Negative của các nhãn khác tăng lên 1
            for k in range(num_label):
                if k != label and k != prediction[i]:
                    confusion_matrix[k][2] += 1
                    
    ## Tính Precision, Recall, F1 cho từng lớp
    for k in range(num_label):
        # Precision = TP/(TP+FP)
        precision = confusion_matrix[k][0]/(confusion_matrix[k][0]+confusion_matrix[k][1])
        confusion_matrix[k].append(precision)
        # Recall = TP/(TP+FN)
        recall = confusion_matrix[k][0]/(confusion_matrix[k][0]+confusion_matrix[k][3])
        confusion_matrix[k].append(recall)
        # F1
        F1 =  2*precision*recall/ (precision+recall)
        confusion_matrix[k].append(F1)
        # confusion_matrix[label] = [TP, FP, TN, FN, precision, recall, F1]
        
    ## In các thông số:
    text = ""
    for k in range(num_label):
        text = text + "-"*100 + "\n"
        text = text + "+ Kết quả cho nhãn " + str(k) + "\n"
        text = text+" "*30+"Actual Positive"+" "*5+ " Actual Negative " + "\n"
        text = text+" "*10+"Predict Positive"+" "*5+"TP="+str(confusion_matrix[k][0])+" "*15+"FP="+str(confusion_matrix[k][1])+ "\n"
        text = text+" "*10+"Predict Negative"+" "*5+"FN="+str(confusion_matrix[k][3])+" "*15+"TN="+str(confusion_matrix[k][2])+ "\n"
        text = text+"\n\n"
        text = text+" "*5+"- "+"Precision: "+ str(confusion_matrix[k][4]) + "\n"
        text = text+" "*5+"- "+"Recall:    "+ str(confusion_matrix[k][5]) + "\n"
        text = text+" "*5+"- "+"F1:        "+ str(confusion_matrix[k][6]) + "\n"
    print(text)
    return text