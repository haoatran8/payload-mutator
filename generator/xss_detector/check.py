import pickle
import numpy as np
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
#from utils import GeneSeg  # Bổ sung import cho GeneSeg (để xử lý payload)
import tensorflow as tf
from xss_detector.utils import GeneSeg

# Model path
#model_file = "C:/TEST/generator/xss_detector/file/Conv_model.keras"
model_file = "C:/TEST/generator/xss_detector/file/MLP_model.keras"
#model_file = "C:/TEST/generator/xss_detector/file/LSTM_model.keras"

vec_file = "C:/KLTN/xss_detector/file/word2vec.pickle"

# Load mô hình MLP
detection_model = load_model(model_file)

# Load embeddings từ file word2vec.pickle
with open(vec_file, "rb") as f:
    word2vec_model = pickle.load(f)
    dictionary = word2vec_model["dictionary"]
    reverse_dictionary = word2vec_model["reverse_dictionary"]
    embeddings = word2vec_model["embeddings"]

# Hàm xử lý mẫu dữ liệu để chuyển đổi thành chỉ số và thêm padding
def process_dataset(data, dictionary, max_length=532):
    d_index = []
    for word in data:
        if word in dictionary.keys():
            d_index.append(dictionary[word])
        else:
            d_index.append(dictionary["UNK"])

    # Thêm padding nếu độ dài < max_length
    if len(d_index) < max_length:
        d_index.extend([-1] * (max_length - len(d_index)))

    # Cắt bớt nếu độ dài > max_length
    d_index = d_index[:max_length]

    return d_index

# Hàm chuyển đổi vector từ dạng chỉ số sang word2vec
def vector_to_word2vec(vector, embeddings, reverse_dictionary):
    vector_embedded = []
    for d in vector:
        if d != -1:
            vector_embedded.append(embeddings[reverse_dictionary[d]])
        else:
            vector_embedded.append([0.0] * len(embeddings["UNK"]))
    return np.array(vector_embedded)

# Hàm dự đoán nhãn của mẫu dữ liệu
def check_xss(payload, dictionary, embeddings, reverse_dictionary, detection_model):
    # Tiền xử lý dữ liệu đầu vào
    pl=payload
    data = GeneSeg(pl)  # Sử dụng GeneSeg để tách từ (phụ thuộc vào phương thức của bạn)
    #print("Tien xu ly data: " ,data)
    processed_data = process_dataset(data, dictionary)
    processed_data.reverse()
    processed_data = np.array(processed_data)

    # Chuyển đổi vector thành vector word2vec
    vector_word2vec = vector_to_word2vec(processed_data, embeddings, reverse_dictionary)

    # Thêm một chiều để phù hợp với input của model
    vector_word2vec = np.expand_dims(vector_word2vec, axis=0)
    print(vector_word2vec.shape)
    # Dự đoán nhãn của mẫu dữ liệu
    predicted_prob = detection_model.predict(vector_word2vec)
    
    # Sử dụng argmax để quyết định nhãn (0 hoặc 1)
    predicted_label = np.argmax(predicted_prob)
    print(predicted_label)

    if predicted_label == 1:
        print('XSS Modified Payload\n')
    else:
        print('Normal Modified Payload\n')
    return predicted_label

#Check demo
# Ví dụ dữ liệu
#payload = 'Search%3D%3C/script%3E%3Cimg/%2A%00/src%3D%22worksinchrome%26colon%3Bprompt%26%23x28%3B1%26%23x29%3B%22/%00%2A/onerror%3D%27eval%28src%29%27%3E>'
#payload = '&#14<!--<!--&#14InterferenceString%00%00<!--InterferenceStringInterferenceString>!--&#14>>!--&#14>!-->!-->!--InterferenceString>!--InterferenceString&#14InterferenceString%00>img/src=&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;:top[/al/.source+/ert/`1`>>img/src=&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;:top[/al/.source+/ert/`1`>>img/src=&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;:top[/al/.source+/ert/`1`>>img/src=&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;:top[/al/.source+/ert/`1`>-->-->-->-->-->-->-->-->'
payload = 'Helloword<script.. alert(1)<script> alert(1)</script>'
predicted_label = check_xss(payload, dictionary, embeddings, reverse_dictionary, detection_model)
print(predicted_label)