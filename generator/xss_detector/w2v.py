import csv
import pickle
import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import GeneSeg
from gensim.models import Word2Vec

learning_rate = 0.1
vocabulary_size = 3000
batch_size = 128
embedding_size = 128
num_skips = 4
skip_window = 5
num_sampled = 64
num_iter = 5
plot_only = 100
log_dir = "word2vec.log"
plt_dir = "file\\word2vec.png"
vec_dir = "file\\word2vec.pickle"

start = time.time()
words = []
datas = []
with open("data\\xssed.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, fieldnames=["payload"])
# phân đoạn các từ, lưu vào datas
    for row in reader:
        payload = row["payload"]
        word = GeneSeg(payload)
        datas.append(word)
        words += word


# Construct dataset
def build_dataset(datas, words):
    count = [["UNK", -1]]
    counter = Counter(words)
    count.extend(counter.most_common(vocabulary_size - 1))
    vocabulary = [c[0] for c in count]
    data_set = []
    for data in datas:
        d_set = []
        for word in data:
            if word in vocabulary:
                d_set.append(word)
            else:
                d_set.append("UNK")
                count[0][1] += 1
        data_set.append(d_set)
    return data_set

# Xây dưng model w2v
data_set = build_dataset(datas, words)

model = Word2Vec(data_set, vector_size=embedding_size, window=skip_window, negative=num_sampled, epochs=num_iter)
embeddings = model.wv

#Show biểu đồ 
def plot_with_labels(low_dim_embs, labels, filename=plt_dir):
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")
        f_text = "vocabulary_size=%d;batch_size=%d;embedding_size=%d;skip_window=%d;num_iter=%d" % (
            vocabulary_size, batch_size, embedding_size, skip_window, num_iter
        )
        plt.figtext(0.03, 0.03, f_text, color="green", fontsize=10)
    plt.savefig(filename)
    plt.show()

#Giảm chiều vector xuống 2 D
pca = PCA(n_components=2)
plot_words = embeddings.index_to_key[:plot_only]
plot_embeddings = []
for word in plot_words:
    plot_embeddings.append(embeddings[word])
low_dim_embs = pca.fit_transform(plot_embeddings)
plot_with_labels(low_dim_embs, plot_words)

# Save word vector 
def save(embeddings):
    dictionary = {embeddings.index_to_key[i]: i for i in range(len(embeddings.index_to_key))}
    reverse_dictionary = {v: k for k, v in dictionary.items()}
    word2vec = {"dictionary": dictionary, "embeddings": embeddings, "reverse_dictionary": reverse_dictionary}
    with open(vec_dir, "wb") as f:
        pickle.dump(word2vec, f)


save(embeddings) 
end = time.time()
print("Over job in ", end - start)
print("Saved words vec to", vec_dir)