

(c) JMC 2018

---

**데이터 파일 디렉토리 찾기**

```python
import os
os.listdir('.') # 현재 위치에서 파일 탐색하기
data_path = '../data/books_text_full/test/'
filename = '../data/books_text_full/test/13th_Reality-4.txt'
```

## I. vocabulary 만들기

**define vocabulary**

```python
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# 텍스트 파일의 내용을 변수 text로 리턴하는 함수
def load_doc(filename):
    # read only로 파일을 엽니다.
    file = open(filename, 'r', errors='replace')
    # 모든 텍스트를 읽습니다.
    text = file.read()
    # 파일을 닫습니다.
    file.close()
    return text

def clean_doc(doc):
    # white space 기준으로 tokenize 합니다.
    tokens = doc.split()
    # 각 token에서 모든 구두점을 삭제합니다.
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # 각 token에서 alaphabet으로만 이루어지지 않은 모든 단어를 삭제합니다.
    tokens = [word for word in tokens if word.isalpha()]
    # 각 token에서 stopwrods를 삭제합니다.
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # 각 token에서 1글자 이하인 모든 단어를 삭제합니다.
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# 텍스트 파일을 불러와서 vocab에 추가하는 함수
def add_doc_to_vocab(filename, vocab):
    # 텍스트 파일을 불러옵니다.
    doc = load_doc(filename)
    # 텍스트 파일을 clean toekn으로 리턴합니다.
    tokens = clean_doc(doc)
    # clean token을 vocab에 추가합니다.
    vocab.update(tokens)

# 폴더에 있는 모든 문서를 vocab에 추가하는 함수
def process_docs(directory, vocab, is_train):
    # 폴더에 있는 모든 파일을 순회합니다.
    for filename in listdir(directory):
        # 인덱스가 새겨진 파일 이름과 is_train 인자를 기준으로 test set으로 분류할 모든 파일을 건너뜁니다.
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # 폴더에 있는 파일의 절대 경로를 구합니다.
        path = directory + '/' + filename
        # 텍스트 파일을 불러와서 vocab에 추가하는 함수를 실행합니다.
        add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
    # 각 문장을 하나의 텍스트 일부로 바꿉니다.
    data = '\n'.join(lines)
    # 파일을 쓰기 모드로 엽니다.
    file = open(filename, 'w')
    # 변환한 텍스트를 파일에 씁니다.
    file.write(data)
    # 파일을 닫습니다.
    file.close()

# vocab을 Counter() 객체로 할당합니다.
vocab = Counter()
# 폴더를 지정하고 폴더 내 모든 문서를 vocab에 추가합니다.
process_docs(data_path, vocab, True)
# vocab의 크기를 출력합니다.
print(len(vocab))
# vocab에서 가장 많이 등장한 50개 단어를 출력합니다.
print(vocab.most_common(50))

# token을 min_occurence 기준으로 유지합니다.
min_occurence = 1
tokens = [k for k,c in vocab.items() if c >= min_occurence]
print(len(tokens))
# token을 vocab 파일로 저장합니다.
save_list(tokens, 'corpusToLines_vocab.txt')
print("\n# 단어 {}개의 [corpusToLines_vocab.txt]로 저장했습니다.".format(len(tokens)))

# 보카를 불러옵니다.
vocab_filename = 'corpusToLines_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
print("# 단어 {}개의 [{}]을 [vocab]으로 불러왔습니다.".format(len(vocab), vocab_filename))
```

## II. corpusToLines: 전체 코퍼스를 문장 단위 리스트로 만들기

**total_lines**

```python
from string import punctuation
from os import listdir
from gensim.models import Word2Vec

def load_doc(filename):
    file = open(filename, 'r', errors='replace')
    text = file.read()
    file.close()
    return text

def doc_to_lines(doc):
    total_lines = []
    lines = [i.lower() for i in doc.splitlines() if i]  # 공백 문장 제거 및 모든 문장 소문자 변경

    return lines

def process_directory(data_path):
    total_lines = []
    for filename in listdir(data_path):
        filepath = data_path + '/' + filename
        doc = load_doc(filepath)
        lines = doc_to_lines(doc)
        print(filename, ":", len(lines))
        total_lines += lines
    return total_lines


sentences = process_directory(data_path)
save_list(sentences, 'total_lines.txt')
print("\n# 문장 {}개의 [total_lines.txt]로 저장했습니다.".format(len(sentences)))
filename = 'total_lines.txt'
total_lines = load_doc(filename)
total_lines = [i for i in total_lines.splitlines()]
total_vocab = set()
for i in total_lines:
    total_vocab.update(i)
print("# unique words in [total_lines.txt]: [{}]".format(len(total_vocab)))
```

**clean_lines**:

```python
def doc_to_clean_lines(filename):
    total_lines = load_doc(filename)
    clean_lines = [i.lower() for i in total_lines.splitlines() if len(i) > 5 if "." in i] # 5개 단어 이상으로 이루어지고 마침표가 있는 문장만 포함

    return clean_lines

filename = "total_lines.txt"
clean_lines = doc_to_clean_lines(filename)
save_list(clean_lines, 'clean_lines.txt')
print("# 문장 {}개가 [clean_lines.txt]로 저장되었습니다.".format(len(clean_lines)))
filename = 'clean_lines.txt'
clean_lines = load_doc(filename)
clean_lines = [i for i in clean_lines.splitlines()]
clean_vocab = set()
for i in clean_lines:
    clean_vocab.update(i)
print("# unique words in [clean_lines.txt]: [{}]".format(len(clean_vocab)))
```

**vocab_lines**:

```python
def doc_to_vocab_lines(filename):
    clean_lines = load_doc(filename)
    vocab_lines = []
    for i in clean_lines.splitlines():
        words = i.split()
        words = [word for word in words if word in vocab]
        words = [word for word in words if len(words) >= 5]
        vocab_line = " ".join(words)
        if len(vocab_line):
            vocab_line += "."
            vocab_line = [vocab_line]
            vocab_lines += vocab_line

    return vocab_lines

filename = "clean_lines.txt"
vocab_lines = doc_to_vocab_lines(filename)
save_list(vocab_lines, 'vocab_lines.txt')
print("# 문장 {}개가 [vocab_lines.txt]로 저장되었습니다.".format(len(vocab_lines)))
filename = 'vocab_lines.txt'
vocab_lines = load_doc(filename)
vocab_lines = [i for i in vocab_lines.splitlines()]
vocab_vocab = set()
for i in vocab_lines:
    vocab_vocab.update(i)
print("# unique words in [vocab_lines.txt]: [{}]".format(len(vocab_vocab)))
```

## III. word2vec

**list_lines: word2vec 만들기 전 모든 문장을 token 묶음의 리스트로 만들기**

```python
filename = "vocab_lines.txt"

file = open(filename, 'r', errors='replace')
text = file.read()
file.close()

vocab_lines = [i for i in text.splitlines()]

list_lines = []
for i in vocab_lines:
    i = i.split()
    list_lines.append(i)

print(list_lines[0])
```

**word2vec**

```python
sentences = list_lines
print("Total training sentences:{}".format(len(sentences)))

wv_sz = 100
# word2vec 모델을 훈련시킵니다.
model = Word2Vec(sentences, size=wv_sz, window=5, workers=8, min_count=1)
# 모델의 vocabulary size를 요약합니다.
words = list(model.wv.vocab)
print("Vocabulary size: %d" % len(words))
print("Wordvector size: %d" % (wv_sz))
print("Embedding size: {}x{}".format(len(words), wv_sz))

# 모델을 ASCII 포맷으로 저장합니다.
filename = 'fantasy_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
print("\n# word2vec 파일 [{}]이 저장되었습니다.".format(filename))
```

## IV. Use pre-trained word vector

**encoded_lines to Xtrain**

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocab_lines)
encoded_lines = tokenizer.texts_to_sequences(vocab_lines)
print("-"*60,"# [vocab_lines]가 [encoded_lines]로 인코딩되었습니다.", "-"*60, sep='\n')
print("BEFORE: \n{}".format(vocab_lines[0]))
print("\nAFTER: \n{}".format(encoded_lines[0]))

max_length = max([len(s.split()) for s in vocab_lines])
Xtrain = pad_sequences(encoded_lines, maxlen=max_length, padding='post')
print("\n","-"*60,"# [encoded_lines]가 [Xtrain]으로 패딩되었습니다. (max_length:{})".format(max_length), "-"*60, sep='\n')
print("AFTER: \n{}".format(Xtrain[0]))
```

**load embedding and embedding_layer**

```python
from keras.layers import Embedding

# embedding을 dictionary로 불러옵니다.
def load_embedding(filename):
    # embedding을 메모리에 올려두되 첫번째 줄은 생략합니다.
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    # 단어와 벡터를 연결하는 map을 생성합니다.
    embedding = {}
    for line in lines:
        parts = line.split()  # 1번째 voca의 word vector (100-d)
        # parts[0] : 1번째 voca의 word vector의 1번째 값
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32') # parts[0]이 key값이 되고, 나머지 99개가 value값
    return embedding

# 불러온 embedding을 기준으로 Embedding layer의 weight matrix를 생성합니다.
def get_weight_matrix(embedding, vocab): # vocab = tokenizer_word_index (tokenizer의 vocabulary ex. master:11)
    vocab_size = len(vocab) + 1 # for unknown words
    weight_matrix = np.zeros((vocab_size, 100))
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word) # embedding에서 tokenizer.word_index
    return weight_matrix

vocab_size = len(tokenizer.word_index) + 1  # tokenizer.word_index: 정수-단어 맵핑 딕셔너리

raw_embedding = load_embedding('fantasy_embedding_word2vec.txt')
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

print("-"*60,"# [embedding_layer]가 생성되었습니다.", "-"*60, sep='\n')
print("vocab_size:{} ".format(vocab_size))
```

---

**END**
