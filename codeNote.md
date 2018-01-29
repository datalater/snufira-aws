

(c) JMC 2018

---

**데이터 파일 디렉토리 찾기**

```python
import os
os.listdir('../data/books_text_full/test/')
data_path = '../data/books_text_full/test/'
filename = '../data/books_text_full/test/13th_Reality-1.txt'
```

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
min_occurence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurence]
print(len(tokens))
# token을 vocab 파일로 저장합니다.
save_list(tokens, 'corpusToLines_vocab.txt')
```

**전체 코퍼스를 문장 단위 리스트로 만들기**

```python
from string import punctuation
from os import listdir
from gensim.models import Word2Vec

# 텍스트 파일의 내용을 변수 text로 리턴하는 함수
def load_doc(filename):
    # read only로 파일을 엽니다.
    file = open(filename, 'r', errors='replace')
    # 모든 텍스트를 읽습니다.
    text = file.read()
    # 파일을 닫습니다.
    file.close()
    return text


def doc_to_lines(doc):
    total_lines = []
    lines = [i for i in doc.splitlines() if i]

    return lines


def process_docs(directory, vocab, is_train):
    total_lines = []
    for filename in listdir(directory):
        path = directory + '/' + filename
        doc = load_doc(path)
        doc_lines = doc_to_lines(doc)

        total_lines += doc_lines

    return total_lines


def save_total_lines(lines, filename):
    data = "\n".join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# 모든 training set을 불러옵니다.
train_docs = process_docs(data_path, vocab, True)
sentences = train_docs
print('Total training sentences: %d' % len(sentences))

save_list(sentences, 'total_lines.txt')
print("tota_lines {}개가 [total_lines.txt]로 저장되었습니다.".format(len(sentences)))
```

**clean_lines**:

```python
def doc_to_clean_lines(filename):
    total_lines = load_doc(filename)
    clean_lines = [i for i in text.splitlines() if len(i) > 3 if "." in i] # 3개 단어 이상으로 이루어지고 마침표가 있는 문장만 포함

    return clean_lines

filename = "total_lines.txt"
clean_lines = doc_to_clean_lines(filename)
save_list(clean_lines, 'clean_lines.txt')
print("문장 {}개가 [clean_lines.txt]로 저장되었습니다.".format(len(clean_lines)))
```



---

**END**
