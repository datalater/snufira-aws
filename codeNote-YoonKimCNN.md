
(c) JMC 2018

---

# 01 Repository

**Files**

+ data/rt-polaritydata/
    + rt-polarity.neg
    + rt-polarity.pos`
+ data_helpers.py
+ eval.py
+ text_cnn.py
+ train.py
+ README.md

---

**README.md**

Training:

```bash
./train.py
```

Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the eval.py script to load your data.

---

# 02 File Analysis

### data_helpers.py

**clean_str()**

string이 들어오면 string을 clean하는 함수 `clean_str()`를 정의합니다.


```python
import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
```

**load_data_and_labels**

positive_data_file의 path와 negative_data_file의 path가 함수 인자로 들어오면, 파일을 열고 각각 한 줄씩 리스트의 요소로 만듭니다.
한 파일당 하나의 리스트로 구성됩니다.
x_text는 positive와 negative를 모두 합한 리스트인데, x_text의 구성요소는 함수 `clean_str()`를 거칩니다.


```python
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
```

---
