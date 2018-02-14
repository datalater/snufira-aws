(c) JMC 2018

**Remarkable quotes**

---

## 01 Efficient Estimation of Word Representations in Vector Space

word를 vector representation 했다. 성능은 word similarity task로 측정했다.
다른 모델에 비해 SOTA 성능을 냈다.

단어를 최소 단위로 보는 이유는 simplicity, robustness, 간단한 모델이 복잡한 모델보다 더 성능이 나은 점 때문이다.

## 02 Distributed Representations of Words and Phrases and their Compositionality

분산 표현으로 단어를 벡터 스페이스로 임베딩을 하면 서로 비슷한 단어를 비슷한 위치에 임베딩하기 때문에 NLP 성능이 잘 나온다.

신경망으로 학습한 word representation은 여러 가지 언어학적 규칙과 패턴을 인코드한다.
대부분의 패턴은 linear translation으로 나타난다.
e.g. "Madrid" - "Spain" + "France" = "Paris"

## 03 Distributed Representations of Sentence and Documents

머신러닝 알고리즘 대부분은 input을 길이가 고정된 feature vector로 나타내야 한다.
대표적인게 BoW이다.
그런데 BoW는 "powerful", "strong", "Paris" 간의 거리가 같게 나온다.
word2vec을 쓰면 BoW와 달리 word order와 word semantics가 보존된다.

doc2vec은 document를 vector로 만드는 것인데 document vector는 해당 문서에 있는 단어를 잘 예측하도록 학습된다.

## 04 Skip-Thought Vectors

책에 나오는 텍스트는 연속성이 있다.
이러한 연속성을 사용해서 encode 된 중심문장으로 이전 문장과 이후 문장을 reconstruct하는 encoder-decoder 모델을 훈련시킨다.
그러면 semantic과 syntactic 성질을 공유하는 문장들끼리 비슷한 sentence 벡터가 된다.

skip-thought vector를 학습시키는 모델은 여러 가지로 변형시킬 수 있다.
(a) deep encoders and decoders (b) larger context windows (c) encoding and decoding paragraphs, (d) other encoders, such as convnets.

## 05 Rethinking Skip-Thought: A Neighborhood based Approach

**Abstract**: skip-thought 모델은 인접한 문장 간의 semantic continuity를 학습하는 모델이다.
skip-thought model에서 neighborhood information을 고려한 skip-thought neighbor model을 제안한다.
neighborhood information은 인접한 문장을 뜻하며, 이러한 인접한 문장들은 weak supervision으로 작용한다.

**Model**: skip-thought 모델은 중심 문장을 encode하고 이전 문장과 다음 문장을 각각 decode하므로 decoder가 2개 필요하다.
반면 skip-thought neighbor 모델은 인접한 문장에 same semantic continuity가 흐른다고 가정해서 decoder를 하나만 사용해서 이전 문장과 다음 문장을 reconstruct 한다.

**Findings**: 성능 측면에서는 대부분 skip-thought 모델만큼 성능이 나오고, target이 하나인 경우는 skip-thought보다 더 잘 나오더라.

## 06 Learning Distributed Representations of Sentences from Unlabelled Data

**Abstract**: unsupervised 방식으로 sentence embedding을 하는 최고의 방법은 알려져 있지 않다.
이 논문은 unsupervised sentence embedding에 속하는 모델들을 체계적으로 비교한다.

**Findings**: 깊고 복잡한 모델은 supervised task에서 잘 작동하고, 얕은 log-linear 모델은 unsupervised task에서 잘 작동한다.
가령 skip-thought, SDAE가 깊고 복잡한 모델에 속하며, FastSent는 얕은 log-linear 모델에 속한다.

**sentence embedding이 중요한 이유**: 개별 단어와 달리 phrase나 sentence야 말로 사람의 일반적인 지식(human-like general world knowledge)을 인코드하고 있고, sentence embedding이야말로 현재 언어 이해에서 빠져있는 핵심 부분이기 때문이다.

**SkiptThought Vectors**: 중심문장은 RNN을 거쳐서 encode 되고, 두 가지 문장으로 decode 된다.
중요한 점은 RNN이 각 타임스텝마다 단일 세트의 업데이트 가중치를 사용하기 때문에 encoder와 decoder는 중심 문장에 있는 단어의 순서에 민감하다는 것이다.
타겟 문장의 각 위치에 대해서, 디코더는 모델의 vocabulary에 대한 softmax distribution을 계산한다.
각 훈련 데이터의 loss는 backpropagate 되어 encoder를 훈련시키고, 훈련된 encoder는 단어의 나열인 문장을 single vector로 맵핑할 수 있게 된다.

**ParagraphVector (doc2vec)**: sentence embedding을 하는 2가지 log-linear 모델이다.
DBOW 모델은 sentence vector s를 학습하는데 s는 문장 S에 속한 단어를 예측하는데 최적화된 softmax distribution을 정의한다.
DM 모델은 연속된 단어 k-gram을 선택하고 setence vector s는 연속된 k개의 단어에 대한 벡터이다.
DM 모델의 벡터 s는 그 다음에 등장할 단어에 대한 softmax prediction이다.

**SDAEs**: Skip-thought는 데이터 텍스트에 semantic continuity가 존재한다고 가정한다. 그러나 이런 가정이 있으면 continuity가 약한 텍스트에는 모델을 쓸 수가 없다.
그래서 이런 한계를 피하고자 denoising autoencoders(DAEs)를 붙인다.
DAEs는 변형에 중요한 feature들로 sentence embedding을 하게 된다.
DAE를 거쳐서 encode 된 데이터는 깊은 신경망으로 classification을 할 때 더 robust한 성능을 보인다.
DAE는 다양한 길이를 가진 문장에 noise 함수를 적용한다.
노이즈 함수는 문장을 구성하는 단어를 삭제하거나 연속된 단어의 순서를 바꾸는 역할을 한다.
그리고 나서 LSTM에 근거한 encoder-decoder 구조를 학습시키는데 다른 점은 denoising objective를 부여해서 corrupted source 문장이 original source 문장을 예측하도록 만든다.
학습된 모델은 새로운 문장을 distributed representation으로 인코드할 수 있다.
SkipThought와 달리 SDAEs는 문장의 연속성이 없는 아무런 순서가 없는 문장에서도 sentence embedding을 할 수 있다.

**FastSent**: SkipThought의 성능이 잘 나오는 것을 보면 인접한 문장의 내용으로 sentence semantics를 유추할 수 있다는 것을 알 수 있다.
SkipThought 모델은 sentence-level에서 distributional hypothesis를 적용한 것이라고 볼 수 있다.
그럼에도 불구하고 많은 깊은 신경망 모델과 마찬가지로 SkipThought는 훈련시키려면 매우 오래 걸린다.
FastSent는 같은 신호를 이용하도록 설계된 간단한 sentence model인데 비용이 매우 저렴하다.
context 문장을 BOW로 나타낸 상태에서 FastSent 모델은 인접한 문장을 BoW로 나타내서 예측한다.




---
