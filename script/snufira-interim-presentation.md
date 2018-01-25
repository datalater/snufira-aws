
(c) 2018 JMC

**Script For SNUFIRA Capstone Project Interim Presentation**


---

**표지**:

안녕하세요. 이상구 교수님과 현일성 멘토 조교님의 지도를 받으면서 Sentence Embedding 프로젝트를 하고 있는 조민철, 손규빈, 장한솔 팀에서 이번 중간 발표를 맡은 조민철입니다.

**2**:

저희가 맡은 프로젝트 주제는 Sentence Embedding인데, Sentence가 문장이라는 것은 알고 있어도 Embedding이 무엇인지는 잘 모르실 수도 있겠습니다.
그래서 이해를 돕기 위해 word embedding부터 간략하게 설명하겠습니다.
embedding은 한 마디로 특정 개념을 벡터로 표현하는 것입니다.
이때 벡터값은 벡터 스페이스 상에서 하나의 점으로 맵핑되기 때문에 embedding이라는 표현을 사용합니다.

벡터로 표현하는 것은 매우 중요합니다.
왜냐하면 컴퓨터는 계산 가능한 숫자값이 있어야 그 의미를 파악할 수 있기 때문입니다.
가령 "cat"이라는 단어를 컴퓨터에게 이해시키기 위해서 숫자들의 나열인 벡터값으로 표현한 것을 word embedding이라고 하는데, 사람이 생각하는 cat과 dog의 관계를 컴퓨터가 인식할 수 있으려면 cat과 dog의 word vector 값에 그 의미 관계가 적용될 있어야 합니다.

**3**:

word vector를 잘 만들게 되면 놀라운 일이 벌어집니다.
컴퓨터에게 king이 남성에게 쓰는 단어이고 queen이 여성에게 쓰는 단어라는 정보를 주지 않아도 알아서 man과 woman의 관계와 같다는 것을 컴퓨터가 이해할 수 있게 됩니다.
따라서 사람은 정교한 워드 벡터를 만드는 모델을 만들어두고 텍스트 코퍼스 데이터를 대량으로 쏟아붓기만 하면, 컴퓨터가 알아서 정교한 vector representation이 만들 수 있고, 그렇게 되면 단어 간의 의미관계를 컴퓨터가 학습하게 되는 겁니다.

**4**:

저희 프로젝트는 word level에서 한 단계 더 나아가서 sentence vector를 만드는 것입니다.
슬라이드에 나와 있는 것처럼 문장을 정교하게 embedding할 수 있으면, 컴퓨터가 사람이 쓰는 문장을 매우 수준 높게 이해할 수 있게 되는 것입니다.

**5**:

따라서 저희 프로젝트의 목적은 Baseline 논문과 비교해서 Sentence Embedding 모델의 성능을 개선하는 것입니다.

여기서 더 나은 모델의 성능이란,
문장의 feature를 더 풍부하고 더 정확하게 포착하는 sentence vector를 만드는 것을 뜻합니다.
그리고 성능을 측정하는 기준은 NLP 논문에서 공통으로 사용하는 benchmark data를 넣어서 classification이 잘 된다면 더 나은 sentence vector를 만든 것으로 볼 수 있습니다.

**6**:

모델의 성능을 측정하는 방법에 대해서 좀 더 상세히 말씀드리겠습니다.
저희가 하는 NLP 분야의 sentence embedding 모델은 머신 러닝 알고리즘에 input으로 넣을 vector를 얼마나 정교하게 만들었는지를 평가하는 것입니다.
그러므로 모델의 훈련과 평가 방법이 나뉩니다.

먼저 파트 원, sentence embedding 모델링을 보겠습니다.
이 부분은 모델을 훈련하는 부분입니다.
BookCorpus라는 데이터셋을 넣어서 저희가 만들 모델을 통해 sentence vector를 만듭니다.
이때 정교한 sentence vector를 만들기 위해 모델의 weight는 학습됩니다.

여기서 BookCorpus 데이터셋은 소설 데이터로써 7천만 개 문장과 9억 8천개 단어로 이루어진 만큼 그 크기가 매우 방대하여 특정 도메인에 편향될 가능성이 매우 적습니다.
또한 소설인만큼 이야기와 대화, 감정 표현 등 문장의 연속성이 보장되는 데이터이기 때문에 사용했으며, 참고로 이 데이터는 토론토 대학교에서 사용 허가를 받았습니다.

**7**:

파트원에서 학습이 완료되면 파트투로 넘어갑니다.
파트투에서는 파트원에서 학습이 이미 끝난 모델을 사용합니다.
이때 baseline 논문들이 공통적으로 사용한 benchmark 데이터를 사용합니다.
그래야 모델 간의 정당한 비교가 가능하기 때문입니다.
benchmark 데이터를 저희가 학습시킨 모델에 넣어서 sentence vector를 만들고, 그 sentence vector를 아주 간단한 classifier에 넣어서 classification을 합니다.
이때 classification accuracy가 높을수록 더 정교한 sentence embedding 모델이 됩니다.

**8**:

프로젝트를 시작하면서 sentence embedding과 관련한 여러 가지 논문을 읽었는데, 흐름을 잡기 위해 인용 관계를 넣어서 논문맵을 만들어봤습니다.
저희가 다뤘던 논문들은 supervised learning과 unsupervised learning으로 구분해서 슬라이드 하단에 그 모델명을 적어두었습니다.
그 중 저희가 스스로 모델을 고안하기 위해 집중적으로 참고한 논문은 doc2vec, skip-thought, CNN-static, self-attentive sentence embedding입니다.

**9**:

분석 총평은 슬라이드와 같습니다.
저희가 취할 기존 모델의 장점은 unsupervised learning 이라는 점과 CNN 알고리즘을 활용할 경우 pre-trained word vector를 사용하는 것이 더 성능이 높다는 점입니다.
그리고 기존 모델 중에서 SOTA를 달성한 알고리즘 중 문장 간의 관계와 문장 내의 정보를 동시에 추출해서 sentence embedding을 하는 알고리즘이 없다는 것이 개선 가능성이었습니다.

**10**:

따라서 저희가 제안하는 모델은 2가지입니다.
첫 번째 모델은, CBOW skip-thought로 만든 sentence vector에 attention 모델을 추가한 "Attention based CBOW Skip-thought"입니다.
두 번째 모델은, 문장 내 정보를 CNN으로 추출하고, 추출한 feature map을 CNN으로 한 번 더 결합해서 문장 간의 정보까지 추출하는 "Skip-CNN"입니다.

**11**:





---
