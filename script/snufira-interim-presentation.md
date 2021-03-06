
(c) 2018 JMC

**Script For SNUFIRA Capstone Project Interim Presentation**


---

**표지**:

안녕하세요. 이상구 교수님과 현일성 멘토 조교님의 지도를 받으면서 Sentence Embedding 프로젝트를 하고 있는 조민철, 손규빈, 장한솔 팀에서 이번 중간 발표를 맡은 조민철입니다.

**2 cat과 dog의 의미를 사람이 인식하는 것처럼 잘 표현하는 것이 word embedding**:

저희가 맡은 프로젝트 주제는 Sentence Embedding인데, Sentence가 문장이라는 것은 알고 있어도 Embedding이 무엇인지는 잘 모르실 수도 있겠습니다.
그래서 이해를 돕기 위해 word embedding부터 간략하게 설명하겠습니다.
embedding은 한 마디로 특정 개념을 벡터로 표현하는 것입니다.
이때 벡터값은 벡터 스페이스 상에서 하나의 점으로 맵핑되기 때문에 embedding이라는 표현을 사용합니다.

벡터로 표현하는 것은 매우 중요합니다.
왜냐하면 컴퓨터는 계산 가능한 숫자값이 있어야 그 의미를 파악할 수 있기 때문입니다.
가령, "cat"과 "dog"의 의미를 컴퓨터에게 어떻게 이해시킬 수 있을까요?
사람이 생각하는 cat과 dog의 관계 또는 의미를 컴퓨터가 이해하려면 cat과 dog의 word vector 값이 아주 정교하게 조정되어야 하고, 이러한 작업을 embedding이라고 합니다.

**3 정교한 vector == word 간의 의미를 알아서 학습한다**:

word vector를 잘 만들게 되면 놀라운 일이 벌어집니다.
스페인, 이탈리아 같은 각 나라의 이름이 포함된 텍스트 코퍼스를 컴퓨터에게 던져주고, 정교한 워드 임베딩을 실시하면 컴퓨터는 각 나라와 수도 간의 관계를 **알아서** 학습하게 됩니다.
컴퓨터에게 수도가 어떤 뜻인지 알려주지 않아도 말이죠.
따라서, 어떤 단어에 해당하는 정답을 주지 않더라도 컴퓨터가 스스로 학습할 수 있는 겁니다.
그래서 일본 더하기 중국 빼기 도쿄를 하면 중국의 수도인 베이징이 나오게 됩니다.

**4 beyond word level**:

저희 프로젝트는 word level에서 한 단계 더 나아가서 sentence vector를 만드는 것입니다.
슬라이드에 나와 있는 것처럼 문장을 정교하게 embedding할 수 있으면, 컴퓨터가 사람이 쓰는 문장을 매우 수준 높게 이해할 수 있게 되는 것입니다.

**5 성능 개선의 의미, 정답이 있는 benchmark data**:

저희 프로젝트의 목적은 Baseline 논문과 비교해서 Sentence Embedding 모델의 성능을 개선하는 것입니다.

여기서 더 나은 모델의 성능이란,
문장의 feature를 더 풍부하고 더 정확하게 포착하는 sentence vector를 만드는 것을 뜻합니다.
그리고 성능을 측정하는 기준은 NLP 논문에서 공통으로 사용하는 benchmark data를 넣습니다.
benchmark data는 정답이 나와있습니다.
해당 sentence vector가 정답을 잘 맞춰서 classification 정확도가 높다면 더 나은 sentence vector를 만든 것으로 볼 수 있습니다.

**6 모델 훈련 방법, BookCorpus 소개**:

성능 측정과 실험 방법에 대해 상세한 설명을 하겠습니다.
저희가 만드는 sentence embedding model은 직접적으로 loss를 측정할 수 없습니다.
모델의 평가 방법이 직관적이지 않을 수 있기 때문에 모델의 성능을 측정하는 방법에 대해서 좀 더 상세히 말씀드리겠습니다.

먼저 파트 원, sentence embedding 모델링을 보겠습니다.
이 부분은 모델을 훈련하는 부분입니다.
BookCorpus라는 데이터셋을 넣어서 저희가 만들 모델을 통해 sentence vector를 만듭니다.
이때 정교한 sentence vector를 만들기 위해 모델의 weight는 학습됩니다.

모델의 훈련 데이터로 BookCorpus를 선정한 이유를 말씀드리겠습니다.
BookCorpus 데이터셋은 소설 데이터로써 7천만 개 문장과 9억 8천만개 단어로 이루어진 만큼 그 크기가 매우 방대하여 특정 도메인에 편향될 가능성이 매우 적습니다.
그만큼 실제 현업에서도 다양한 분야에서 이 모델을 사용할 수 있게 됩니다.
또한 소설인만큼 이야기와 대화, 감정 표현 등 문장의 연속성이 보장되는 데이터입니다.
문장의 연속성이 보장되지 않으면 기본적으로 워드 임베딩과 같은 문장간의 관계를 습득하기 어렵습니다.
그래서 문장의 연속성이 보장되는 BookCorpus 데이터를 사용했습니다.
참고로 BookCorpus 데이터는 토론토 대학교에서 사용 계약서를 보내고 직접 사용 허가를 받았음을 알려드립니다.

**7 모델 평가 방법**:

파트원에서 학습이 완료되면 파트투로 넘어갑니다.
파트투에서는 파트원에서 학습이 이미 끝난 모델을 사용합니다.
여기서부터는 모델의 훈련이 이미 끝난 상태이고 모델의 성능을 평가합니다.
이때 baseline 논문들이 공통적으로 사용한 benchmark 데이터를 사용합니다.
그래야 모델 간의 정당한 비교가 가능하기 때문입니다.
benchmark 데이터를 저희가 학습시킨 모델에 넣어서 sentence vector를 만들고, 그 sentence vector를 아주 간단한 classifier에 넣어서 classification을 합니다.
이때 classification accuracy가 높을수록 더 정교한 sentence embedding 모델이 됩니다.

**8 우리 모델의 근거가 되는 베이스라인 논문, 추가질문은 appendix**:

다음은 저희가 새로운 모델을 어떤 근거로 만들었는지 말씀드리겠습니다.

상단의 그림은 저희가 sentence embedding 관련해서 읽은 논문들을 인용관계로 만들어 본 논문 맵입니다.
그 중 저희가 모델을 고안하기 위해 집중적으로 참고한 논문은 doc2vec, skip-thought, CNN-static, self-attentive sentence embedding입니다.
각 논문의 자세한 내용은 추가 질문 주시면 survey 해드리겠습니다.

**9 취할 장점, 개선할 여지**:

분석 총평은 다음과 같습니다.
저희가 취할 기존 모델의 장점은 실제 현업에서는 레이블된 데이터가 턱없이 부족하기 때문에 unsupervised learning을 선택할 것이라는 점과, CNN 알고리즘을 활용할 경우 pre-trained word vector를 사용하는 것이 더 성능이 높다는 점입니다.
그리고 기존 모델 중에서 SOTA를 달성한 알고리즘 중 문장 간의 관계와 문장 내의 정보를 동시에 추출해서 sentence embedding을 하는 알고리즘이 없다는 것이 개선 가능성이었습니다.

**10 2가지 모델 제안**:

따라서 저희가 제안하는 모델은 2가지입니다.
첫 번째 모델은, CBOW skip-thought에 근거해서 attention 모델을 추가한 "Attention based CBOW Skip-thought"입니다.
두 번째 모델은, 문장 내 정보를 CNN으로 추출하고, 추출한 feature map을 CNN으로 한 번 더 결합해서 문장 간의 정보까지 추출하는 "Skip-CNN"입니다.

**11 encoding-decoding**:

먼저 skip-thought를 활용한 attention 모델입니다.
예를 들어서 모델을 설명하겠습니다.
문장이 총 5문장이 있다고 할게요.
그럼 중심문장을 제외하면 앞 문장 2개, 뒷 문장 2개가 남겠죠?
핵심은 주변 문장을 넣어서 중심문장 벡터를 만드는 오토인코더 모델을 학습시키는 것입니다.

차근차근 그림으로 설명하겠습니다.
설명 스티커를 잘 따라와주세요.

[설명1]을 보면,
t-n번째 문장들은 중심문장을 기준으로 앞뒤 문장을 표현한 것입니다.
이때 각 문장을 구성하는 단어에 대한 벡터는 pre-trained word vector를 사용할 수도 있고, random initialize된 값을 사용할 수도 있습니다.
아직 코드를 돌리고 있는 중이라 어떤 것을 사용할지는 성능을 보고 판단할 생각입니다.

이렇게 중심문장의 주변 문장인 context 문장들이 화살표를 통해 encoding layer를 거쳐서 [설명2]에 써 있는 context vector로 만들어집니다.
이때 encoding 방법은 각 벡터의 값을 가중치를 적용하여 element-wise로 계산하는 attention 방법과 biLSTM 방법을 후보로 두고 코드를 돌려본 뒤 어떤 방법을 쓸지 결정할 예정입니다.

[설명3]을 보면,
context 벡터가 화살표를 통해 decoding layer를 거쳐서
어떤 벡터를 만들어냅니다.
이때 decoded vector는 중심문장 vector가 되도록 loss function을 구성합니다.
즉 본 모델을 요약하면, 주변문장이 중심문장을 만들도록 모델의 weight를 학습시키는 것입니다.
학습이 끝나면 encoded vector가 중심문장의 sentence vector가 되고, 이러한 sentence vector는 기존 self-attentive 모델이나 skip-thought 모델보다 더 넓은 wider context를 고려한 embedding을 하게 됩니다.


**12 CNN feature map 간의 유사도 측정, 중심문장 유사도 == 주변문장 유사도**:

두 번째는 skip-CNN 모델입니다.
텍스트 코퍼스에 있는 모든 문장을 넣어줍니다.
각 문장은 word vector로 구성되어 있으며
CNN을 활용할 경우 pre-trained word vector를 쓰는 게 더 성능이 높으므로 pre-trained word vector를 사용합니다.
각 문장은 Convolution layer와 max pooling layer를 통과하여 최종 feature map이 완성됩니다.
그리고 3개씩 1쌍으로 구성하여 유사도를 비교합니다.
가령, 2, 3, 4번째 문장과 25, 26, 27번째 문장의 유사도를 비교합니다.
이때 2번째 문장과 25번째 문장의 코사인 유사도를 평균낸 것이 3번째 문장과 26번째 문장의 코사인 유사도와 비슷해야 한다는 게 loss function입니다.

이러한 loss function은 문장 간 연속성이 존재할 때 그것을 최대한 반영하는 의도입니다.
학습이 끝나면 최대한 문장 간의 연속성을 반영하는 sentence embedding을 하게 됩니다.

**13 베이스라인 논문 성능 검증 및 우리 모델 구현 중**:

다음은 중간 결과입니다.
먼저 Sentence Embedding 논문을 읽으면서 기존 모델의 코드를 직접 실행해보니 대체로 논문의 결과와 비슷하게 나왔습니다.
다음은 방금 제시한 2가지 모델 중 각각 baseline 코드들을 가져와서 필요한 부분 또는 없는 부분을 직접 코드를 구현하거나 수정하는 방식으로 구현 중에 있습니다.

**14 지금까지 배운 내용 공개**:

중간 발표까지 프로젝트를 진행하면서 얻은 학습 포인트는 크게 3가지 입니다.
NLP 분야에서 word embedding과 달리 sentence embedding은 아직까지 합의될 만한 모델이 존재하지 않는데, 그 이유를 크게 3가지 이유로 정리해봤습니다.

첫째, 분산 표현 가설이 잘 들어맞지 않습니다.
그 이유는 한 코퍼스 내에서 완전히 동일한 문장이 등장할 확률은 매우 낮고, 특정 영역의 코퍼스에서는 인접한 문장끼리 의미가 유사하지 않은 경우들도 있기 때문입니다.

둘째, 한 문장이 두 가지 이상의 주제를 담고 있는 복문인 경우 고정된 길이의 벡터에 담아낸다는 것은 논리적으로 성립한다고 보기 어렵습니다.

셋째, 단어와 달리 문장은 맥락이나 어조에 따라 의미가 달라지기는 경우가 많습니다.

**15 향후 계획 Plan A, Plan B**:

향후 계획을 말씀드리겠습니다.

만약 저희가 만든 모델이 성능이 좋게 나온다면, 고안한 모델을 조금씩 수정해가면서 최적의 hyperparameter를 찾고, 그것을 논문으로 간결하게 정리하는 것입니다.

만약 모델의 성능이 baseline에 비해 유의미한 결과가 나오지 않는다면, 리서치했던 논문들을 survey하고, 시도했던 모델들을 비교하면서 특정 layer 구조가 어떤 feature를 잡아내는지에 대한 intuition을 정리하는 것이 final contribution이 될 것입니다.

감사합니다.



---
