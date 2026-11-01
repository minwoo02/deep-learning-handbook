# Deep Learning Handbook

딥러닝의 개념과 수학적 원리를 이해하고, PyTorch로 직접 모델을 구현하며 학습하는 Handbook입니다.

[문서 사이트](https://minwoo02.github.io/deep-learning-handbook/) · [학습 로드맵](https://minwoo02.github.io/deep-learning-handbook/roadmap/)

## 학습 구성

| 분류 | 내용 |
| --- | --- |
| **Theory** | Foundations & Tasks, Model Architecture, Weight & Initialization, Loss, Gradient / Autograd, Optimizer, Normalization, Regularization, Evaluation |
| **Practice** | PyTorch 기초, 회귀·분류 구현, 기술별 심화 실습, scikit-learn 입문 |

Regression·Classification은 **해결할 문제(Task)**, MLP·CNN·RNN은 **모델 구조(Architecture)** 로 구분합니다. 관련 이론과 실습은 문서 사이의 링크로 연결합니다.

### 권장 학습 순서

PyTorch Basics → Regression → Classification → CNN → RNN

- **Regression**: 수동 gradient → Autograd → SGD → nn.Module → Dataset/DataLoader → ANN 회귀
- **Classification**: XOR 모델 구조 → Iris 다중 분류
- **CNN·RNN**: 구현 실습 준비 중

실습 문서 앞의 구성표에서 Task, Model, nn.Module, Autograd, Loss, Optimizer, Activation을 확인할 수 있습니다.

## 자료 구성

- [Theory](docs/theory/): 개념과 수학적 원리
- [Practice](docs/practice/): 모델 구현과 학습 과정
- [Learning Roadmap](docs/roadmap.md): 학습 순서와 전체 개념 지도

## 주요 학습 문서

| 주제 | 문서 |
| --- | --- |
| 활성화 함수 | [activation-functions.md](docs/theory/architectures/activation-functions.md) |
| Perceptron·MLP·역전파 | [perceptron-mlp-backpropagation.md](docs/theory/architectures/perceptron-mlp-backpropagation.md) |
| 선형회귀부터 ANN | [linear-to-ann-regression.md](docs/practice/regression/linear-to-ann-regression.md) |
| Iris 분류 | [iris-classification-pipeline.md](docs/practice/classification/iris-classification-pipeline.md) |
| 가중치 초기화 | [weight-initialization.md](docs/practice/tech-weight/weight-initialization.md) |

## 문서 작성

문서 수정과 기여 방법은 [문서 작성·관리 가이드](CONTRIBUTING.md)를 참고하세요.

## 저작권

제3자 이미지와 인용의 권리는 원저작자에게 있습니다. 이 저장소 전체에 별도의 포괄적 라이선스를 부여하지 않습니다.
