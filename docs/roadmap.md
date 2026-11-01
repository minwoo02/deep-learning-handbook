# Learning Roadmap

[TOC]

이론을 모두 읽고 나서 실습을 시작할 필요는 없다. **개념을 이해하고 → 코드로 확인하고 → 결과를 해석하는 순서**로 학습한다.

## 전체 학습 흐름

기초 개념 → 신경망 구조 → PyTorch 기초 → Regression → Classification → 학습 개선 → 평가·검증 → CNN·RNN

각 단계에서 **Theory를 먼저 읽고 Practice로 이어간다.** 이미 아는 개념은 건너뛰고, 구현 중 막히는 부분을 이론 문서에서 다시 확인해도 좋다.

## 1. 딥러닝과 문제 유형 이해하기

모델이 해결하려는 문제가 무엇인지, 입력과 출력이 어떤 형태인지 먼저 구분한다.

| Theory · 먼저 읽기 | Practice · 연결하기 |
| --- | --- |
| [딥러닝의 역사와 전체 흐름](theory/foundations/deep-learning-history.md) | 전체 흐름을 익히는 단계이다. |
| [Supervised Learning](theory/foundations/supervised-learning.md) → [Classification과 Regression](theory/foundations/classification-vs-regression.md) | 이후 회귀·분류 실습에서 Task와 입력·출력을 구분하는 기준으로 사용한다. |
| [ANN·DNN·CNN·RNN의 차이](theory/architectures/neural-network-architectures.md) | Task와 Model Architecture를 구분한다. |

## 2. Perceptron에서 MLP로 확장하기

하나의 선형 경계가 갖는 한계와 Hidden Layer·Activation이 필요한 이유를 이해한다.

| Theory · 먼저 읽기 | Practice · 연결하기 |
| --- | --- |
| [TLU와 Perceptron의 기하학적 이해](theory/architectures/perceptron-geometry.md) | [scikit-learn Perceptron](practice/sklearn/sklearn-perceptron.md) |
| [Perceptron·MLP·Backpropagation](theory/architectures/perceptron-mlp-backpropagation.md) → [Activation Function](theory/architectures/activation-functions.md) | [scikit-learn MLPClassifier](practice/sklearn/sklearn-mlp-classifier.md) |
| [XOR와 MLP](theory/architectures/xor-and-mlp.md) → [공간 변환 관점의 신경망](theory/architectures/neural-networks-as-space-transformations.md) | 비선형 문제를 해결하는 구조를 이해하고, 5단계의 PyTorch XOR 모델 분석으로 이어간다. |

scikit-learn 실습은 모델 사용법을 먼저 경험하는 선택 경로이다. PyTorch 구현부터 시작하려면 개념을 읽고 다음 단계로 넘어가도 된다.

## 3. PyTorch와 Parameter 익히기

| Theory · 먼저 읽기 | Practice · 연결하기 |
| --- | --- |
| [Parameter와 Hyperparameter](theory/weight/parameters-and-hyperparameters.md) | [PyTorch Basics](practice/stages/basics.md) → [회귀 종합 실습의 Tensor 기본 연습](practice/regression/linear-to-ann-regression.md#3-tensor) |

Tensor의 생성·형태·연산을 먼저 익힌다. `nn.Module`과 Dataset/DataLoader는 다음 회귀 과정에서 필요한 시점에 도입한다.

## 4. Regression으로 학습 과정을 직접 구현하기

같은 선형회귀 문제에서 시작하여 **gradient 계산과 Parameter 갱신을 하나씩 자동화**한다.

| 순서 | Theory · 먼저 읽기 | Practice · 이어서 하기 |
| --- | --- | --- |
| 1. 수동 gradient | [Gradient Descent와 Learning Rate](theory/gradients/gradient-descent-and-learning-rate.md), [MSE Loss](theory/loss/probabilistic-loss-functions.md) | [수동 gradient](practice/stages/manual.md) |
| 2. Autograd | [Backpropagation과 Chain Rule](theory/architectures/perceptron-mlp-backpropagation.md) | [Autograd로 미분하고 직접 갱신하기](practice/stages/autograd.md) |
| 3. SGD | [SGD](theory/optimizers/sgd.md) | [Optimizer로 Parameter 갱신하기](practice/stages/sgd.md) |
| 4. nn.Module | [Weight·Parameter](theory/weight/parameters-and-hyperparameters.md) | [nn.Module 도입](practice/stages/module.md), [사용자 정의 모델 상세 설명](practice/pytorch-basics/custom-nn-module.md) |
| 5. 데이터 처리 | 입력 Feature와 Target의 역할을 다시 확인한다. | [Dataset/DataLoader 도입](practice/stages/data.md), [데이터 파이프라인 상세 설명](practice/pytorch-basics/datasets-and-dataloaders.md) |
| 6. ANN 회귀 | [Activation Function](theory/architectures/activation-functions.md), [Adam](theory/optimizers/adam.md) | [ANN 회귀 학습](practice/stages/ann.md) |

마지막에는 [회귀 평가 지표](theory/evaluation/regression-metrics.md)를 읽고 예측 오차를 해석한다. 더 큰 데이터셋에 적용해 보고 싶다면 [California Housing 회귀](practice/sklearn/california-housing-regression.md)로 확장할 수 있다.

Loss 문서의 확률적 유도는 처음부터 모두 이해하지 않아도 된다. 먼저 MSE가 무엇을 측정하는지 익히고, 아래의 확률·통계 경로에서 수학적 배경을 보충한다.

## 5. Classification으로 출력과 Loss의 차이 이해하기

회귀에서 익힌 학습 흐름을 재사용하며, 이진 분류와 다중 분류에서 출력과 Loss가 어떻게 달라지는지 확인한다.

| Theory · 먼저 읽기 | Practice · 이어서 하기 |
| --- | --- |
| [Classification과 Regression](theory/foundations/classification-vs-regression.md) → [XOR와 MLP](theory/architectures/xor-and-mlp.md) | [XOR 모델 구조 분석](practice/stages/xor.md): 계층과 forward 연산을 살펴본다. |
| [Bernoulli Distribution](theory/foundations/bernoulli-distribution.md), [Binary Cross-Entropy](theory/loss/probabilistic-loss-functions.md) | 이진 분류의 확률·출력·Loss 관계를 익힌다. XOR 문서는 모델 정의까지 다룬다. |
| [Categorical Distribution](theory/foundations/categorical-distribution.md), [Cross-Entropy Loss](theory/loss/probabilistic-loss-functions.md) | [Iris 기본 MLP 학습](practice/classification/iris-mlp-training.md) → [Dataset부터 학습·평가까지](practice/classification/iris-classification-pipeline.md) |
| [분류 평가 지표](theory/evaluation/classification-metrics.md) | Iris 실습의 예측값과 평가 결과를 해석한다. |

## 6. 학습을 개선하는 기술 익히기

새로운 모델을 계속 만들기보다, 회귀·분류에서 익힌 학습 과정을 바탕으로 각 기술의 역할과 적용 위치를 구분한다.

| 개념 | Theory · 이해할 내용 | Practice · 구현 연결 |
| --- | --- | --- |
| Weight & Initialization | [Parameter](theory/weight/parameters-and-hyperparameters.md), [초기화](theory/weight/initialization.md) | [Xavier·He·LeCun Initialization](practice/tech-weight/weight-initialization.md) |
| Optimizer | [SGD](theory/optimizers/sgd.md) → [Adam](theory/optimizers/adam.md) → [AdamW](theory/optimizers/adamw.md) | [AdamW의 원리와 구현](practice/tech-optimizers/adamw.md) |
| Learning Rate | [Gradient Descent와 Learning Rate](theory/gradients/gradient-descent-and-learning-rate.md) | [Scheduler·Warmup·Cosine Annealing](practice/tech-optimizers/learning-rate-schedulers.md) |
| Normalization | [Normalization](theory/normalization/index.md) | [Batch Normalization의 원리와 구현](practice/tech-normalization/batch-normalization.md) |
| Gradient 처리 | [Gradient와 Backpropagation](theory/architectures/perceptron-mlp-backpropagation.md) | [Gradient Clipping](practice/tech-gradients/gradient-clipping.md) |
| Regularization | [Overfitting·L1/L2·Dropout·Early Stopping](theory/regularization/regularization.md) | [California Housing의 규제·Early Stopping 설정](practice/sklearn/california-housing-regression.md), [AdamW의 Weight Decay](practice/tech-optimizers/adamw.md) |

일부 문서는 원리와 구현을 함께 다룬다. 개념별 안내 페이지에서 설명을 확인한 뒤 연결된 구현 예제를 읽으면 된다.

## 7. 평가·검증·튜닝으로 학습 과정 완성하기

| Theory · 먼저 읽기 | Practice · 확인할 내용 |
| --- | --- |
| [회귀 평가 지표](theory/evaluation/regression-metrics.md) | 회귀 예측의 오차를 MAE·MSE·RMSE·R²로 해석한다. [California Housing 실습](practice/sklearn/california-housing-regression.md)에서 계산 예제를 확인한다. |
| [분류 평가 지표](theory/evaluation/classification-metrics.md) | [Iris 학습·평가](practice/classification/iris-classification-pipeline.md)의 결과를 해석한다. |
| [Cross Validation](theory/evaluation/cross-validation.md) | 문서 안의 Hold-out·K-Fold·Stratified K-Fold 예제를 확인한다. |
| [Hyperparameter Tuning](theory/evaluation/hyperparameter-tuning.md) | Parameter 학습과 Hyperparameter 선택의 차이를 정리한다. |

## 8. CNN·RNN으로 모델 구조 확장하기

| Theory · 먼저 읽기 | Practice · 다음 과정 |
| --- | --- |
| [CNN 개념 안내](theory/architectures/cnn.md), [신경망 구조 비교](theory/architectures/neural-network-architectures.md) | [CNN 구현 실습 · 준비 중](practice/stages/cnn.md) |
| [RNN 개념 안내](theory/architectures/rnn.md), [신경망 구조 비교](theory/architectures/neural-network-architectures.md) | [RNN 구현 실습 · 준비 중](practice/stages/rnn.md) |

## 확률·통계 보충 경로

Loss와 Regularization의 수학적 근거를 더 깊게 이해하고 싶을 때 읽는 경로이다. 회귀·분류를 구현하면서 필요한 주제를 골라 읽어도 된다.

1. [확률분포와 PMF·PDF·CDF](theory/foundations/probability-distributions.md)
2. [Bernoulli Distribution](theory/foundations/bernoulli-distribution.md) → [Categorical Distribution](theory/foundations/categorical-distribution.md)
3. [Laplace Distribution과 MAE·L1](theory/foundations/laplace-distribution.md)
4. [Likelihood·MLE·MAP](theory/foundations/likelihood-and-estimation.md) → [Bayes' Theorem](theory/foundations/bayes-theorem.md)
5. [Loss의 확률적 유도](theory/loss/probabilistic-loss-functions.md) → [Regularization](theory/regularization/regularization.md)

## 개념 지도

목차는 개념의 분류를, 위 학습 경로는 읽는 순서를 나타낸다. 개념을 누르면 관련 문서로 이동한다.

<nav class="handbook-concept-map" aria-label="개념별 관련 문서">
<ul><li><a href="index.md">Deep Learning Handbook</a><ul><li><a href="theory/foundations/index.md">Theory</a><ul><li><a href="theory/foundations/index.md">Foundations &amp; Tasks</a><ul><li><a href="theory/foundations/classification-vs-regression.md">Regression</a></li><li><a href="theory/foundations/supervised-learning.md">Classification</a></li></ul></li><li><a href="theory/architectures/index.md">Model Architecture</a><ul><li><a href="theory/architectures/mlp.md">MLP</a></li><li><a href="theory/architectures/cnn.md">CNN</a></li><li><a href="theory/architectures/rnn.md">RNN</a></li></ul></li><li><a href="theory/weight/index.md">Weight &amp; Initialization</a></li><li><a href="theory/loss/index.md">Loss</a></li><li><a href="theory/gradients/index.md">Gradient / Autograd</a></li><li><a href="theory/optimizers/index.md">Optimizer</a><ul><li><a href="theory/optimizers/sgd.md">SGD</a></li><li><a href="theory/optimizers/adam.md">Adam</a></li><li><a href="theory/optimizers/adamw.md">AdamW</a></li></ul></li><li><a href="theory/normalization/index.md">Normalization</a></li><li><a href="theory/regularization/index.md">Regularization</a></li><li><a href="theory/evaluation/index.md">Evaluation</a></li></ul></li><li><a href="practice/stages/basics.md">Practice</a><ul><li><a href="practice/stages/basics.md">PyTorch Basics</a></li><li><a href="practice/regression/index.md">Regression</a><ul><li><a href="practice/stages/manual.md">Manual gradient</a></li><li><a href="practice/stages/autograd.md">Autograd</a></li><li><a href="practice/stages/sgd.md">SGD</a></li><li><a href="practice/stages/module.md">nn.Module</a></li><li><a href="practice/stages/data.md">Dataset / DataLoader</a></li><li><a href="practice/stages/ann.md">ANN regression</a></li></ul></li><li><a href="practice/classification/index.md">Classification</a><ul><li><a href="practice/stages/xor.md">XOR model</a></li><li><a href="practice/classification/iris-classification-pipeline.md">Iris multiclass</a></li></ul></li><li><a href="practice/stages/cnn.md">CNN · 예정</a></li><li><a href="practice/stages/rnn.md">RNN · 예정</a></li></ul></li></ul></li></ul>
</nav>
