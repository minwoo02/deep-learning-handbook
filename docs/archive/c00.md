# [DL_c] 딥러닝 구현 및 실습 — 전체 목차 및 학습 가이드

원문: [[DL_c] 딥러닝 구현 및 실습 — 전체 목차 및 학습 가이드](https://velog.io/@haru02/DLc-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B5%AC%ED%98%84-%EB%B0%8F-%EC%8B%A4%EC%8A%B5-%EC%A0%84%EC%B2%B4-%EB%AA%A9%EC%B0%A8-%EB%B0%8F-%ED%95%99%EC%8A%B5-%EA%B0%80%EC%9D%B4%EB%93%9C) · 수집: 2026-11-01 · 원문 보존 이관본

## 실습 구성표

| 항목 | 실제 원문 코드 기준 |
| --- | --- |
| Task | 해당 없음 |
| Model | 목차 |
| nn.Module | 해당 없음 |
| Autograd | 해당 없음 |
| Loss | 해당 없음 |
| Optimizer | 해당 없음 |
| Activation | 해당 없음 |

복수 예제의 값은 하나의 통합 학습 설정을 뜻하지 않습니다. 코드 조각은 독립 실행 파일로 간주하지 않습니다.

[원본 목록·검증 범위](../sources.md)

---

Machine Learning과 Deep Learning의 개념을 이해하는 것에서 나아가, **직접 모델을 구현하고 학습 과정을 분석하기 위한 실습 시리즈**이다.

이 시리즈에서는 scikit-learn으로 기본적인 Perceptron과 MLP를 구현하는 것부터 시작하여, PyTorch를 이용한 Dataset 구성, 모델 정의, Training Loop, Evaluation, Optimization 및 학습 안정화 기법까지 단계적으로 다룬다.

단순히 코드를 실행하여 결과를 확인하는 것보다 다음 질문에 답할 수 있도록 학습하는 것을 목표로 한다.

- 코드의 각 구성 요소는 어떤 역할을 하는가?
- 모델은 어떤 순서로 Forward Propagation과 Backpropagation을 수행하는가?
- Parameter는 어떤 과정을 통해 업데이트되는가?
- Learning Rate와 Optimizer는 학습 과정에 어떤 영향을 미치는가?
- 학습 결과는 어떤 지표로 평가하는가?
- 학습이 불안정할 때 어떤 방법을 적용할 수 있는가?

수학적 정의와 이론 중심의 내용은 별도의 **[👾DL_a] 딥러닝 기초 개념 정리** 시리즈에서 다룬다.

---

## Part 1. scikit-learn으로 시작하는 신경망 구현

scikit-learn을 이용하여 Perceptron과 Multi-Layer Perceptron을 구현하고, Classification과 Regression 문제에 적용한다.

1. [scikit-learn으로 Perceptron 구현하기](../practice/sklearn/c01.md)
2. [scikit-learn MLPClassifier 이해하기](../practice/sklearn/c02.md)
3. [scikit-learn MLPRegressor로 California Housing 집값 예측하기](../practice/sklearn/c03.md)

**핵심 실습 내용**

- `sklearn.linear_model.Perceptron`
- `sklearn.neural_network.MLPClassifier`
- `sklearn.neural_network.MLPRegressor`
- Classification과 Regression 모델 학습
- Dataset 분할과 모델 평가

이 Part에서는 scikit-learn이 제공하는 API를 이용하여 모델을 학습하고 평가하는 전체적인 과정을 익힌다.

---

## Part 2. PyTorch의 기본 구조와 학습 과정

PyTorch에서 Neural Network를 구현하기 위해 필요한 주요 구성 요소를 이해한다.

특히 Model, Dataset, DataLoader, Loss Function, Optimizer가 어떤 관계를 가지는지 살펴본다.

4. [PyTorch Custom 모델 구현: nn.Module 상속부터 모델 관리까지](../practice/pytorch-basics/c04.md)
5. [PyTorch Dataset과 DataLoader 이해하기 — 클래스 상속부터 데이터 전처리까지](../practice/pytorch-basics/c05.md)
6. [PyTorch 딥러닝 모델 구현 실습: 선형회귀부터 ANN 학습·평가까지](../practice/regression/c06.md)

**핵심 실습 내용**

- Tensor와 Model Parameter
- `nn.Module` 상속과 `forward()`
- `Dataset`과 `DataLoader`
- Loss Function과 Optimizer
- `loss.backward()`와 `optimizer.step()`
- Training과 Evaluation
- 모델 저장 및 관리

이 Part에서는 개별 구성 요소를 이해하고, 하나의 Training Pipeline으로 연결하는 과정을 학습한다.

```text
Dataset
   │
   ▼
DataLoader
   │
   ▼
Model(nn.Module)
   │
   ▼
Forward Propagation
   │
   ▼
Loss Calculation
   │
   ▼
Backward Propagation
   │
   ▼
Optimizer Step
   │
   ▼
Model Evaluation
```

---

## Part 3. PyTorch MLP 구현과 Classification 실습

앞에서 학습한 PyTorch의 주요 구성 요소를 이용하여 실제 Neural Network를 구현한다.

간단한 MLP부터 XOR Problem, Iris Dataset을 이용한 Multiclass Classification까지 단계적으로 확장한다.

7. [PyTorch로 MLP 직접 구현하기](../practice/classification/c07.md)
8. [PyTorch로 XOR 해결하기: MLP 클래스 구조 분석](../practice/classification/c08.md)
9. [PyTorch로 Iris Dataset 다중 분류 구현하기: Dataset부터 ANN 학습 및 평가까지](../practice/classification/c09.md)

**핵심 실습 내용**

- Multi-Layer Perceptron 구현
- Hidden Layer와 Activation Function
- XOR Problem과 Non-linear Decision Boundary
- Multiclass Classification
- Cross-Entropy Loss
- Model Training과 Prediction
- Accuracy 및 Classification Metrics를 이용한 성능평가

이 Part에서는 단순히 모델을 정의하는 것을 넘어, **실제 Dataset을 이용하여 학습하고 결과를 해석하는 전체 과정**을 다룬다.

---

## Part 4. 신경망 최적화와 학습 안정화

Deep Neural Network의 학습 성능은 Model Architecture뿐 아니라 Learning Rate, Optimizer, Weight Initialization 등 다양한 설정의 영향을 받는다.

학습 과정에서 Learning Rate가 부적절하면 Loss가 느리게 감소하거나 진동할 수 있으며, Gradient의 크기가 지나치게 작거나 커지면 Vanishing·Exploding Gradient 문제가 발생할 수 있다.

이러한 문제를 이해하고 해결하기 위해 Learning Rate Scheduling, Adaptive Optimization, Weight Initialization, Normalization, Gradient Clipping 등의 방법을 학습한다.

10. [Learning Rate 완전 정리: 학습률 설정부터 Scheduler·Warmup·Cosine Annealing과 PyTorch 구현까지](../practice/tech-optimizers/c10.md)
11. [AdamW 완전 정리: Adam의 Moment Estimation부터 L2 Regularization·Weight Decay와 PyTorch 구현까지](../practice/tech-optimizers/c11.md)
12. [PyTorch 가중치 초기화 완전 정리: Xavier·He·LeCun Initialization과 Gradient 문제](../practice/tech-weight/c12.md)
13. [PyTorch Batch Normalization 완전 정리: 동작 원리부터 Running Statistics와 학습·추론까지](../practice/tech-normalization/c13.md)
14. [PyTorch Gradient Clipping 완전 정리: Exploding Gradient부터 Norm·Value Clipping 구현까지](../practice/tech-gradients/c14.md)

**핵심 실습 내용**

- Learning Rate와 Optimization Stability
- Learning Rate Scheduler와 Cosine Annealing
- Linear Warmup과 Learning Rate Decay
- SGD, Momentum, Adam, AdamW의 차이
- First·Second Moment Estimation
- L2 Regularization과 Decoupled Weight Decay
- Vanishing Gradient와 Exploding Gradient
- Xavier·He·LeCun Initialization
- PyTorch `torch.nn.init`
- Batch Normalization과 Running Statistics
- `model.train()`과 `model.eval()`
- Norm Clipping과 Value Clipping
- Training Stability와 Convergence

### 각 기법의 역할

이 Part에서 다루는 방법들은 모두 신경망의 학습 과정과 관련이 있지만, 적용 목적과 시점은 서로 다르다.

| Method | 주요 역할 | 적용 시점 |
| --- | --- | --- |
| Learning Rate Scheduler | Learning Rate를 Training 진행에 따라 조절 | 지정된 Scheduler Step |
| AdamW | Adaptive Moment Estimation과 Decoupled Weight Decay | Optimizer Step |
| Weight Initialization | 초기 Weight의 Scale 및 Distribution 설정 | Training 시작 전 |
| Batch Normalization | Activation 정규화 및 Learnable Scale·Shift 적용 | Forward Propagation |
| Gradient Clipping | 지나치게 큰 Gradient 제한 | Backpropagation 이후, Optimizer Step 이전 |

### PyTorch Training Pipeline

이러한 기법들은 실제 Training Pipeline에서 다음과 같이 연결할 수 있다.

```text
Before Training
       │
       ▼
Model Definition
       │
       ▼
Weight Initialization
       │
       ▼
Optimizer & LR Scheduler 설정
       │
       ▼
────────────────────────────
       Training Loop
────────────────────────────
       │
       ▼
Input Batch
       │
       ▼
Forward Propagation
       │
       ├── Linear / Conv
       ├── Batch Normalization
       └── Activation Function
       │
       ▼
Loss Calculation
       │
       ▼
Backpropagation
       │
       ▼
Gradient Clipping (Optional)
       │
       ▼
Optimizer Step (AdamW 등)
       │
       ▼
LR Scheduler Step (설정에 따라)
       │
       ▼
Next Training Step
```

위 흐름은 개념적인 Training Pipeline이다.

Weight Initialization은 학습이 시작되기 전에 수행되며, Batch Normalization은 Model Architecture에 포함되어 Forward Propagation 과정에서 동작한다.

Gradient Clipping은 필요에 따라 `loss.backward()` 이후, `optimizer.step()` 이전에 적용한다.

AdamW는 Gradient의 Moment Estimation을 이용하여 Parameter를 업데이트하고, Weight Decay를 별도로 적용한다.

Learning Rate Scheduler는 설정에 따라 Epoch 또는 Optimizer Step 단위로 실행되며, `ReduceLROnPlateau`처럼 Validation Metric을 이용하는 Scheduler는 평가 이후에 호출할 수 있다.

이처럼 각 기법은 서로 다른 역할을 수행하지만, 하나의 Training Process 안에서 함께 활용할 수 있다.

---

## 전체 학습 흐름

```text
scikit-learn
      │
      ├── Perceptron
      ├── MLPClassifier
      └── MLPRegressor
      │
      ▼
PyTorch Fundamentals
      │
      ├── nn.Module
      ├── Dataset
      ├── DataLoader
      ├── Loss Function
      └── Optimizer
      │
      ▼
Neural Network Implementation
      │
      ├── Linear Regression
      ├── MLP
      ├── XOR
      └── Iris Classification
      │
      ▼
Optimization & Training Stability
      │
      ├── Learning Rate & Scheduler
      ├── AdamW Optimizer
      ├── Weight Initialization
      ├── Batch Normalization
      └── Gradient Clipping
      │
      ▼
Model Training & Evaluation
```

이 시리즈에서는 먼저 scikit-learn을 이용하여 Neural Network를 사용하는 방법을 익힌다.

이후 PyTorch로 직접 Model과 Training Loop를 구성하면서 Forward Propagation, Backpropagation, Parameter Update 과정의 세부적인 동작을 이해한다.

마지막으로 Learning Rate Scheduling, AdamW, Weight Initialization, Batch Normalization, Gradient Clipping을 학습하면서 **Neural Network의 학습을 효율적으로 제어하고 안정화하는 방법**을 살펴본다.

---

## 관련 시리즈: 딥러닝 기초 개념 정리

코드를 작성하기 전에 수학적 원리와 Neural Network의 동작을 이해하고 싶다면 다음 시리즈를 함께 참고할 수 있다.

**[→ 딥러닝 기초 개념 정리 시리즈 바로가기](https://velog.io/@haru02/series/DL-%EC%A0%95%EB%A6%AC-%EB%B0%8F-%EA%B6%81%EA%B8%88%EC%A6%9D-%ED%95%B4%EA%B2%B0)**

기초 개념 시리즈에서는 다음 내용을 다룬다.

- Machine Learning과 Deep Learning
- Perceptron, MLP, Backpropagation
- Parameter와 Hyperparameter
- Gradient Descent와 Learning Rate
- Probability Distribution과 Likelihood
- MLE·MAP와 Loss Function
- Classification Metrics
- Cross Validation과 Generalization
- Hyperparameter Tuning

두 시리즈는 서로 보완적인 관계로 구성되어 있다.

```text
DL_a: 딥러닝 기초 개념 정리
       │
       ▼
Why & How It Works
       │
       ▼
DL_c: 딥러닝 구현 및 실습
       │
       ▼
How to Implement & Evaluate
```

DL_a에서는 이론과 수학적 원리를 이해하고, DL_c에서는 이를 실제 코드로 구현하고 실험 결과를 분석하는 방식으로 학습할 수 있다.

---

> 이 시리즈는 scikit-learn과 PyTorch를 활용하여 딥러닝을 학습하면서 정리한 개인 실습 기록이다. 각 코드의 동작 원리와 실습 결과를 이해하고, 새로운 실험이나 학습 내용을 추가하면서 지속적으로 보완할 예정이다.
