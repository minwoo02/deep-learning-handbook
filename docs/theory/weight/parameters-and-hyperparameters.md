# Parameter vs Hyperparameter: 모델이 학습하는 값과 사람이 정하는 값

[TOC]


관련 실습: [PyTorch Custom 모델 구현: nn.Module 상속부터 모델 관리까지](../../practice/pytorch-basics/custom-nn-module.md) · [PyTorch 가중치 초기화 완전 정리: Xavier·He·LeCun Initialization과 Gradient 문제](../../practice/tech-weight/weight-initialization.md) · [PyTorch Batch Normalization 완전 정리: 동작 원리부터 Running Statistics와 학습·추론까지](../../practice/tech-normalization/batch-normalization.md)


---

Machine Learning과 Deep Learning을 공부하다 보면 자주 혼동되는 개념이 **Parameter(파라미터)**와 **Hyperparameter(하이퍼파라미터)**이다.

두 개념 모두 모델의 동작과 학습 결과에 영향을 미치지만, 역할은 서로 다르다.

가장 기본적인 차이는 다음과 같다.

> **Parameter는 Training을 통해 모델이 학습하는 값이고, Hyperparameter는 모델의 구조나 Training 방법을 결정하기 위해 설정하는 값이다.**

예를 들어 하나의 인공 뉴런은 다음과 같이 표현할 수 있다.

$$
\boxed{ z=\mathbf{w}^{T}\mathbf{x}+b }
$$

여기서 Weight $\mathbf{w}$와 Bias $b$는 Model Parameter이다.

반면 다음과 같은 값들은 Hyperparameter에 해당한다.

```text
Learning Rate
Batch Size
Number of Epochs
Number of Hidden Layers
Number of Neurons
Activation Function
Weight Initialization
Optimizer
Dropout Rate
Weight Decay
```

Parameter는 주어진 모델 구조 안에서 데이터를 이용하여 학습된다.

Hyperparameter는 학습 과정 자체를 어떻게 구성할지 결정하며, 일반적으로 사용자가 설정하거나 별도의 탐색 절차를 통해 선택한다.

이 글에서는 두 개념의 차이와 함께 대표적인 Hyperparameter들이 Neural Network의 학습에 어떤 영향을 미치는지 정리한다.

---

## 1. Parameter란?

**Parameter**는 모델 내부에서 사용되며, 일반적으로 Training 과정에서 데이터와 Loss Function을 이용하여 학습되는 값이다.

Neural Network에서 가장 대표적인 Parameter는 다음과 같다.

- Weight
- Bias

인공 뉴런의 기본 연산을 다시 살펴보자.

$$
\boxed{ z=\sum_{i=1}^{n}w_ix_i+b }
$$

여기서

- $x_i$: Input Feature
- $w_i$: Weight
- $b$: Bias
- $z$: Linear Combination의 결과

이다.

Weight는 각 Input Feature가 Output에 미치는 영향을 조절한다.

Bias는 Linear Combination에 추가되는 Offset으로, Activation Function이나 Decision Boundary의 위치를 조절할 수 있다.

따라서 Weight와 Bias의 값에 따라 동일한 Input에서도 서로 다른 Output이 생성된다.

---

### 1.1 Parameter는 어떻게 학습되는가?

Neural Network의 Training은 일반적으로 다음 순서로 진행된다.

```text
Input Data
    │
    ▼
Forward Propagation
    │
    ▼
Prediction
    │
    ▼
Loss Calculation
    │
    ▼
Backpropagation
    │
    ▼
Gradient Calculation
    │
    ▼
Optimizer
    │
    ▼
Parameter Update
```

모델은 현재 Parameter를 이용하여 Prediction을 계산한다.

이후 실제 Target과 Prediction을 비교하여 Loss Function을 계산한다.

$$
\boxed{ L(\theta) }
$$

여기서 $\theta$는 모델의 Parameter 전체를 나타낸다.

Backpropagation에서는 각 Parameter에 대한 Gradient를 계산한다.

$$
\boxed{ \nabla_\theta L }
$$

Gradient Descent의 기본적인 Parameter Update는 다음과 같다.

$$
\boxed{ \theta_{t+1} = \theta_t-\eta\nabla_\theta L(\theta_t) }
$$

여기서

- $\theta_t$: 현재 Parameter
- $\eta$: Learning Rate
- $\nabla_\theta L$: Loss에 대한 Gradient

이다.

이 과정에서 실제로 업데이트되는 대상은 $\theta$이다.

반면 Learning Rate $\eta$는 Update의 크기를 조절하는 Hyperparameter이다.

---

## 2. PyTorch에서 Model Parameter 확인

PyTorch에서는 `nn.Module` 내부에 학습 가능한 Parameter를 정의할 수 있다.

대표적인 예가 `nn.Linear`이다.

```python
import torch
import torch.nn as nn

layer = nn.Linear(
    in_features=4,
    out_features=10
)
```

이 Layer는 다음 연산을 수행한다.

$$
\boxed{ \mathbf{y}=W\mathbf{x}+\mathbf{b} }
$$

PyTorch의 `nn.Linear`에서 Weight Shape는 다음과 같다.

```python
print(layer.weight.shape)
```
```text
torch.Size([10, 4])
```

Bias Shape는 다음과 같다.

```python
print(layer.bias.shape)
```
```text
torch.Size([10])
```

따라서 Parameter 수는 다음과 같다.

$$
\boxed{ 4\times10+10=50 }
$$

이다.

전체 Parameter를 확인하려면 다음과 같이 작성할 수 있다.

```python
for name, parameter in layer.named_parameters():

    print(
        name,
        parameter.shape,
        parameter.requires_grad
    )
```

출력은 다음과 같은 형태이다.

```text
weight torch.Size([10, 4]) True
bias torch.Size([10]) True
```

여기서 `requires_grad=True`는 해당 Tensor에 대해 자동미분을 수행하도록 설정되어 있다는 의미이다.

실제로 Optimizer가 Parameter를 업데이트하려면 계산된 Gradient가 존재하고, 해당 Parameter가 Optimizer의 관리 대상에 포함되어 있어야 한다.

---

## 3. Hyperparameter란?

**Hyperparameter**는 모델의 Architecture 또는 Training Procedure를 결정하기 위해 설정하는 값이다.

대표적으로 다음과 같은 Hyperparameter가 존재한다.

| Category | Hyperparameter | 역할 |
| --- | --- | --- |
| Architecture | Number of Layers | 모델의 깊이 결정 |
| Architecture | Number of Neurons | 각 Layer의 크기 결정 |
| Architecture | Activation Function | 비선형 변환 결정 |
| Initialization | Weight Initialization | 초기 Weight 분포 결정 |
| Optimization | Optimizer Type | Parameter Update 방법 결정 |
| Optimization | Learning Rate | Update Scale 조절 |
| Optimization | Momentum / Betas | Optimizer의 갱신 특성 조절 |
| Training | Batch Size | 한 번에 처리할 Sample 수 |
| Training | Number of Epochs | 데이터 반복 학습 횟수 |
| Regularization | Dropout Rate | Training 중 일부 Activation 제거 |
| Regularization | Weight Decay | Weight 크기에 대한 제약 |
| Training Control | LR Scheduler | Learning Rate 변화 방식 결정 |
| Training Control | Early Stopping | 학습 종료 조건 결정 |

Hyperparameter의 특징은 일반적인 Backpropagation에서 Model Weight처럼 직접 최적화되는 대상이 아니라는 것이다.

하지만 Hyperparameter도 모델의 학습 성능에 상당한 영향을 미친다.

특히 다음 세 가지 영역에서 중요하다.

```text
Hyperparameters
│
├── Model Architecture
│   ├── Depth
│   ├── Width
│   └── Activation Function
│
├── Optimization
│   ├── Optimizer
│   ├── Learning Rate
│   └── Learning Rate Schedule
│
└── Training & Generalization
    ├── Batch Size
    ├── Epochs
    ├── Weight Initialization
    ├── Dropout
    └── Weight Decay
```

---

## 4. Parameter와 Hyperparameter의 핵심 차이

두 개념을 비교하면 다음과 같다.

| 구분 | Parameter | Hyperparameter |
| --- | --- | --- |
| 의미 | 모델이 사용하는 내부 값 | 모델 구조 및 학습 설정 |
| 대표적인 예 | Weight, Bias | Learning Rate, Batch Size |
| 결정 방법 | 일반적으로 Training을 통해 학습 | 사용자가 지정하거나 탐색 |
| Gradient | 일반적으로 Loss에 대한 Gradient 계산 | 일반적인 학습에서는 직접 계산하지 않음 |
| Optimizer | Update의 대상 | Optimizer의 설정을 결정 |
| 역할 | 모델의 Prediction 결정 | 모델이 학습되는 방식에 영향 |

다음과 같이 요약할 수 있다.

```text
Parameter
    │
    ▼
What the model learns


Hyperparameter
    │
    ▼
How the model is configured and trained
```

단, 두 개념을 구분할 때 **학습 중 값이 변화하는지 여부만을 기준으로 판단하는 것은 정확하지 않다.**

Learning Rate는 Scheduler에 의해 학습 중 변화할 수 있다.

반대로 Weight는 Freeze하여 업데이트하지 않을 수도 있다.

따라서 핵심은 해당 값이 **모델의 학습 대상인지, 또는 학습 절차를 제어하는 설정인지**이다.

---

## 5. Number of Hidden Layers

신경망의 Layer 수는 대표적인 Architecture Hyperparameter이다.

Hidden Layer가 많아질수록 일반적으로 Network Depth가 증가한다.

예를 들어 다음과 같은 구조를 생각할 수 있다.

#### Shallow Network

```text
Input
  │
  ▼
Hidden Layer
  │
  ▼
Output
```

#### Deep Network

```text
Input
  │
  ▼
Hidden Layer 1
  │
  ▼
Hidden Layer 2
  │
  ▼
Hidden Layer 3
  │
  ▼
Output
```

Depth가 증가하면 여러 단계의 Feature Transformation을 수행할 수 있다.

이와 관련된 개념이 **Hierarchical Feature Extraction(계층적 특징 추출)** 이다.

---

### 5.1 Hierarchical Feature Extraction

예를 들어 Image Recognition에서 Neural Network의 Layer들은 서로 다른 수준의 Feature를 학습할 수 있다.

```text
Input Image
    │
    ▼
Low-level Features
Edges / Lines / Orientation
    │
    ▼
Intermediate Features
Shapes / Textures / Patterns
    │
    ▼
High-level Features
Objects / Semantic Structures
    │
    ▼
Prediction
```

낮은 Layer에서는 상대적으로 단순한 Feature를 학습하고, 높은 Layer에서는 이를 조합한 복잡한 Representation을 학습할 수 있다.

따라서 적절한 Architecture에서 Depth를 증가시키면 복잡한 Function을 효율적으로 표현할 수 있다.

하지만 **Layer가 깊을수록 항상 성능이 좋은 것은 아니다.**

Depth가 증가하면 다음 문제도 고려해야 한다.

- Vanishing Gradient
- Exploding Gradient
- Optimization Difficulty
- Computational Cost
- Model Complexity

따라서 Depth는 Task와 Dataset, 모델 구조에 따라 결정해야 한다.

---

## 6. Number of Neurons per Hidden Layer

각 Hidden Layer에 포함되는 Neuron 수는 Network Width를 결정하는 Hyperparameter이다.

예를 들어 다음 코드를 살펴보자.

```python
layer = nn.Linear(4, 100)
```

이 Layer는 4개의 Input Feature를 100개의 Output Feature로 변환한다.

```text
Input Layer
 4 Features
     │
     ▼
Hidden Layer
 100 Neurons
```

여기서 `100`이 Hidden Layer Size에 해당한다.

이 값은 Training 중 Gradient Descent에 의해 자동으로 80이나 120으로 바뀌지 않는다.

---

### 6.1 Width가 증가하면?

일반적으로 Hidden Layer의 Width를 증가시키면 모델의 표현 능력이 증가할 수 있다.

하지만 동시에 Parameter 수도 증가한다.

예를 들어 다음 구조를 생각하자.

```python
nn.Linear(4, 100)
```

Weight Parameter 수는 다음과 같다.

$$
4\times100=400
$$

Bias Parameter 수는 100이다.

따라서 전체 Parameter 수는

$$
\boxed{ 400+100=500 }
$$

이다.

반면 다음과 같이 Width를 200으로 증가시키면

```python
nn.Linear(4, 200)
```

전체 Parameter 수는 다음과 같다.

$$
\boxed{ 4\times200+200=1000 }
$$

이다.

즉 Width를 변경하면 실제로 학습해야 할 Parameter 수도 달라진다.

---

### 6.2 Depth와 Width의 관계

신경망의 표현 능력은 단순히 Parameter 수만으로 결정되지 않는다.

같은 Parameter 수를 가진 모델이라도 Layer의 깊이와 폭을 어떻게 구성하느냐에 따라 서로 다른 함수를 표현할 수 있다.

일부 Task에서는 Wider Network보다 Deeper Network가 Parameter-efficient한 Representation을 학습할 수 있다.

반면 모든 문제에서 Deeper Network가 더 효율적인 것은 아니다.

따라서 Depth와 Width는 함께 고려해야 한다.

```text
Model Capacity
      │
      ├── Depth
      ├── Width
      ├── Connectivity
      └── Activation Function
```

모델이 지나치게 단순하면 Underfitting이 발생할 수 있다.

반대로 모델이 데이터에 비해 지나치게 복잡하면 Overfitting의 위험이 증가할 수 있다.

다만 실제 Overfitting은 모델 크기만으로 결정되는 것이 아니라 Dataset, Regularization, Training Procedure 등 여러 요소의 영향을 받는다.

---

## 7. Activation Function

Activation Function도 Architecture를 결정하는 중요한 Hyperparameter이다.

Neural Network에서 Linear Transformation 이후 비선형 함수를 적용하면 다음과 같다.

$$
\boxed{ \mathbf{a}^{(l)} = f\left( W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \right) }
$$

여기서 $f$가 Activation Function이다.

Activation Function은 단순히 Output을 변환하는 역할만 하는 것이 아니다.

다음과 같은 특성에도 영향을 미친다.

- Non-linearity
- Model Representation
- Gradient Flow
- Training Stability
- Optimization Performance

대표적인 Activation Function은 다음과 같다.

| Activation | 주요 특징 |
| --- | --- |
| Sigmoid | 출력을 0~1 범위로 변환 |
| Tanh | 출력을 -1~1 범위로 변환 |
| ReLU | 음수 입력을 0으로 변환 |
| Leaky ReLU | 음수 영역에 작은 기울기 유지 |
| ELU | 음수 영역에 Exponential Function 사용 |
| GELU | Gaussian CDF 기반의 부드러운 비선형 변환 |
| SiLU / Swish | Sigmoid 기반의 Smooth Activation |

특히 Hidden Layer에서 비선형 Activation을 제거하면 여러 Linear Layer를 쌓아도 전체 변환은 하나의 Affine Transformation으로 결합될 수 있다.

따라서 Activation Function은 DNN이 복잡한 비선형 관계를 표현하는 데 중요한 요소이다.

---

### 7.1 Output Layer의 Activation Function

Output Layer에서는 Task에 따라 Activation Function을 선택한다.

대표적인 관계는 다음과 같다.

| Task | 대표적인 출력 처리 |
| --- | --- |
| Binary Classification | Sigmoid |
| Multiclass Classification | Softmax |
| Regression | Identity / Linear Output |

다만 PyTorch에서는 Loss Function이 내부적으로 Activation을 포함하는 경우가 있다.

예를 들어 다음 Loss Function을 사용하는 경우이다.

```python
nn.BCEWithLogitsLoss()
```

이 함수는 내부적으로 Sigmoid와 BCE 계산을 결합한다.

따라서 모델에서 Sigmoid를 별도로 적용하지 않고 Raw Logit을 전달한다.

마찬가지로

```python
nn.CrossEntropyLoss()
```

에는 일반적으로 Softmax를 적용하기 전의 Raw Logit을 전달한다.

즉 Activation Function 선택은 Task뿐 아니라 Loss Function과도 함께 고려해야 한다.

---

## 8. Weight Initialization

Weight Initialization은 Training을 시작하기 전에 Weight의 초기값을 결정하는 방법이다.

예를 들어 다음과 같은 방법을 사용할 수 있다.

- Random Normal Initialization
- Random Uniform Initialization
- Xavier Initialization
- He Initialization
- LeCun Initialization

Weight의 초기값이 지나치게 크거나 작으면 Forward Propagation과 Backpropagation에서 Signal Scale이 불안정해질 수 있다.

따라서 Initialization 방법은 Activation Function과 함께 고려하는 경우가 많다.

대표적으로 다음과 같은 조합이 사용된다.

| Activation Function | 고려할 Initialization |
| --- | --- |
| Tanh | Xavier |
| ReLU | He(Kaiming) |
| Leaky ReLU | Negative Slope를 고려한 He |
| SELU | LeCun 계열 |

예를 들어 ReLU를 사용하는 Layer에서는 다음과 같이 초기화할 수 있다.

```python
nn.init.kaiming_normal_(
    layer.weight,
    nonlinearity="relu"
)
```

중요한 점은 다음과 같다.

**Weight Initialization 방법은 Hyperparameter이지만, 초기화된 Weight Tensor 자체는 Model Parameter이다.**

```text
Initialization Method
       │
       ▼
   Hyperparameter
       │
       ▼
Initial Weight Values
       │
       ▼
   Model Parameters
       │
       ▼
      Training
```

즉 초기화 방법과 실제 Weight는 서로 구분해야 한다.

---

## 9. Learning Rate

Learning Rate는 Parameter Update의 Scale을 결정하는 대표적인 Hyperparameter이다.

기본적인 Gradient Descent는 다음과 같다.

$$
\boxed{ \theta_{t+1} = \theta_t-\eta\nabla_\theta L(\theta_t) }
$$

여기서 $\eta$가 Learning Rate이다.

PyTorch에서는 다음과 같이 설정한다.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

여기서

```python
model.parameters()
```

는 Optimizer가 관리하는 Model Parameter이다.

반면

```python
lr=0.01
```

은 Learning Rate Hyperparameter이다.

---

### 9.1 Learning Rate가 큰 경우

Learning Rate가 지나치게 크면 Parameter Update가 커질 수 있다.

```text
Large Learning Rate
        │
        ▼
Large Update Step
        │
        ▼
Oscillation / Instability
        │
        ▼
Convergence Failure 가능
```

따라서 Loss가 심하게 진동하거나 발산할 수 있다.

---

### 9.2 Learning Rate가 작은 경우

반대로 Learning Rate가 지나치게 작으면 Parameter가 조금씩 변화한다.

```text
Small Learning Rate
        │
        ▼
Small Update Step
        │
        ▼
Slow Optimization
```

따라서 학습 속도가 느려질 수 있다.

다만 실제 Parameter Update의 크기는 Gradient, Optimizer, Momentum 및 Adaptive Scaling 등의 영향을 함께 받는다.

따라서 Learning Rate만으로 모든 Optimizer의 실제 Update 크기를 결정할 수 있는 것은 아니다.

---

## 10. Learning Rate Scheduler

Hyperparameter는 Training 시작 전에 설정한다고 해서 반드시 끝까지 같은 값을 유지해야 하는 것은 아니다.

대표적인 예가 Learning Rate Scheduler이다.

Learning Rate Scheduler는 Training 과정에서 Learning Rate를 변화시키는 방법이다.

예를 들어 다음과 같이 구성할 수 있다.

```text
Epoch 1~10
Learning Rate = 0.01

Epoch 11~20
Learning Rate = 0.001

Epoch 21~
Learning Rate = 0.0001
```

PyTorch에서는 다음과 같이 구현할 수 있다.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)
```

각 Epoch의 Training이 끝난 후 다음과 같이 호출할 수 있다.

```python
scheduler.step()
```

이때 Scheduler는 설정된 규칙에 따라 Learning Rate를 변경한다.

그러나 Learning Rate가 변화한다고 해서 Model Weight와 동일한 종류의 Parameter가 되는 것은 아니다.

Scheduler의 설정과 Learning Rate는 여전히 Training Procedure를 제어하는 Hyperparameter에 해당한다.

---

## 11. Batch Size

Batch Size는 하나의 Training Step에서 처리하는 Sample 수를 의미한다.

예를 들어 다음과 같이 설정할 수 있다.

```python
batch_size = 32
```

전체 Training Dataset에 1,000개의 Sample이 있다고 하자.

$$
\boxed{ N=1000 }
$$

Batch Size가 32라면 필요한 Mini-batch 수는 다음과 같다.

$$
\boxed{ \left\lceil\frac{1000}{32}\right\rceil=32 }
$$

이다.

마지막 Batch까지 사용한다면 하나의 Epoch에서 32번의 Mini-batch 처리가 수행된다.

일반적인 Mini-batch Training에서 각 Batch마다 Optimizer Step을 수행한다면 32번의 Parameter Update가 발생한다.

---

### 11.1 Batch Size가 작은 경우

작은 Batch Size를 사용하면 일반적으로 다음과 같은 특성이 나타날 수 있다.

- 상대적으로 적은 Memory 사용
- Mini-batch Gradient의 변동 증가
- 동일한 Epoch에서 더 많은 Update Step 발생

반면 Batch Size가 크면 다음 특성이 나타날 수 있다.

- 큰 Batch 처리에 더 많은 Memory 필요
- Gradient Estimate의 변동이 상대적으로 감소
- Hardware에 따라 높은 Parallel Processing Efficiency 확보 가능

다만 Batch Size가 크다고 학습 속도가 항상 빠르거나 Generalization이 항상 좋아지는 것은 아니다.

Batch Size의 효과는 Model, Hardware, Optimizer, Learning Rate 등에 따라 달라진다.

따라서 Batch Size는 Memory 효율과 Optimization 특성을 함께 고려하여 결정해야 한다.

---

## 12. Epoch

Epoch는 전체 Training Dataset을 한 번 사용하는 학습 단위를 의미한다.

예를 들어 다음과 같이 설정할 수 있다.

```python
epochs = 100
```

Training Loop는 다음과 같다.

```python
for epoch in range(epochs):

    for X, y in train_loader:

        # Forward
        # Backward
        # Parameter Update
        ...
```

이 경우 Training Dataset을 총 100번 반복하여 학습하도록 설정한다.

---

### 12.1 Epoch와 Step의 차이

Epoch와 Step은 서로 다른 개념이다.

**Epoch**

전체 Training Dataset을 한 번 사용하는 단위이다.

**Step**

일반적으로 Optimizer가 Parameter를 한 번 업데이트하는 단위이다.

예를 들어

$$
N=1000
$$

$$
B=32
$$

이고 마지막 Batch도 사용한다면 하나의 Epoch에는 32개의 Mini-batch가 존재한다.

각 Batch에서 한 번씩 Parameter를 업데이트하는 경우

$$
\boxed{ \text{Steps per Epoch}=32 }
$$

이다.

100 Epoch를 수행하면

$$
\boxed{ \text{Total Steps}=3200 }
$$

이 된다.

다만 Gradient Accumulation을 사용하면 여러 Mini-batch에 대해 한 번의 Optimizer Step을 수행할 수도 있다.

따라서 Mini-batch 수와 Optimizer Step 수가 항상 동일한 것은 아니다.

---

## 13. Optimizer

Optimizer는 Gradient를 이용하여 Model Parameter를 업데이트하는 알고리즘이다.

대표적인 Optimizer는 다음과 같다.

- SGD
- SGD with Momentum
- RMSProp
- Adam
- AdamW

Optimizer를 선택하는 것 자체가 Hyperparameter이다.

또한 각 Optimizer에는 추가적인 Hyperparameter가 존재한다.

예를 들어 AdamW에서는 다음과 같이 설정할 수 있다.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

각 인자의 의미는 다음과 같다.

| Argument | 역할 |
| --- | --- |
| `model.parameters()` | Optimizer가 관리할 Model Parameters |
| `lr` | Learning Rate |
| `betas` | Gradient의 이동평균 계수 |
| `weight_decay` | Weight Decay Strength |

따라서 하나의 Optimizer 코드 안에 Model Parameter와 여러 Hyperparameter가 함께 존재한다.

---

## 14. Regularization Hyperparameters

Regularization은 모델이 Training Data에 지나치게 적합되는 현상을 완화하기 위해 사용하는 방법이다.

대표적으로 다음과 같은 방법이 존재한다.

- Dropout
- Weight Decay
- Early Stopping

이러한 방법들의 적용 여부와 설정값도 Hyperparameter가 될 수 있다.

---

### 14.1 Dropout Rate

Dropout은 Training 중 일부 Activation을 무작위로 제거하는 방법이다.

PyTorch에서는 다음과 같이 설정한다.

```python
dropout = nn.Dropout(p=0.2)
```

여기서

```python
p=0.2
```

는 Dropout Probability이다.

즉 Training 과정에서 각 Activation을 20%의 확률로 0으로 만든다.

Dropout Rate가 증가하면 일반적으로 Regularization 효과가 강해질 수 있다.

하지만 지나치게 큰 Dropout Rate는 모델이 충분한 Feature를 학습하지 못하게 할 수 있다.

따라서 적절한 값을 선택해야 한다.

---

### 14.2 Weight Decay

Weight Decay는 Weight의 크기를 제한하는 방향으로 Optimization에 영향을 주는 설정이다.

L2 Regularization에서는 다음과 같은 목적함수를 사용할 수 있다.

$$
\boxed{ J(\theta) = L_{\mathrm{data}}(\theta) + \lambda\|\theta\|_2^2 }
$$

여기서 $\lambda$는 Regularization Strength이다.

PyTorch에서는 다음과 같이 설정할 수 있다.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

다만 AdamW의 Weight Decay는 일반적인 Loss Function에 L2 Penalty를 추가하는 방식과 구현상 차이가 있다.

AdamW는 Adaptive Gradient Update와 Weight Decay를 분리하는 Decoupled Weight Decay를 사용한다.

따라서 Weight Decay와 L2 Regularization을 모든 Optimizer에서 완전히 동일한 개념으로 취급해서는 안 된다.

---

### 14.3 Early Stopping

Early Stopping은 Validation Performance가 더 이상 개선되지 않을 때 Training을 종료하는 방법이다.

예를 들어 Validation Loss가 일정 기간 개선되지 않으면 학습을 중단할 수 있다.

```text
Epoch 1
Validation Loss ↓

Epoch 2
Validation Loss ↓

Epoch 3
Validation Loss ↓

Epoch 4
Validation Loss ↑

Epoch 5
Validation Loss ↑

Epoch 6
Validation Loss ↑

        │
        ▼
    Early Stopping
```

여기서 개선되지 않은 Epoch를 몇 번까지 허용할지 결정하는 `patience` 역시 Hyperparameter이다.

Early Stopping은 Overfitting 완화와 불필요한 계산 감소에 도움이 될 수 있다.

---

## 15. PyTorch에서 Parameter와 Hyperparameter 함께 확인하기

이번에는 하나의 MLP 모델에서 두 개념이 어떻게 구분되는지 살펴보자.

다음과 같이 Hyperparameter를 설정한다.

```python
hidden_size = 64
num_hidden_layers = 2

learning_rate = 0.001
batch_size = 32
epochs = 20

dropout_rate = 0.1
weight_decay = 0.01
```

이 값들은 모델의 구조와 Training Procedure를 결정한다.

이제 MLP를 정의한다.

```python
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_hidden_layers=2,
        dropout_rate=0.1
    ):

        super().__init__()

        layers = []

        in_features = input_size

        for _ in range(num_hidden_layers):

            layers.append(
                nn.Linear(
                    in_features,
                    hidden_size
                )
            )

            layers.append(nn.ReLU())

            layers.append(
                nn.Dropout(dropout_rate)
            )

            in_features = hidden_size

        layers.append(
            nn.Linear(
                in_features,
                output_size
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)
```

모델을 생성한다.

```python
model = MLP(
    input_size=4,
    hidden_size=hidden_size,
    output_size=3,
    num_hidden_layers=num_hidden_layers,
    dropout_rate=dropout_rate
)
```

이 모델의 구조는 다음과 같다.

```text
Input(4)
    │
    ▼
Linear(4, 64)
    │
    ▼
ReLU
    │
    ▼
Dropout(0.1)
    │
    ▼
Linear(64, 64)
    │
    ▼
ReLU
    │
    ▼
Dropout(0.1)
    │
    ▼
Linear(64, 3)
    │
    ▼
Output Logits
```

여기에서

```text
Hidden Size = 64
Hidden Layers = 2
Dropout Rate = 0.1
```

은 Hyperparameter이다.

반면 각 `nn.Linear` 내부의 Weight와 Bias는 Model Parameter이다.

---

### 15.1 실제 Model Parameter 확인

다음 코드를 이용하여 Parameter를 확인할 수 있다.

```python
for name, parameter in model.named_parameters():

    print(
        name,
        parameter.shape
    )
```

출력은 다음과 같은 형태이다.

```text
network.0.weight torch.Size([64, 4])
network.0.bias torch.Size([64])

network.3.weight torch.Size([64, 64])
network.3.bias torch.Size([64])

network.6.weight torch.Size([3, 64])
network.6.bias torch.Size([3])
```

이처럼 Model Parameter는 실제 Tensor로 존재한다.

모델 전체의 Parameter 수는 다음과 같이 계산할 수 있다.

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(total_params)
```

출력은 다음과 같다.

```text
4675
```

이를 수식으로 계산하면 다음과 같다.

$$
\begin{aligned} \text{Layer 1} &=4\times64+64=320\\ \text{Layer 2} &=64\times64+64=4160\\ \text{Output Layer} &=64\times3+3=195 \end{aligned}
$$

따라서

$$
\boxed{ 320+4160+195=4675 }
$$

이다.

여기서 Hyperparameter인 `hidden_size` 또는 `num_hidden_layers`를 변경하면 모델의 Parameter 수 역시 달라진다.

하지만 모델을 생성한 이후의 각 Weight 값은 Training을 통해 학습된다.

---

## 16. Training Loop에서 두 개념 구분하기

이제 간단한 Synthetic Dataset을 생성하여 Training Loop를 구성해 보자.

```python
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

torch.manual_seed(42)

X = torch.randn(512, 4)

score = (
    X[:, 0]
    + 0.5 * X[:, 1]
    - X[:, 2]
)

y = (score > 0).long()
```

이 예제는 4개의 Feature와 두 개의 Class를 가진 가상의 Dataset이다.

따라서 출력 Layer의 크기를 2로 변경한다.

```python
model = MLP(
    input_size=4,
    hidden_size=hidden_size,
    output_size=2,
    num_hidden_layers=num_hidden_layers,
    dropout_rate=dropout_rate
)
```

Dataset과 DataLoader를 생성한다.

```python
dataset = TensorDataset(X, y)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)
```

Loss Function과 Optimizer를 정의한다.

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
```

이제 Training Loop를 실행한다.

```python
for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        # Forward Propagation
        logits = model(X_batch)

        # Loss Calculation
        loss = criterion(
            logits,
            y_batch
        )

        # Backpropagation
        loss.backward()

        # Parameter Update
        optimizer.step()

        total_loss += (
            loss.item() * len(X_batch)
        )

    avg_loss = total_loss / len(dataset)

    print(
        f"Epoch {epoch + 1:02d} "
        f"| Loss: {avg_loss:.4f}"
    )
```

이 코드는 Hyperparameter가 어떤 역할을 하는지 보여준다.

```text
Hyperparameters
│
├── hidden_size
├── num_hidden_layers
├── dropout_rate
├── learning_rate
├── batch_size
├── epochs
└── weight_decay
        │
        ▼
Model Architecture & Training Procedure
        │
        ▼
Forward Propagation
        │
        ▼
Loss
        │
        ▼
Backpropagation
        │
        ▼
Gradient
        │
        ▼
Optimizer Step
        │
        ▼
Model Parameter Update
        │
        ├── Weight
        └── Bias
```

이 예제는 개념 확인을 위한 Training 코드이다.

실제 모델의 Hyperparameter를 비교하고 Generalization Performance를 평가할 때는 Training Set과 별도의 Validation Set을 사용해야 한다.

---

## 17. Hyperparameter Tuning

Hyperparameter를 설정할 때 어려운 점은 어떤 값이 가장 적절한지 사전에 알기 어렵다는 것이다.

예를 들어 Learning Rate를 다음과 같이 설정할 수 있다.

```text
0.1
0.01
0.001
0.0001
```

Hidden Layer Size도 다음과 같이 선택할 수 있다.

```text
32
64
128
256
```

그러나 어떤 조합이 가장 좋은지는 Dataset과 Model Architecture에 따라 달라진다.

따라서 여러 Hyperparameter 조합을 실험하여 적절한 값을 찾는 과정을 **Hyperparameter Tuning**이라고 한다.

대표적인 방법은 다음과 같다.

- Manual Search
- Grid Search
- Random Search
- Bayesian Optimization

일반적으로 Hyperparameter Tuning에서는 Training Set으로 Model Parameter를 학습하고, Validation Set으로 Hyperparameter 설정을 비교한다.

```text
Hyperparameter Candidates
          │
          ▼
      Model Training
          │
          ▼
     Validation Metric
          │
          ▼
      Comparison
          │
          ▼
Best Hyperparameter Configuration
```

여기서 중요한 점은 Test Set을 반복적으로 Hyperparameter 선택에 사용하지 않는 것이다.

Test Set은 최종적으로 선택한 모델의 Generalization Performance를 평가하는 데 사용하는 것이 원칙이다.

Hyperparameter Tuning의 알고리즘과 구현 방법은 별도의 글에서 자세히 다룰 수 있다.

---

## 18. Parameter와 Hyperparameter에 대한 주의사항

### 18.1 Hyperparameter는 Training 동안 반드시 고정되어야 할까?

그렇지 않다.

Learning Rate Scheduler를 이용하면 Training 중 Learning Rate를 변경할 수 있다.

하지만 Learning Rate는 일반적인 Model Weight와 같이 Backpropagation으로 학습되는 Parameter가 아니다.

따라서 **값이 변하는지 여부만으로 Parameter와 Hyperparameter를 구분해서는 안 된다.**

---

### 18.2 Parameter를 사람이 직접 설정할 수도 있을까?

가능하다.

예를 들어 Weight를 직접 초기화할 수 있다.

```python
nn.init.constant_(
    model.network[0].weight,
    0.5
)
```

또한 Pretrained Parameter를 불러올 수도 있다.

```python
model.load_state_dict(
    pretrained_state_dict
)
```

따라서 Parameter는 반드시 Random Initialization으로 시작해야 하는 것은 아니다.

중요한 것은 해당 Tensor가 모델 내부에서 Prediction을 결정하는 데 사용되고, 일반적인 Training 과정에서 학습 대상이 될 수 있다는 점이다.

---

### 18.3 모든 Model Parameter가 항상 업데이트될까?

그렇지 않다.

Transfer Learning에서는 일부 Layer를 Freeze할 수 있다.

```python
for parameter in model.network[0].parameters():

    parameter.requires_grad = False
```

이 경우 해당 Parameter는 일반적인 Backpropagation에서 Gradient를 계산하지 않는다.

하지만 여전히 Model Parameter이다.

반면 Freeze되지 않은 다른 Parameter는 계속 학습될 수 있다.

---

### 18.4 Weight와 Bias만 Parameter일까?

아니다.

Model에 따라 다른 학습 가능한 Parameter도 존재한다.

예를 들어 Batch Normalization에서는 다음 값들이 학습 가능한 Parameter가 될 수 있다.

$$
\boxed{ \gamma,\beta }
$$

PReLU에서는 음수 영역의 기울기 $\alpha$가 학습 가능한 Parameter이다.

$$
\boxed{ f(x)= \begin{cases} x,&x\ge0\\ \alpha x,&x<0 \end{cases} }
$$

따라서 Parameter를 단순히 Weight와 Bias에만 한정해서 이해해서는 안 된다.

**모델 구조에 포함되어 학습되는 값이라면 다른 종류의 값도 Parameter가 될 수 있다.**

---

### 18.5 Hyperparameter도 자동으로 최적화할 수 있을까?

가능하다.

Grid Search, Random Search, Bayesian Optimization과 같은 방법을 이용하면 Hyperparameter를 자동으로 탐색할 수 있다.

또한 일부 고급 학습 방법에서는 Gradient-based Hyperparameter Optimization을 수행하기도 한다.

따라서 Hyperparameter를 반드시 사람이 직접 하나씩 정해야 한다고 이해하는 것도 정확하지 않다.

핵심은 일반적인 Model Parameter 최적화와 Hyperparameter 선택이 서로 다른 수준의 Optimization이라는 점이다.

---

## 19. Programming에서 Parameter라는 용어

Programming에서도 Parameter라는 단어를 사용한다.

하지만 Programming의 Function Parameter와 Machine Learning의 Model Parameter는 다른 개념이다.

예를 들어 다음 함수를 생각하자.

```python
def train_model(
    learning_rate,
    epochs
):
    ...
```

여기서 `learning_rate`와 `epochs`는 Function Parameter이다.

함수가 값을 전달받기 위해 정의한 변수라는 의미이다.

반면 Machine Learning의 Model Parameter는 다음과 같은 값이다.

$$
\boxed{ \mathbf{y}=W\mathbf{x}+\mathbf{b} }
$$

여기서

$$
\boxed{ W,\mathbf{b} }
$$

는 Model Parameter이다.

즉 문맥에 따라 같은 `Parameter`라는 용어가 서로 다른 의미로 사용될 수 있다.

```text
Programming
     │
     ▼
Function Parameter


Machine Learning
     │
     ▼
Model Parameter
```

---

## 20. 핵심 비교 및 정리

지금까지 살펴본 개념을 정리하면 다음과 같다.

| 개념 | 의미 |
| --- | --- |
| Model Parameter | 모델 내부에서 학습되는 값 |
| Weight | Input의 영향을 조절하는 Parameter |
| Bias | Linear Transformation에 추가되는 Parameter |
| Hyperparameter | 모델 구조와 학습 방식을 결정하는 설정 |
| Depth | Hidden Layer의 개수 |
| Width | Hidden Layer의 Neuron 수 |
| Activation Function | 비선형 변환을 결정하는 요소 |
| Weight Initialization | 초기 Weight의 Scale과 Distribution 결정 |
| Optimizer | Parameter Update 방법 |
| Learning Rate | Optimizer의 Update Scale 조절 |
| Batch Size | Mini-batch당 Sample 수 |
| Epoch | 전체 Training Dataset의 반복 단위 |
| Dropout Rate | Training 중 Activation 제거 확률 |
| Weight Decay | Weight Decay Strength |
| LR Scheduler | Learning Rate 변화 규칙 |
| Hyperparameter Tuning | 좋은 Hyperparameter Configuration 탐색 |

---

## 최종 정리

Parameter와 Hyperparameter는 모두 Neural Network의 학습에 영향을 미치지만 역할이 다르다.

**Parameter**는 모델이 Training Data로부터 학습하는 내부 값이다.

대표적으로 Weight와 Bias가 있으며, Backpropagation으로 계산된 Gradient를 이용하여 Optimizer가 업데이트한다.

반면 **Hyperparameter**는 Model Architecture와 Training Procedure를 결정하는 설정값이다.

대표적으로 Learning Rate, Batch Size, Epoch, Hidden Layer Size, Activation Function, Weight Initialization 등이 있다.

이를 전체적인 학습 과정으로 정리하면 다음과 같다.

```text
Hyperparameter Selection
          │
          ▼
Model Architecture
          │
          ▼
Weight Initialization
          │
          ▼
Training Data
          │
          ▼
Forward Propagation
          │
          ▼
Loss Calculation
          │
          ▼
Backpropagation
          │
          ▼
Optimizer
          │
          ▼
Parameter Update
          │
          ▼
Validation Performance
          │
          ▼
Hyperparameter Tuning
```

즉 Model Parameter는 하나의 Training Procedure 안에서 학습되고, Hyperparameter는 그 Training Procedure와 Architecture를 설정하는 역할을 한다.

따라서 최종적으로 다음과 같이 기억할 수 있다.

$$
\boxed{ \text{Parameter} = \text{Model이 학습하는 내부 값} }
$$

$$
\boxed{ \text{Hyperparameter} = \text{Model의 구조와 학습을 제어하는 설정값} }
$$

중요한 것은 어떤 값이 학습 도중 변경되는지보다 **어떤 역할로 최적화되고 사용되는지**를 이해하는 것이다.

---

### 핵심 요약

> Parameter는 Weight와 Bias처럼 모델이 데이터를 이용하여 학습하는 내부 값이다. Hyperparameter는 Learning Rate, Batch Size, Hidden Layer Size, Activation Function처럼 모델의 구조와 Training 방법을 결정하는 설정이다. Model Parameter는 일반적으로 Backpropagation과 Optimizer를 통해 갱신되며, Hyperparameter는 별도의 탐색 및 Validation 절차를 통해 선택할 수 있다.

---

## 참고 자료

- [Hyper-Parameters in Deep Learning — Dsaint31](https://dsaint31.me/mkdocs_site/ML/ch08/dl_hyperparameters/)
- [Batch Size - Dsaint31](https://dsaint31.me/mkdocs_site/ML/ch08/dl_batchsize/)
- [PyTorch — Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch — Hyperparameter Tuning](https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
- [PyTorch — torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html)
- [PyTorch — torch.optim](https://docs.pytorch.org/docs/stable/optim.html)
