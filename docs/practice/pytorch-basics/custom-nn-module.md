# PyTorch Custom 모델 구현: nn.Module 상속부터 모델 관리까지

[TOC]


## 실습 구성표

| 항목 | 실습 구성 |
| --- | --- |
| Task | 모델 정의·관리 · 복수 예제 |
| Model | Linear / MLP · 예제별 상이 |
| nn.Module | 사용 |
| Autograd | backward 예제 있음 |
| Loss | MSELoss 예제 있음 |
| Optimizer | Adam 예제 있음 |
| Activation | ReLU 예제 있음 |



관련 이론: [Perceptron에서 MLP와 Backpropagation까지: 신경망은 어떻게 학습하는가?](../../theory/architectures/perceptron-mlp-backpropagation.md) · [Parameter vs Hyperparameter: 모델이 학습하는 값과 사람이 정하는 값](../../theory/weight/parameters-and-hyperparameters.md)


---

PyTorch에서 인공신경망(Artificial Neural Network, ANN)을 구현할 때 가장 기본이 되는 클래스는 `torch.nn.Module`이다.

`nn.Linear`, `nn.Conv2d`, `nn.ReLU`와 같은 신경망 계층뿐 아니라 여러 계층을 조합하여 만든 사용자 정의 모델(Custom Model)도 `nn.Module`을 기반으로 구현한다.

그렇다면 단순히 Python 클래스를 정의하는 것과 `nn.Module`을 상속하여 신경망을 만드는 것은 어떤 차이가 있을까?

핵심은 **PyTorch가 모델 내부의 계층과 학습 가능한 파라미터를 자동으로 등록하고 관리할 수 있다는 점**이다.

`nn.Module`을 사용하면 신경망의 구조를 정의하는 것뿐 아니라 다음 작업도 수행할 수 있다.

- 신경망 계층(Layer) 등록 및 관리
- Weight와 Bias 같은 학습 파라미터 관리
- Forward Propagation 정의
- CPU 및 GPU 장치 이동
- 학습 및 평가 모드 전환
- 모델의 상태 저장과 복원
- Hook을 이용한 내부 연산 관찰

이 글에서는 `nn.Module`의 구조를 이해하고, 실제 사용자 정의 ANN을 구현하면서 주요 메서드의 동작을 살펴본다.

### 이번 실습의 모델 및 구현 구성

이번 실습에서는 PyTorch의 `nn.Module`을 상속하여 사용자 정의 Neural Network를 구현하고, 모델의 구조와 Parameter를 관리하는 방법을 학습한다.

모델 예제로는 여러 Fully Connected Layer를 연결한 간단한 **Multi-Layer Perceptron(MLP)** 구조를 사용한다.

| 항목 | 사용 기술 및 구성 | 역할 |
| --- | --- | --- |
| Framework | PyTorch | Neural Network 구현 |
| Model Architecture | MLP (Multi-Layer Perceptron) | 여러 Layer로 구성된 신경망 |
| Model Definition | `nn.Module` 상속 | 사용자 정의 모델 구현 |
| Layer | `nn.Linear` | Fully Connected Layer |
| Activation Function | `nn.ReLU` | 비선형성 추가 |
| Initialization | `__init__()` | Layer 정의 및 초기화 |
| Forward Propagation | `forward()` | 입력 데이터의 연산 순서 정의 |
| Parameter Management | `parameters()`, `named_parameters()` | Weight와 Bias 확인 |
| Model State | `state_dict()` | 모델 Parameter 및 Buffer 관리 |
| Model Saving / Loading | `torch.save()`, `torch.load()` | 모델 상태 저장 및 복원 |

#### 모델의 기본 구조

```text
Input
  │
  ▼
Linear Layer
  │
  ▼
ReLU Activation
  │
  ▼
Linear Layer
  │
  ▼
Output
```

여기서 세 가지 개념을 구분해야 한다.

- **MLP**는 구현하려는 Neural Network의 구조(Model Architecture)이다.
- **`nn.Module`**은 PyTorch에서 사용자 정의 모델을 구현하기 위한 기본 클래스이다.
- **ReLU**는 Neural Network에 Non-linearity를 추가하는 Activation Function이다.

따라서 이번 글의 핵심은 MLP의 학습 성능을 비교하는 것이 아니라, **`nn.Module`을 이용해 Neural Network를 직접 정의하고, Forward Propagation과 Parameter 관리가 어떻게 이루어지는지 이해하는 것**이다.

---

## 1. torch.nn.Module이란?

`torch.nn.Module`은 PyTorch에서 모든 신경망 모델과 계층의 기반이 되는 클래스이다.

다음과 같이 불러올 수 있다.

```python
import torch
import torch.nn as nn
```

PyTorch에서 일반적으로 사용하는 계층들은 `nn.Module`을 상속한 클래스이다.

예를 들어 다음과 같다.

```python
nn.Linear
nn.Conv2d
nn.ReLU
nn.BatchNorm1d
nn.Dropout
```

이러한 계층들을 조합하여 하나의 신경망 모델을 만들 수 있다.

```text
Custom Neural Network
│
├── Linear Layer
│
├── ReLU Layer
│
├── Linear Layer
│
├── ReLU Layer
│
└── Output Layer
```

여기서 각각의 Layer도 `nn.Module`이고, 이 Layer들을 포함하는 전체 모델 역시 `nn.Module`이다.

즉 PyTorch에서는 **Model과 Layer가 모두 Module이라는 공통 구조를 가진다.**

---

### 1.1 nn.Module을 상속하는 이유

일반적인 Python 클래스에서도 신경망의 연산을 직접 구현할 수 있다.

하지만 단순한 Python 클래스는 PyTorch가 제공하는 모델 관리 기능을 자동으로 지원하지 않는다.

반면 `nn.Module`을 상속하면 다음과 같은 메서드를 사용할 수 있다.

| 메서드 | 기능 |
| --- | --- |
| `parameters()` | 학습 가능한 파라미터 조회 |
| `named_parameters()` | 파라미터 이름과 값 조회 |
| `children()` | 직접 등록된 하위 모듈 조회 |
| `modules()` | 전체 하위 모듈 재귀 조회 |
| `add_module()` | 하위 모듈 등록 |
| `register_buffer()` | Buffer 등록 |
| `state_dict()` | 모델 상태 조회 |
| `load_state_dict()` | 모델 상태 복원 |
| `to()` | 모델의 장치 및 자료형 이동 |
| `train()` | 학습 모드 설정 |
| `eval()` | 평가 모드 설정 |
| `apply()` | 모듈에 함수를 재귀적으로 적용 |

이러한 기능 때문에 `nn.Module`은 신경망을 체계적으로 구현하고 관리하기 위한 기본 클래스라고 볼 수 있다.

---

## 2. Custom Model의 기본 구조

사용자 정의 신경망을 구현할 때는 다음과 같이 `nn.Module`을 상속한다.

```python
import torch
import torch.nn as nn

class CustomModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

여기서 중요한 부분은 다음 세 가지이다.

1. `class CustomModel(nn.Module)`
2. `__init__()`
3. `forward()`

---

### 2.1 class CustomModel(nn.Module)

```python
class CustomModel(nn.Module):
```

이 코드는 `CustomModel`이라는 새로운 클래스를 정의하면서 `nn.Module`을 상속한다는 의미이다.

Python에서는 클래스 이름 뒤의 괄호에 부모 클래스를 지정한다.

따라서

```python
class CustomModel(nn.Module):
```

은 `CustomModel`이 `nn.Module`의 기능을 상속받는다는 의미이다.

이러한 관계를 **Inheritance(상속)** 라고 한다.

```text
torch.nn.Module
       │
       │ Inheritance
       ▼
   CustomModel
```

상속을 통해 사용자 정의 모델에서도 `parameters()`, `state_dict()`, `to()` 등의 메서드를 사용할 수 있다.

---

### 2.2 **init**()의 역할

```python
def __init__(self):
    super().__init__()
```

`__init__()`은 클래스의 인스턴스가 생성될 때 초기화 작업을 수행하는 메서드이다.

신경망에서는 주로 다음 요소들을 정의한다.

- Linear Layer
- Convolution Layer
- Activation Function
- Batch Normalization
- Dropout
- 기타 Submodule

예를 들어 다음과 같다.

```python
class CustomModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 2)
```

이 코드에서는 세 개의 Submodule을 정의한다.

```text
CustomModel
│
├── linear1 : Linear(4, 16)
│
├── relu    : ReLU()
│
└── linear2 : Linear(16, 2)
```

각각의 계층을 `self.linear1`, `self.relu`, `self.linear2`와 같은 인스턴스 속성(Attribute)으로 지정하면 PyTorch가 이를 하위 모듈로 등록한다.

#### super().**init**()이 필요한 이유

`super().__init__()`은 부모 클래스인 `nn.Module`의 초기화 메서드를 호출한다.

이를 통해 PyTorch가 Module을 관리하는 데 필요한 내부 구조가 초기화된다.

따라서 Submodule이나 Parameter를 등록하기 전에 부모 클래스의 초기화가 이루어져야 한다.

```python
def __init__(self):
    super().__init__()  # 부모 클래스 초기화

    self.linear = nn.Linear(4, 2)
```

이 순서를 지키는 것이 중요하다.

---

### 2.3 forward()의 역할

`forward()`는 입력 데이터가 모델의 계층들을 어떤 순서로 통과하는지 정의한다.

이를 Forward Propagation(순전파)이라고 한다.

예를 들어 다음과 같다.

```python
def forward(self, x):

    x = self.linear1(x)

    x = self.relu(x)

    x = self.linear2(x)

    return x
```

입력 데이터는 다음 순서로 처리된다.

```text
Input Tensor
     │
     ▼
Linear Layer
     │
     ▼
ReLU
     │
     ▼
Linear Layer
     │
     ▼
Output Tensor
```

즉 `__init__()`에서는 사용할 계층들을 정의하고, `forward()`에서는 그 계층들을 어떻게 연결하여 연산할지 정의한다.

**`__init__()`은 모델의 구성 요소를 정의하고, `forward()`는 실제 데이터의 흐름을 정의한다.**

---

## 3. Custom ANN 구현하기(MLP)

이제 `nn.Module`을 상속하여 간단한 Multi-Layer Perceptron(MLP)을 구현해 보자.

MLP는 여러 개의 Fully Connected Layer로 구성된 신경망이다.

이번 예제에서는 다음 구조를 사용한다.

```text
Input Layer
    │
    │ 4 Features
    ▼
Linear(4, 32)
    │
    ▼
   ReLU
    │
    ▼
Linear(32, 32)
    │
    ▼
   ReLU
    │
    ▼
Linear(32, 2)
    │
    ▼
Output Layer
```

---

### 3.1 CustomMLP 클래스 구현

```python
import torch
import torch.nn as nn


class CustomMLP(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        # 첫 번째 은닉층
        self.linear1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()

        # 두 번째 은닉층
        self.linear2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()

        # 출력층
        self.linear3 = nn.Linear(32, output_dim)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        x = self.linear3(x)

        return x
```

이 모델은 입력 데이터가 두 개의 은닉층을 통과한 뒤 최종 출력값을 생성하도록 구성되어 있다.

---

### 3.2 모델 인스턴스 생성

```python
model = CustomMLP(
    input_dim=4,
    output_dim=2
)

print(model)
```

출력은 다음과 같은 구조를 가진다.

```text
CustomMLP(
  (linear1): Linear(in_features=4, out_features=32, bias=True)
  (relu1): ReLU()
  (linear2): Linear(in_features=32, out_features=32, bias=True)
  (relu2): ReLU()
  (linear3): Linear(in_features=32, out_features=2, bias=True)
)
```

`print(model)`을 사용하면 등록된 Submodule과 모델의 구조를 확인할 수 있다.

---

### 3.3 입력 데이터 전달

모델에 입력 데이터를 전달해 보자.

```python
x = torch.randn(5, 4)

output = model(x)

print("Input Shape:", x.shape)
print("Output Shape:", output.shape)
```

출력은 다음과 같다.

```text
Input Shape: torch.Size([5, 4])
Output Shape: torch.Size([5, 2])
```

입력 Tensor의 Shape는

$$
(5,4)
$$

이다.

여기서

- 5: Batch Size
- 4: Input Features

이다.

출력 Tensor의 Shape는

$$
(5,2)
$$

이다.

즉 5개의 샘플에 대해 각각 2개의 출력값이 생성된다.

출력 Shape가 어떻게 결정되는지는 마지막 Linear Layer를 보면 이해할 수 있다.

```python
self.linear3 = nn.Linear(32, 2)
```

이 계층은 32차원 입력을 2차원 출력으로 변환한다.

---

## 4. Submodule을 등록하는 방법

PyTorch에서 Submodule을 등록하는 방법은 여러 가지가 있다.

대표적으로 다음 방법을 사용할 수 있다.

1. Instance Attribute로 등록
2. `add_module()`로 등록
3. `nn.ModuleList`로 등록
4. `nn.Sequential`로 등록

각 방식은 모델 구조를 작성하는 방법이 다르지만, 등록된 모듈의 Parameter를 PyTorch가 관리한다는 점은 동일하다.

---

### 4.1 Instance Attribute로 등록

가장 일반적인 방법은 `self`를 이용하여 계층을 속성으로 등록하는 것이다.

```python
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
```

여기서

```python
self.fc1 = nn.Linear(4, 8)
```

은 단순히 Python 변수에 객체를 저장하는 것 이상의 의미를 가진다.

`nn.Module`은 이처럼 등록된 Submodule을 내부적으로 관리한다.

그 결과 하위 계층의 Parameter도 `model.parameters()` 등을 통해 접근할 수 있게 된다.

---

### 4.2 add_module()로 등록

`add_module()`은 Submodule에 이름을 지정하여 등록하는 메서드이다.

기본 형태는 다음과 같다.

```python
self.add_module(name, module)
```

예를 들어

```python
self.add_module("fc1", nn.Linear(4, 8))
```

처럼 사용할 수 있다.

이는 다음 코드와 비슷한 역할을 한다.

```python
self.fc1 = nn.Linear(4, 8)
```

차이는 계층을 등록하는 코드를 동적으로 작성할 때 `add_module()`이 유용하다는 점이다.

---

### 4.3 add_module()을 이용한 ANN 구현

```python
class DynamicMLP(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.add_module(
            "fc1",
            nn.Linear(input_dim, 32)
        )

        self.add_module(
            "relu1",
            nn.ReLU()
        )

        self.add_module(
            "fc2",
            nn.Linear(32, output_dim)
        )

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x
```

모델을 생성한다.

```python
model = DynamicMLP(4, 2)

print(model)
```

`add_module()`을 통해 등록한 계층도 모델의 Submodule로 관리된다.

따라서 Parameter 조회, Device 이동, 모델 상태 저장과 같은 기능을 동일하게 사용할 수 있다.

---

## 5. 일반 Python List에 Layer를 저장하면 발생하는 문제

이 부분은 사용자 정의 신경망을 구현할 때 주의해야 하는 사항이다.

다음 코드를 살펴보자.

```python
class WrongModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = [
            nn.Linear(4, 8),
            nn.Linear(8, 2)
        ]

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
```

겉으로 보기에는 정상적인 신경망처럼 보인다.

실제로 `forward()`를 호출하면 연산을 수행할 수도 있다.

하지만 문제가 있다.

```python
model = WrongModel()

print(list(model.parameters()))
```

결과는 다음과 같다.

```text
[]
```

왜 이런 문제가 발생할까?

일반적인 Python `list`에 담긴 Module들은 `nn.Module`의 Submodule로 자동 등록되지 않기 때문이다.

따라서 PyTorch가 모델의 Parameter를 조회할 때 해당 계층들을 찾지 못한다.

이러한 상태에서는 다음 문제가 발생할 수 있다.

- Optimizer에 전달할 Parameter를 찾지 못함
- `model.to(device)`를 호출해도 해당 계층이 함께 이동하지 않음
- `state_dict()`에 해당 계층의 Parameter가 포함되지 않음

이 문제를 해결하려면 `nn.ModuleList`를 사용해야 한다.

---

## 6. nn.ModuleList

`nn.ModuleList`는 여러 Submodule을 리스트 형태로 관리하면서 PyTorch에 정상적으로 등록할 수 있도록 하는 클래스이다.

일반 Python `list`와 달리 내부 Module들이 등록된다.

```python
class CorrectModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(4, 8),
            nn.Linear(8, 2)
        ])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
```

이제 Parameter를 확인하면 다음과 같다.

```python
model = CorrectModel()

for name, param in model.named_parameters():
    print(name, param.shape)
```

출력은 다음과 같다.

```text
layers.0.weight torch.Size([8, 4])
layers.0.bias torch.Size([8])
layers.1.weight torch.Size([2, 8])
layers.1.bias torch.Size([2])
```

즉 `nn.ModuleList`에 포함된 계층의 Parameter가 정상적으로 등록된다.

#### 일반 List와 ModuleList 비교

| 구분 | Python List | nn.ModuleList |
| --- | --- | --- |
| Layer 저장 | 가능 | 가능 |
| Module 자동 등록 | 불가능 | 가능 |
| Parameter 추적 | 자동 추적 안 됨 | 가능 |
| Device 이동 | 자동 적용 안 됨 | 가능 |
| state_dict 포함 | 자동 포함 안 됨 | 가능 |
| 반복문 사용 | 가능 | 가능 |

여기서 `ModuleList`는 계층을 저장하고 등록하는 기능을 제공한다.

**계층을 자동으로 순차 실행하는 기능은 제공하지 않는다.**

따라서 실제 연산은 `forward()`에서 정의해야 한다.

---

## 7. nn.Sequential과 ModuleList의 차이

`nn.Sequential` 역시 여러 계층을 관리하는 클래스이다.

하지만 `ModuleList`와 역할이 다르다.

`ModuleList`는 계층들을 등록하고 보관하는 역할을 한다.

반면 `Sequential`은 계층들을 등록하면서, 정의한 순서대로 실행하는 `forward()`도 제공한다.

예를 들어 다음과 같다.

```python
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
```

이 모델은 입력 데이터가 다음 순서로 전달되도록 구성된다.

```text
Input
  │
  ▼
Linear(4, 32)
  │
  ▼
ReLU
  │
  ▼
Linear(32, 2)
  │
  ▼
Output
```

따라서 다음과 같이 직접 호출할 수 있다.

```python
x = torch.randn(5, 4)

output = model(x)

print(output.shape)
```

출력은 다음과 같다.

```text
torch.Size([5, 2])
```

#### 비교

| 구분 | ModuleList | Sequential |
| --- | --- | --- |
| Submodule 등록 | 가능 | 가능 |
| 순차 실행 기능 | 직접 구현 | 자동 제공 |
| 조건문 및 반복문 활용 | 유연함 | 상대적으로 제한적 |
| 적합한 구조 | 복잡하거나 동적인 구조 | 단순한 순차 구조 |

단순한 MLP에서는 `nn.Sequential`이 편리하다.

하지만 여러 경로로 분기되거나 조건에 따라 연산 방식이 달라지는 신경망에서는 `ModuleList`와 사용자 정의 `forward()`가 유용하다.

---

## 8. Parameter와 Buffer

`nn.Module`은 모델 내부의 Tensor를 용도에 따라 구분하여 관리할 수 있다.

대표적인 두 종류는 다음과 같다.

- Parameter
- Buffer

---

### 8.1 Parameter란?

Parameter는 모델에 등록되어 관리되는 Tensor이다.

일반적으로 신경망의 Weight와 Bias가 Parameter에 해당한다.

예를 들어

```python
layer = nn.Linear(4, 2)
```

라고 하면 내부적으로 다음과 같은 Parameter가 생성된다.

```python
print(layer.weight.shape)
print(layer.bias.shape)
```

출력은 다음과 같다.

```text
torch.Size([2, 4])
torch.Size([2])
```

Weight는 다음 형태의 행렬이다.

$$
W\in\mathbb{R}^{2\times4}
$$

Bias는 다음 형태의 벡터이다.

$$
b\in\mathbb{R}^{2}
$$

Linear Layer는 다음 연산을 수행한다.

$$
\boxed{ y=xW^T+b }
$$

이다.

일반적으로 이러한 Parameter들은 학습 과정에서 Optimizer에 의해 갱신된다.

다만 `nn.Parameter`라고 해서 반드시 학습되는 것은 아니다.

`requires_grad=False`로 설정하거나 Optimizer에서 제외하면 학습을 통한 갱신이 이루어지지 않을 수 있다.

---

### 8.2 nn.Parameter를 직접 등록하기

`nn.Parameter`를 사용하면 학습 가능한 Tensor를 직접 정의할 수 있다.

```python
class CustomLinear(nn.Module):

    def __init__(self):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(2, 4)
        )

        self.bias = nn.Parameter(
            torch.zeros(2)
        )

    def forward(self, x):

        return x @ self.weight.T + self.bias
```

여기서

```python
self.weight = nn.Parameter(...)
```

를 통해 Tensor가 모델의 Parameter로 등록된다.

따라서 다음과 같이 조회할 수 있다.

```python
model = CustomLinear()

for name, param in model.named_parameters():
    print(name, param.shape)
```

출력은 다음과 같다.

```text
weight torch.Size([2, 4])
bias torch.Size([2])
```

---

### 8.3 Buffer란?

Buffer는 모델에 등록되는 Tensor이지만, 일반적으로 Optimizer를 통한 학습 대상은 아니다.

대표적인 예시는 Batch Normalization에서 사용하는 Running Statistics이다.

예를 들어

- Running Mean
- Running Variance

등이 있다.

Buffer는 Parameter와 달리 Gradient 기반 최적화 대상이 아닐 수 있지만, 모델의 동작에 필요한 상태를 저장한다.

여기서 중요한 점은 **Buffer가 항상 값이 변하지 않는 상수라는 의미는 아니라는 것**이다.

예를 들어 Batch Normalization의 Running Mean과 Running Variance는 학습 과정에서 갱신될 수 있다.

다만 일반적인 Weight나 Bias처럼 Optimizer를 통해 학습되는 것은 아니다.

---

### 8.4 register_buffer() 사용하기

Buffer를 등록할 때는 `register_buffer()`를 사용한다.

```python
class BufferExample(nn.Module):

    def __init__(self):

        super().__init__()

        self.register_buffer(
            "scale",
            torch.tensor(2.0)
        )

    def forward(self, x):

        return x * self.scale
```

모델을 생성한다.

```python
model = BufferExample()

print(list(model.named_buffers()))
```

결과에는 `scale`이라는 이름의 Buffer가 포함된다.

반면

```python
print(list(model.parameters()))
```

의 결과는 빈 리스트이다.

```text
[]
```

즉 `scale`은 모델에 등록된 Buffer이지만 Parameter는 아니다.

기본적으로 등록된 Buffer는 `state_dict()`에 포함되며, `model.to(device)`를 사용하면 함께 이동한다.

`persistent=False`로 등록한 Buffer는 `state_dict()`에서 제외된다.

---

## 9. nn.Module의 주요 메서드

이제 `nn.Module`이 제공하는 주요 메서드를 살펴보자.

먼저 다음 모델을 기준으로 예제를 작성한다.

```python
import torch
import torch.nn as nn


class SimpleANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


model = SimpleANN()
```

---

### 9.1 parameters()

`parameters()`는 모델에 등록된 Parameter를 조회하는 메서드이다.

```python
for param in model.parameters():
    print(param.shape)
```

출력은 다음과 같다.

```text
torch.Size([8, 4])
torch.Size([8])
torch.Size([2, 8])
torch.Size([2])
```

출력된 Parameter는 다음과 같다.

1. fc1.weight
2. fc1.bias
3. fc2.weight
4. fc2.bias

`parameters()`는 기본적으로 현재 Module과 하위 Module들의 Parameter를 재귀적으로 조회한다.

```python
model.parameters(recurse=True)
```

여기서 `recurse=True`가 기본값이다.

반면

```python
model.parameters(recurse=False)
```

를 사용하면 현재 Module에 직접 등록된 Parameter만 조회한다.

---

### 9.2 named_parameters()

`named_parameters()`는 Parameter의 이름과 값을 함께 조회한다.

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

출력은 다음과 같다.

```text
fc1.weight torch.Size([8, 4])
fc1.bias torch.Size([8])
fc2.weight torch.Size([2, 8])
fc2.bias torch.Size([2])
```

`parameters()`와 달리 Parameter의 이름을 함께 확인할 수 있다.

따라서 특정 Layer의 Weight나 Bias를 확인할 때 유용하다.

#### Optimizer와의 관계

Optimizer는 일반적으로 다음과 같이 생성한다.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

이때 `model.parameters()`는 Optimizer에 모델의 Parameter들을 전달한다.

Optimizer는 전달받은 Parameter를 대상으로 업데이트를 수행한다.

즉,

```text
nn.Module
   │
   ▼
parameters()
   │
   ▼
Optimizer
   │
   ▼
Parameter Update
```

의 관계를 가진다.

---

### 9.3 named_buffers()

`named_buffers()`는 모델에 등록된 Buffer를 이름과 함께 조회한다.

```python
bn = nn.BatchNorm1d(8)

for name, buffer in bn.named_buffers():
    print(name, buffer.shape)
```

일반적으로 다음과 같은 Buffer를 확인할 수 있다.

```text
running_mean torch.Size([8])
running_var torch.Size([8])
num_batches_tracked torch.Size([])
```

여기서 `named_buffers()`의 반환값은 Dictionary 자체가 아니라 이름과 Buffer Tensor의 쌍을 순회하는 Iterator이다.

필요하다면 다음과 같이 Dictionary로 변환할 수 있다.

```python
buffer_dict = dict(bn.named_buffers())
```

---

## 10. children()과 modules()의 차이

`children()`과 `modules()`는 모두 모델 내부의 Module을 조회하는 메서드이다.

하지만 조회 범위가 다르다.

- `children()`: 직접 등록된 자식 Module 조회
- `modules()`: 자기 자신을 포함하여 전체 Module을 재귀적으로 조회

이 차이를 이해하기 위해 중첩된 모델을 만들어 보자.

---

### 10.1 중첩된 모델 정의

```python
class HiddenBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU()
        )

    def forward(self, x):

        return self.layers(x)


class NestedANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.block = HiddenBlock()
        self.output = nn.Linear(8, 2)

    def forward(self, x):

        x = self.block(x)
        x = self.output(x)

        return x
```

모델을 생성한다.

```python
model = NestedANN()

print(model)
```

구조는 다음과 같다.

```text
NestedANN
│
├── HiddenBlock
│   │
│   └── Sequential
│       ├── Linear(4, 8)
│       └── ReLU
│
└── Linear(8, 2)
```

---

### 10.2 children()

`children()`은 직접적인 자식 Module만 조회한다.

```python
for name, child in model.named_children():
    print(name, type(child).__name__)
```

출력은 다음과 같다.

```text
block HiddenBlock
output Linear
```

즉 `NestedANN`에 직접 등록된 두 Module만 나타난다.

`HiddenBlock` 내부의 `Sequential`, `Linear`, `ReLU`는 직접 조회되지 않는다.

이를 트리 구조로 나타내면 다음과 같다.

```text
NestedANN
│
├── HiddenBlock  ← children()
│
└── Linear       ← children()
```

---

### 10.3 modules()

`modules()`는 자기 자신을 포함하여 전체 Module을 재귀적으로 조회한다.

```python
for module in model.modules():
    print(type(module).__name__)
```

출력은 다음과 같다.

```text
NestedANN
HiddenBlock
Sequential
Linear
ReLU
Linear
```

즉 `modules()`는 중첩된 계층까지 모두 조회한다.

#### children()과 modules() 비교

| 구분 | children() | modules() |
| --- | --- | --- |
| 자기 자신 포함 | X | O |
| 직접 자식 조회 | O | O |
| 중첩된 하위 Module 조회 | X | O |
| 재귀적 탐색 | X | O |

이름까지 확인하려면 다음 메서드를 사용할 수 있다.

```python
model.named_children()
```
```python
model.named_modules()
```

두 메서드 역시 Iterator를 반환한다.

---

## 11. apply()를 이용한 Weight Initialization

신경망을 학습하기 전에는 Weight와 Bias의 초기값이 설정되어야 한다.

PyTorch의 `nn.Linear`와 같은 계층은 생성 시 기본 초기화를 수행한다.

하지만 사용자가 원하는 초기화 방식을 직접 적용할 수도 있다.

대표적인 초기화 방식은 다음과 같다.

- Xavier Initialization
- Kaiming Initialization
- Normal Initialization
- Uniform Initialization

이러한 초기화는 `torch.nn.init`에서 제공하는 함수를 사용할 수 있다.

---

### 11.1 torch.no_grad()의 역할

Weight Initialization은 일반적으로 Gradient를 계산할 필요가 없는 작업이다.

따라서 다음과 같이 `torch.no_grad()`를 사용할 수 있다.

```python
with torch.no_grad():

    nn.init.xavier_uniform_(model.output.weight)

    nn.init.zeros_(model.output.bias)
```

`torch.no_grad()`는 해당 구간의 연산을 Autograd 그래프에 기록하지 않도록 한다.

다만 PyTorch의 일반적인 `nn.init` 함수들은 내부적으로 Gradient 추적을 방지하도록 구현되어 있으므로, 초기화 함수 호출마다 별도로 `torch.no_grad()`를 작성해야 하는 것은 아니다.

`torch.no_grad()`는 초기화 외에도 Gradient가 필요하지 않은 Tensor 연산에서 활용할 수 있다.

---

### 11.2 apply()의 역할

`apply()`는 모델과 모든 하위 Module에 지정한 함수를 재귀적으로 적용한다.

예를 들어 다음 초기화 함수를 정의하자.

```python
def initialize_weights(module):

    if isinstance(module, nn.Linear):

        nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

이제 다음과 같이 적용할 수 있다.

```python
model = NestedANN()

model.apply(initialize_weights)
```

이때 `apply()`는 모델의 모든 하위 Module을 순회하면서 `initialize_weights()`를 호출한다.

그중 `nn.Linear`인 Module에 대해서만 초기화가 수행된다.

따라서 각 Linear Layer를 하나씩 찾아 초기화할 필요가 없다.

---

### 11.3 isinstance()를 사용하는 이유

다음 조건을 살펴보자.

```python
if isinstance(module, nn.Linear):
```

이는 `module`이 `nn.Linear`의 인스턴스이거나 그 하위 클래스의 인스턴스인지 확인한다.

다음과 같이 작성할 수도 있다.

```python
if type(module) == nn.Linear:
```

하지만 `type(module) == nn.Linear`는 정확히 `nn.Linear` 타입인 경우만 일치한다.

반면 `isinstance()`는 상속 관계까지 고려한다.

따라서 일반적으로 다음 방식이 더 유연하다.

```python
isinstance(module, nn.Linear)
```

---

### 11.4 Xavier와 Kaiming Initialization

Xavier Initialization과 Kaiming Initialization은 신경망의 Weight를 초기화하는 대표적인 방법이다.

**Xavier Initialization**

입력과 출력 차원을 고려하여 Weight의 분산을 조절하는 초기화 방식이다.

대표적으로 Tanh와 같은 활성화 함수에서 사용된다.

**Kaiming Initialization**

ReLU 계열 활성화 함수의 특성을 고려하여 Weight를 초기화하는 방식이다.

예를 들어

```python
nn.init.kaiming_uniform_(
    module.weight,
    nonlinearity="relu"
)
```

와 같이 사용할 수 있다.

초기화 방식은 사용하는 활성화 함수와 네트워크 구조에 따라 선택한다.

따라서 모든 계층에 동일한 초기화 방식을 적용하는 것이 항상 최선은 아니다.

---

## 12. state_dict()로 모델 상태 확인하기

`state_dict()`는 모델의 상태를 Dictionary 형태로 반환하는 메서드이다.

이 상태에는 일반적으로 다음 정보가 포함된다.

- 등록된 Parameter
- Persistent Buffer

반면 모델의 Python 클래스 정의나 `forward()` 코드 자체가 저장되는 것은 아니다.

---

### 12.1 state_dict() 사용하기

```python
model = SimpleANN()

state = model.state_dict()

for name, tensor in state.items():
    print(name, tensor.shape)
```

출력은 다음과 같다.

```text
fc1.weight torch.Size([8, 4])
fc1.bias torch.Size([8])
fc2.weight torch.Size([2, 8])
fc2.bias torch.Size([2])
```

여기서 각 Key는 Parameter 또는 Buffer의 이름이다.

각 Value는 해당 상태를 나타내는 Tensor이다.

즉 `state_dict()`는 모델의 학습된 Weight와 Bias 등을 저장하거나 복원할 때 사용된다.

---

### 12.2 state_dict()와 parameters()의 차이

두 메서드는 비슷해 보이지만 목적이 다르다.

| 구분 | parameters() | state_dict() |
| --- | --- | --- |
| 반환 형태 | Parameter Iterator | Dictionary 형태의 상태 |
| Parameter 조회 | O | O |
| Persistent Buffer 포함 | X | O |
| Optimizer에 전달 | 일반적으로 사용 | 일반적으로 사용하지 않음 |
| 모델 상태 저장 | 직접적인 주 용도는 아님 | 주로 사용 |

`parameters()`는 학습에 사용할 Parameter를 조회할 때 주로 사용한다.

반면 `state_dict()`는 모델 상태를 확인하고 저장하거나 복원할 때 사용한다.

---

### 12.3 keep_vars 옵션

`state_dict()`에는 다음 옵션이 존재한다.

```python
model.state_dict(keep_vars=False)
```

기본값은

```python
keep_vars=False
```

이다.

이 경우 반환된 Parameter와 Buffer는 Autograd 추적 관계에서 분리된 Tensor로 제공된다.

반면

```python
model.state_dict(keep_vars=True)
```

를 사용하면 해당 Tensor의 Autograd 관련 상태를 유지한다.

여기서 `keep_vars=True`는 모델 상태를 더 많이 저장하기 위한 옵션이라기보다, 반환된 상태 Tensor의 Autograd 관계를 유지할지를 결정하는 옵션이라고 이해하는 것이 정확하다.

일반적인 모델 저장 및 복원에서는 기본값을 사용하는 것으로 충분하다.

---

## 13. torch.save()와 load_state_dict()

PyTorch에서는 모델의 상태를 파일에 저장할 수 있다.

일반적으로 다음 두 메서드를 함께 사용한다.

- `torch.save()`
- `load_state_dict()`

---

### 13.1 모델 상태 저장하기

```python
model = SimpleANN()

torch.save(
    model.state_dict(),
    "model_weights.pth"
)
```

이 코드는 모델의 Parameter와 Persistent Buffer를 파일에 저장한다.

여기서 `.pth`는 PyTorch 모델 상태 파일에 흔히 사용하는 확장자이다.

---

### 13.2 모델 상태 불러오기

저장한 모델 상태를 복원할 때는 같은 구조의 모델을 먼저 생성한다.

```python
new_model = SimpleANN()
```

그다음 저장된 상태를 불러온다.

```python
state = torch.load(
    "model_weights.pth",
    map_location="cpu",
    weights_only=True
)

new_model.load_state_dict(state)
```

이제 `new_model`에 저장된 Weight와 Bias가 적용된다.

중요한 점은 **`state_dict()`에는 모델 구조 자체가 저장되지 않는다는 것**이다.

따라서 모델의 구조를 정의하는 클래스가 필요하다.

---

### 13.3 strict 옵션

`load_state_dict()`에는 `strict` 옵션이 있다.

```python
model.load_state_dict(
    state,
    strict=True
)
```

기본값은 `True`이다.

`strict=True`에서는 모델이 기대하는 상태의 Key와 불러오는 Dictionary의 Key가 일치해야 한다.

반면

```python
model.load_state_dict(
    state,
    strict=False
)
```

를 사용하면 일부 Key가 누락되거나 예상하지 못한 Key가 존재하는 경우에도 로딩을 진행할 수 있다.

다만 `strict=False`가 Tensor Shape의 불일치까지 무시하는 것은 아니다.

따라서 모델 구조가 달라진 상태에서 Weight를 불러올 때는 Key와 Shape를 함께 확인해야 한다.

---

## 14. CPU와 GPU로 모델 이동하기

PyTorch에서는 Tensor와 Model을 CPU 또는 GPU에서 실행할 수 있다.

이를 위해 `to()` 메서드를 사용한다.

---

### 14.1 CPU와 GPU 확인

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)
```

CUDA GPU가 사용 가능한 환경이라면 `cuda`가 선택된다.

그렇지 않으면 `cpu`가 선택된다.

참고로 macOS의 Apple Silicon 환경에서는 지원되는 경우 `mps` 장치를 사용할 수 있다.

---

### 14.2 모델 이동하기

```python
model = SimpleANN()

model = model.to(device)
```

이 코드는 모델의 등록된 Parameter와 Buffer를 지정된 장치로 이동시킨다.

---

### 14.3 입력 Tensor도 함께 이동해야 한다

모델을 GPU로 이동시켰다면 입력 Tensor도 같은 장치에 있어야 한다.

```python
x = torch.randn(5, 4)

x = x.to(device)

output = model(x)
```

모델은 GPU에 있는데 입력 데이터는 CPU에 있는 경우 Device Mismatch Error가 발생할 수 있다.

따라서 모델과 입력 데이터의 Device가 일치하도록 관리해야 한다.

---

### 14.4 cuda()와 cpu()

다음 메서드를 사용하여 장치를 이동할 수도 있다.

```python
model.cuda()
```

이는 모델을 CUDA GPU로 이동시킨다.

```python
model.cpu()
```

는 모델을 CPU로 이동시킨다.

하지만 다양한 장치를 지원하는 코드를 작성하려면 `to(device)`를 사용하는 것이 편리하다.

특히 CPU, CUDA, MPS 환경을 고려한다면 다음과 같이 작성할 수 있다.

```python
if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


model = model.to(device)
```

---

## 15. train()과 eval()의 차이

신경망에서는 학습할 때와 추론할 때 서로 다르게 동작하는 계층이 있다.

대표적으로 다음과 같다.

- Dropout
- Batch Normalization

이러한 계층의 동작 모드를 제어하기 위해 `train()`과 `eval()`을 사용한다.

---

### 15.1 train()

`train()`은 모델을 학습 모드로 설정한다.

```python
model.train()
```

이때

```python
model.training
```

의 값은 `True`가 된다.

학습 모드에서는 Dropout이 활성화되고, Batch Normalization이 학습 시 동작 방식에 따라 처리된다.

---

### 15.2 eval()

`eval()`은 모델을 평가 모드로 설정한다.

```python
model.eval()
```

이때

```python
model.training
```

의 값은 `False`가 된다.

평가 모드에서는 일반적으로 Dropout이 비활성화되고, Batch Normalization은 학습 과정에서 추적한 Running Statistics를 사용한다.

---

### 15.3 eval()과 no_grad()는 다르다

여기서 중요한 점이 있다.

```python
model.eval()
```

은 모델을 평가 모드로 변경하는 메서드이다.

하지만 **Gradient 계산을 자동으로 비활성화하는 기능은 아니다.**

Gradient 계산을 하지 않으려면 다음과 같이 사용할 수 있다.

```python
model.eval()

with torch.no_grad():
    output = model(x)
```

또는 일반적인 추론 코드에서는 다음을 사용할 수 있다.

```python
model.eval()

with torch.inference_mode():
    output = model(x)
```

두 기능의 역할을 구분하면 다음과 같다.

| 구분 | eval() | no_grad() |
| --- | --- | --- |
| 학습/평가 모드 변경 | O | X |
| Dropout 동작 변경 | O | X |
| BatchNorm 동작 변경 | O | X |
| Gradient 추적 비활성화 | X | O |

즉 모델 평가나 추론을 수행할 때는 일반적으로 `eval()`과 Gradient 추적을 비활성화하는 기능을 함께 사용한다.

---

## 16. forward()와 **call**()의 차이

`nn.Module`을 이해할 때 중요한 부분이다.

사용자 정의 모델에서는 `forward()`를 구현한다.

하지만 실제 모델을 사용할 때는 일반적으로 다음과 같이 호출한다.

```python
output = model(x)
```

그렇다면 왜 다음처럼 직접 호출하지 않을까?

```python
output = model.forward(x)
```

두 코드 모두 Forward 연산을 수행할 수 있지만, 내부적으로 실행되는 과정에는 차이가 있다.

---

### 16.1 forward()의 역할

`forward()`는 모델의 실제 연산을 정의한다.

예를 들어 다음과 같다.

```python
class MyModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(10, 1)

    def forward(self, x):

        print("forward called")

        return self.linear(x)
```

여기서 `forward()`는 입력 Tensor에 Linear 연산을 수행하는 역할을 한다.

---

### 16.2 model(x)를 호출하면?

모델을 생성한다.

```python
model = MyModel()
```

입력 데이터를 만든다.

```python
x = torch.randn(1, 10)
```

이제 다음과 같이 호출한다.

```python
output = model(x)
```

이 코드에서 실제로 실행되는 과정은 개념적으로 다음과 같다.

```text
model(x)
   │
   ▼
nn.Module.__call__()
   │
   ├── Forward Pre-Hook 처리
   │
   ▼
forward(x)
   │
   ▼
Forward Hook 처리
   │
   ▼
Output 반환
```

실제 PyTorch 내부 구현은 버전과 사용 중인 기능에 따라 더 복잡하다.

핵심은 `model(x)`를 호출하면 `nn.Module`에서 제공하는 호출 처리 과정이 실행되고, 그 내부에서 사용자가 정의한 `forward()`가 실행된다는 것이다.

---

### 16.3 왜 forward()를 직접 호출하지 않을까?

다음 두 코드를 비교해 보자.

```python
output = model(x)
```
```python
output = model.forward(x)
```

두 번째 코드도 `forward()`에서 정의한 연산 자체는 수행한다.

하지만 `nn.Module`의 표준 호출 과정에서 처리되는 일부 기능을 우회할 수 있다.

대표적으로 다음과 같은 기능에 영향을 줄 수 있다.

- Forward Pre-Hook
- Forward Hook
- Module 호출을 감싸는 프레임워크 기능

따라서 일반적인 PyTorch 코드에서는

```python
output = model(x)
```

형태를 사용하는 것이 권장된다.

또한 `forward()`를 직접 호출한다고 해서 Tensor 연산의 Autograd가 반드시 동작하지 않는 것은 아니다.

Autograd는 Tensor 연산을 추적하여 계산 그래프를 생성하기 때문이다.

다만 `nn.Module`의 표준 호출 처리 과정을 유지하기 위해 `model(x)`를 사용하는 것이 적절하다.

---

## 17. Hook이란?

Hook은 신경망의 특정 연산 시점에 사용자 정의 함수를 실행할 수 있도록 하는 기능이다.

대표적으로 다음과 같은 Hook이 있다.

- Forward Pre-Hook
- Forward Hook
- Backward Hook

이러한 기능을 사용하면 모델 내부의 입력, 출력 또는 Gradient 등을 관찰할 수 있다.

---

### 17.1 Forward Hook

Forward Hook은 Module의 Forward 연산이 수행된 후 실행된다.

예를 들어 다음과 같은 모델이 있다고 하자.

```python
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)
```

첫 번째 Linear Layer의 출력을 확인하고 싶다면 다음과 같이 작성할 수 있다.

```python
def print_output(module, inputs, output):

    print("Module:", module)

    print("Output Shape:", output.shape)
```

Hook을 등록한다.

```python
handle = model[0].register_forward_hook(
    print_output
)
```

입력 데이터를 전달한다.

```python
x = torch.randn(3, 4)

y = model(x)
```

첫 번째 Linear Layer의 출력 Shape는 다음과 같다.

```text
torch.Size([3, 8])
```

Hook이 실행되면서 해당 Layer의 정보를 확인할 수 있다.

사용이 끝난 Hook은 다음과 같이 제거한다.

```python
handle.remove()
```

Hook은 모델 내부 연산을 분석하거나 디버깅할 때 유용하다.

---

## 18. nn.Module의 전체 동작 흐름

지금까지 살펴본 내용을 하나의 흐름으로 정리하면 다음과 같다.

```text
Custom Model 정의
        │
        ▼
class CustomModel(nn.Module)
        │
        ▼
__init__()
        │
        ├── Layer 등록
        ├── Parameter 등록
        └── Buffer 등록
        │
        ▼
Model Instance 생성
        │
        ▼
model.to(device)
        │
        ▼
Optimizer 생성
        │
        ▼
Input Tensor
        │
        ▼
model(x)
        │
        ▼
nn.Module.__call__()
        │
        ▼
forward(x)
        │
        ▼
Output Tensor
        │
        ▼
Loss 계산
        │
        ▼
Backward Propagation
        │
        ▼
Optimizer Step
        │
        ▼
Parameter Update
```

여기서 `nn.Module`은 모델의 구조와 상태를 관리한다.

반면 Optimizer는 학습 과정에서 Parameter를 갱신하는 역할을 수행한다.

따라서 모델과 Optimizer는 서로 다른 역할을 가지지만, `model.parameters()`를 통해 연결된다.

---

## 19. 간단한 학습 과정 구현

마지막으로 `nn.Module`을 이용한 모델과 Optimizer가 어떻게 연결되는지 확인해 보자.

다음 예제에서는 앞서 구현한 `SimpleANN`을 사용한다.

```python
import torch
import torch.nn as nn

# 모델 생성
model = SimpleANN()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Loss Function
criterion = nn.MSELoss()

# 입력 데이터
x = torch.randn(10, 4)

# Target 데이터
target = torch.randn(10, 2)

# 학습 모드
model.train()

# Gradient 초기화
optimizer.zero_grad()

# Forward Propagation
output = model(x)

# Loss 계산
loss = criterion(output, target)

# Backward Propagation
loss.backward()

# Parameter Update
optimizer.step()

print("Loss:", loss.item())
```

각 단계의 역할은 다음과 같다.

#### ① Model 생성

```python
model = SimpleANN()
```

사용자 정의 신경망 인스턴스를 생성한다.

#### ② Optimizer 생성

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

모델에 등록된 Parameter를 Optimizer에 전달한다.

#### ③ Forward Propagation

```python
output = model(x)
```

입력 데이터가 신경망의 계층들을 통과하여 출력값을 생성한다.

#### ④ Loss 계산

```python
loss = criterion(output, target)
```

모델의 예측값과 Target 사이의 차이를 계산한다.

#### ⑤ Backward Propagation

```python
loss.backward()
```

Autograd를 이용하여 Loss에 대한 Parameter의 Gradient를 계산한다.

#### ⑥ Parameter Update

```python
optimizer.step()
```

계산된 Gradient를 이용해 Optimizer가 Parameter를 갱신한다.

즉 `nn.Module`이 등록하고 관리하는 Parameter들이 실제 학습 과정에서 사용된다.

---

## 20. 핵심 개념 비교

| 개념 | 역할 |
| --- | --- |
| `nn.Module` | 신경망 모델 및 계층의 기반 클래스 |
| `__init__()` | Layer와 Parameter 초기화 |
| `forward()` | Forward Propagation 정의 |
| `model(x)` | 표준 Module 호출 과정 실행 |
| `add_module()` | Submodule 등록 |
| `ModuleList` | 여러 Submodule을 리스트 형태로 등록 |
| `Sequential` | 순차적으로 연결된 Module 구성 |
| `Parameter` | 모델의 파라미터 등록 |
| `Buffer` | 학습 파라미터가 아닌 모델 상태 등록 |
| `parameters()` | Parameter 조회 |
| `named_parameters()` | Parameter 이름과 값 조회 |
| `children()` | 직접 등록된 자식 Module 조회 |
| `modules()` | 전체 Module 재귀 조회 |
| `apply()` | 함수를 Module에 재귀적으로 적용 |
| `state_dict()` | 모델의 Parameter 및 Persistent Buffer 상태 조회 |
| `load_state_dict()` | 모델 상태 복원 |
| `to()` | 모델 장치 및 자료형 이동 |
| `train()` | 학습 모드 설정 |
| `eval()` | 평가 모드 설정 |

---

## 21. 최종 정리

PyTorch에서 사용자 정의 신경망을 구현할 때는 일반적으로 `nn.Module`을 상속한다.

이를 통해 신경망의 계층을 구조화하고, Weight와 Bias 같은 Parameter를 관리할 수 있다.

사용자 정의 모델에서는 주로 다음 두 메서드를 구현한다.

```python
__init__()
```
```python
forward()
```

`__init__()`에서는 모델의 계층과 상태를 정의하고, `forward()`에서는 입력 데이터가 어떤 순서로 처리되는지 정의한다.

또한 계층을 등록할 때는 `self.linear = nn.Linear(...)`, `add_module()`, `ModuleList`, `Sequential` 등의 방식을 사용할 수 있다.

단, 일반 Python `list`에 Module을 저장하면 자동 등록되지 않을 수 있으므로 주의해야 한다.

등록된 Parameter는 다음과 같이 조회할 수 있다.

```python
model.parameters()
```

모델의 상태는 다음과 같이 조회한다.

```python
model.state_dict()
```

모델 상태를 저장할 때는 다음과 같이 사용한다.

```python
torch.save(
    model.state_dict(),
    "model_weights.pth"
)
```

또한 모델을 실행할 때는 일반적으로

```python
output = model(x)
```

형태로 호출한다.

이 방식은 PyTorch의 `nn.Module`이 제공하는 표준 호출 과정을 거쳐 `forward()`를 실행한다.

결과적으로 `nn.Module`은 **신경망 계층의 구성, Parameter 관리, Forward Propagation, 장치 이동, 상태 저장 및 복원을 통합적으로 지원하는 PyTorch의 핵심 클래스**이다.

---

### 핵심 요약

> PyTorch의 `nn.Module`은 단순히 신경망의 연산을 정의하는 클래스가 아니라, 모델의 계층과 파라미터를 등록하고 관리하며 학습과 추론에 필요한 다양한 기능을 제공하는 기반 클래스이다.

---

### 참고 자료

- [PyTorch Custom Model과 torch.nn.Module의 메서드들 — ds31x](https://ds31x.tistory.com/238)
- [PyTorch 공식 문서 — torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch 공식 문서 — ModuleList](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
- [PyTorch 공식 문서 — Serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
