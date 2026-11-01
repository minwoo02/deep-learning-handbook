# PyTorch Dataset과 DataLoader 이해하기 — 클래스 상속부터 데이터 전처리까지

[TOC]


## 실습 구성표

| 항목 | 실습 구성 |
| --- | --- |
| Task | 데이터 파이프라인 · 복수 예제 |
| Model | Dataset / DataLoader |
| nn.Module | 모델 학습 해당 없음 |
| Autograd | 학습 코드 해당 없음 |
| Loss | 해당 없음 |
| Optimizer | 해당 없음 |
| Activation | 해당 없음 |



관련 이론: [Supervised Learning(지도학습) 완전 정리: Classification·Regression부터 Object Detection·Segmentation까지](../../theory/foundations/supervised-learning.md)


---

PyTorch에서 딥러닝 모델을 학습시키려면 입력 데이터(Feature)와 정답 데이터(Label)를 모델에 전달해야 한다.

이때 데이터의 관리와 로딩을 담당하는 것이 `Dataset`과 `DataLoader`이다.

이번 글에서는 Python의 클래스 상속(Inheritance)과 특수 메서드(Special Method)를 먼저 이해하고, 이를 바탕으로 Custom Dataset을 직접 구현하는 방법을 정리한다.

## 이번 실습의 데이터 처리 및 구현 구성

이번 실습에서는 PyTorch의 `Dataset` 클래스를 상속하여 **Custom Dataset을 구현**하고, `DataLoader`를 이용해 학습 데이터를 Mini-batch 단위로 불러오는 방법을 학습한다.

신경망 모델을 직접 구현하는 것이 아니라, **Model Training에 필요한 Input Feature와 Target Label을 체계적으로 관리하는 Data Pipeline**을 구성하는 것이 목적이다.

| 항목 | 사용 기술 및 구성 | 역할 |
| --- | --- | --- |
| Framework | PyTorch | Tensor 및 데이터 처리 |
| Data Structure | `torch.Tensor` | Feature와 Label을 Tensor로 표현 |
| Dataset Definition | `torch.utils.data.Dataset` 상속 | Custom Dataset 구현 |
| Initialization | `__init__()` | Dataset의 데이터와 속성 초기화 |
| Dataset Size | `__len__()` | 전체 Sample 수 반환 |
| Sample Access | `__getitem__()` | Index에 해당하는 Sample 반환 |
| Data Loading | `torch.utils.data.DataLoader` | Dataset에서 Mini-batch 생성 |
| Batch Processing | `batch_size` | 한 번에 불러올 Sample 수 설정 |
| Data Shuffling | `shuffle` | Dataset의 Sample 순서 섞기 |
| Data Iteration | `for` Loop | Mini-batch 단위로 데이터 순회 |

### 데이터 처리의 기본 구조

```text
Input Features (X) + Target Labels (y)
                  │
                  ▼
             Custom Dataset
             (Dataset 상속)
                  │
                  ├── __init__()
                  │     데이터 초기화
                  │
                  ├── __len__()
                  │     Sample 수 반환
                  │
                  └── __getitem__()
                        Sample 접근
                  │
                  ▼
               DataLoader
                  │
                  ├── batch_size
                  │
                  └── shuffle
                  │
                  ▼
             Mini-batch 생성
                  │
                  ▼
           Model Training에 활용
```

여기서 세 가지 개념을 구분해야 한다.

- **`Dataset`** 은 개별 Sample을 저장하고 접근하는 방법을 정의하는 클래스이다.
- **`DataLoader`** 는 Dataset을 이용하여 Mini-batch를 생성하고 데이터를 순회할 수 있도록 하는 도구이다.
- **`nn.Module`** 은 Neural Network의 구조와 Forward Propagation을 정의하기 위한 클래스이며, Dataset 및 DataLoader와는 역할이 다르다.

따라서 이번 글의 핵심은 **`Dataset`을 상속하여 Custom Dataset을 구현하고, `DataLoader`를 통해 데이터를 Mini-batch 단위로 구성하는 과정을 이해하는 것**이다.

---

## 1. Dataset과 DataLoader란?

PyTorch는 데이터 처리를 위해 두 가지 주요 클래스를 제공한다.

- **Dataset**: 개별 데이터 샘플을 관리하고 접근하는 객체
- **DataLoader**: Dataset의 샘플들을 배치(Batch) 단위로 묶어 모델에 제공하는 객체

전체적인 데이터 처리 흐름은 다음과 같다.

```text
Raw Data
   │
   ▼
Dataset
   │
   │ 개별 샘플 접근
   │
   ▼
DataLoader
   │
   │ Shuffle / Mini-batch 구성
   │
   ▼
Deep Learning Model
   │
   ▼
Prediction
   │
   ▼
Loss Calculation
```

예를 들어 다음과 같은 데이터가 있다고 가정하자.

| Index | Feature | Label |
| --- | --- | --- |
| 0 | [1.0, 2.0] | 0 |
| 1 | [3.0, 4.0] | 1 |
| 2 | [5.0, 6.0] | 0 |

여기서:

- `Feature`: 모델에 입력되는 데이터
- `Label`: 모델이 예측해야 하는 정답
- `Sample`: 하나의 Feature와 Label로 구성된 데이터 단위

`Dataset`은 각 샘플에 접근할 수 있도록 하고, `DataLoader`는 여러 샘플을 묶어 모델에 전달한다.

### 1.1 Dataset의 핵심 역할

Dataset은 단순히 데이터를 저장하는 객체가 아니다.

**Dataset의 가장 중요한 역할은 데이터에 대한 접근 인터페이스를 제공하는 것이다.**

예를 들어 다음과 같이 사용할 수 있어야 한다.

```python
dataset[0]  # 첫 번째 샘플
dataset[1]  # 두 번째 샘플
```

데이터가 메모리에 저장되어 있든, CSV 파일에 있든, 이미지 파일로 존재하든, 개발자는 동일한 방식으로 샘플에 접근할 수 있다.

이를 통해 모델 학습 코드와 실제 데이터 저장 방식을 분리할 수 있다.

---

## 2. Python 클래스 상속(Inheritance)

Custom Dataset을 만들기 위해서는 먼저 Python의 클래스 상속 개념을 이해해야 한다.

### 2.1 클래스 이름 뒤에 괄호를 사용하는 이유

다음 코드를 살펴보자.

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    pass
```

여기서 `CustomDataset(Dataset)`의 괄호는 **상속받을 부모 클래스를 지정하는 문법**이다.

- `Dataset`: 부모 클래스(Parent Class)
- `CustomDataset`: 자식 클래스(Child Class)
- `(Dataset)`: Dataset 클래스를 상속한다는 의미

즉, 기존 `Dataset` 클래스의 기능과 구조를 물려받아 새로운 클래스를 정의하는 것이다.

### 2.2 상속 예제

```python
class Animal:
    def breathe(self):
        print("숨을 쉰다")


class Dog(Animal):
    def bark(self):
        print("멍멍!")
```

`Dog` 클래스는 `Animal` 클래스를 상속받는다.

```python
dog = Dog()

dog.breathe()
dog.bark()
```

출력:

```text
숨을 쉰다
멍멍!
```

`Dog` 클래스에는 `breathe()` 메서드를 작성하지 않았지만, 부모 클래스에서 상속받았기 때문에 사용할 수 있다.

**상속은 기존 클래스의 기능을 재사용하면서 새로운 기능을 추가할 수 있도록 하는 객체지향 프로그래밍(OOP)의 핵심 개념이다.**

### 2.3 클래스 상속과 객체 생성의 차이

다음 두 코드에서 괄호의 의미는 서로 다르다.

```python
class CustomDataset(Dataset):
    pass
```

→ `Dataset` 클래스를 상속받아 새로운 클래스를 정의한다.

```python
dataset = CustomDataset()
```

→ `CustomDataset` 클래스로부터 인스턴스(Instance)를 생성한다.

| 문법 | 의미 |
| --- | --- |
| `class A(B):` | B 클래스를 상속받아 A 클래스 정의 |
| `a = A()` | A 클래스의 인스턴스 생성 |

---

## 3. 특수 메서드(Special Method)

Python에서는 메서드 이름의 양쪽에 언더바(`_`)를 두 개씩 사용하는 경우가 있다.

```python
__init__()
__len__()
__getitem__()
__str__()
```

이러한 메서드를 **특수 메서드(Special Method)** 또는 **던더 메서드(Dunder Method)**라고 한다.

Dunder는 Double Underscore의 줄임말이다.

### 3.1 언더바를 두 개씩 사용하는 이유

이는 Python에서 미리 정해 놓은 명명 규칙이다.

특정 문법이나 내장 함수를 사용하면 Python이 해당 이름의 메서드를 호출한다.

| 특수 메서드 | 역할 | 사용 예시 |
| --- | --- | --- |
| `__init__()` | 객체 초기화 | `obj = Class()` |
| `__len__()` | 길이 반환 | `len(obj)` |
| `__getitem__()` | 인덱스로 데이터 접근 | `obj[0]` |
| `__str__()` | 문자열 표현 반환 | `str(obj)` |
| `__iter__()` | 반복자 반환 | `iter(obj)` |

예를 들어:

```python
class MyDataset:
    def __getitem__(self, idx):
        return idx * 10


dataset = MyDataset()

print(dataset[2])
```

출력:

```text
20
```

Python은 `dataset[2]`라는 표현을 보고 `__getitem__()`을 호출한다.

```text
dataset[2]
    │
    ▼
__getitem__(2)
    │
    ▼
   20
```

반대로 메서드 이름을 `getitem()`으로 변경하면 어떨까?

```python
class MyDataset:
    def getitem(self, idx):
        return idx * 10


dataset = MyDataset()

dataset.getitem(2)  # 정상 작동
dataset[2]          # TypeError 발생
```

일반 메서드인 `getitem()`은 직접 호출해야 한다.

반면 `__getitem__()`은 Python의 인덱싱 문법과 연결되어 있다.

**특수 메서드는 Python에서 정해 놓은 문법과 객체 내부의 동작을 연결하는 역할을 한다.**

### 3.2 self는 무엇일까?

특수 메서드를 살펴보면 대부분 첫 번째 매개변수로 `self`를 사용한다.

```python
def __init__(self):
    self.data = [10, 20, 30]
```

여기서 `self`는 메서드가 호출된 **인스턴스 자기 자신을 참조하는 매개변수**이다.

예를 들어:

```python
class MyDataset:

    def __init__(self, data):
        self.data = data


dataset1 = MyDataset([10, 20, 30])
dataset2 = MyDataset([40, 50, 60])

print(dataset1.data)
print(dataset2.data)
```

출력:

```text
[10, 20, 30]
[40, 50, 60]
```

`self.data`는 각 인스턴스에 속하는 변수다.

따라서 서로 다른 인스턴스는 자신만의 데이터를 가질 수 있다.

참고로 `self`라는 이름은 Python에서 강제하는 키워드는 아니지만, 첫 번째 인스턴스 매개변수의 이름으로 사용하는 표준적인 관례이다.

---

## 4. 오버라이딩(Overriding)

오버라이딩은 **부모 클래스에 정의된 메서드를 자식 클래스에서 다시 구현하는 것**이다.

상속과 오버라이딩의 차이는 다음과 같다.

| 개념 | 의미 |
| --- | --- |
| Inheritance | 부모 클래스의 기능을 물려받는 것 |
| Overriding | 상속받은 메서드를 자식 클래스에서 재정의하는 것 |

### 4.1 오버라이딩 예제

```python
class Animal:

    def sound(self):
        print("동물의 소리")


class Dog(Animal):

    def sound(self):
        print("멍멍!")
```

`Dog` 클래스는 부모 클래스의 `sound()` 메서드를 재정의했다.

```python
dog = Dog()
dog.sound()
```

출력:

```text
멍멍!
```

PyTorch에서도 같은 원리를 사용한다.

```python
class CustomDataset(Dataset):

    def __len__(self):
        return 100
```

이 코드는 부모 클래스인 `Dataset`의 `__len__()`을 자식 클래스에서 구현하는 것이다.

PyTorch는 개발자가 사용하는 데이터의 실제 개수를 알 수 없으므로 데이터셋에 맞는 동작을 직접 정의해야 한다.

---

## 5. Custom Dataset 만들기

PyTorch에서는 다음 클래스를 상속하여 자신만의 Dataset을 만들 수 있다.

```python
from torch.utils.data import Dataset
```

일반적인 인덱스 기반 Custom Dataset은 다음 세 가지 메서드를 중심으로 구현한다.

| 메서드 | 역할 |
| --- | --- |
| `__init__()` | 데이터셋 초기화 |
| `__len__()` | 전체 샘플 개수 반환 |
| `__getitem__()` | 특정 인덱스의 샘플 반환 |

```python
class CustomDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
```

참고로 `__init__()`은 초기화가 필요한 경우 구현하는 메서드이며, 모든 Custom Dataset에서 반드시 오버라이딩해야 하는 것은 아니다.

### 5.1 **init**() — 데이터셋 초기화

`__init__()`은 인스턴스가 생성될 때 호출되는 초기화 메서드이다.

```python
def __init__(self, X, y):
    self.X = X
    self.y = y
```

주로 다음 정보를 저장한다.

- 입력 데이터(Feature)
- 정답 데이터(Label)
- 데이터가 저장된 파일 경로
- 전처리 함수(Transform)

필요한 매개변수는 개발자가 자유롭게 추가할 수 있다.

### 5.2 **len**() — 샘플 개수 반환

`__len__()`은 데이터셋에 포함된 전체 샘플 개수를 반환한다.

```python
def __len__(self):
    return len(self.X)
```

예를 들어:

```python
len(dataset)
```

Python은 데이터셋의 `__len__()` 메서드를 사용해 결과를 반환한다.

### 5.3 **getitem**() — 특정 샘플 반환

`__getitem__()`은 전달받은 인덱스에 해당하는 샘플을 반환한다.

```python
def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
```

일반적으로 하나의 샘플을 다음과 같이 반환한다.

```python
(feature, label)
```

반환 형식은 튜플뿐 아니라 딕셔너리도 사용할 수 있다.

```python
return {
    "feature": self.X[idx],
    "label": self.y[idx]
}
```

단, 이후 `DataLoader`에서 데이터를 배치로 구성할 때 반환 자료형에 따라 처리 방식이 달라질 수 있다.

---

## 6. Custom Dataset 구현 예제

실제 데이터를 사용하여 Custom Dataset을 만들어 보자.

### 6.1 전체 코드

```python
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, features, labels):

        # 입력 데이터
        self.X = torch.tensor(
            features,
            dtype=torch.float32
        )

        # 정답 데이터
        self.y = torch.tensor(
            labels,
            dtype=torch.long
        )

        if len(self.X) != len(self.y):
            raise ValueError(
                "Feature와 Label의 개수가 일치해야 합니다."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Raw Data
features = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0]
]

labels = [0, 1, 0, 1, 0]


# Dataset 생성
dataset = CustomDataset(features, labels)


# 전체 샘플 개수
print(len(dataset))


# 첫 번째 샘플
print(dataset[0])
```

출력:

```text
5
(tensor([1., 2.]), tensor(0))
```

여기서 중요한 점은 데이터의 자료형이다.

```python
dtype=torch.float32
```

입력 Feature를 실수형 Tensor로 변환한다.

```python
dtype=torch.long
```

정답 Label을 정수형 Tensor로 변환한다.

이는 클래스 인덱스를 정답으로 사용하는 일반적인 다중 클래스 분류 작업에서 자주 사용되는 설정이다.

회귀 문제나 다른 손실함수를 사용한다면 Label의 자료형도 달라질 수 있다.

### 6.2 데이터셋의 Shape 확인

```python
print(dataset.X.shape)
print(dataset.y.shape)
```

출력:

```text
torch.Size([5, 2])
torch.Size([5])
```

각 차원의 의미는 다음과 같다.

- `X.shape = [5, 2]`: 5개의 샘플, 샘플당 2개의 Feature
- `y.shape = [5]`: 5개의 Label

하나의 샘플에 접근하면:

```python
x, y = dataset[0]

print(x.shape)
print(y.shape)
```

출력:

```text
torch.Size([2])
torch.Size([])
```

`x`는 2개의 값을 가진 1차원 Tensor이다.

반면 `y`는 하나의 정답값을 가진 0차원 Tensor(Scalar Tensor)이다.

이 차이는 나중에 `DataLoader`를 통해 배치를 생성할 때 중요해진다.

---

## 7. Transform을 이용한 데이터 전처리

`transform`은 Dataset에서 샘플을 가져올 때 적용하는 전처리 함수 또는 객체이다.

주로 다음과 같은 작업에 사용한다.

- 정규화(Normalization)
- 크기 조절(Resize)
- 데이터 증강(Data Augmentation)
- 자료형 및 Tensor 변환

### 7.1 transform과 target_transform의 차이

| 매개변수 | 적용 대상 |
| --- | --- |
| `transform` | 입력 Feature |
| `target_transform` | 정답 Label |
| `transforms` | 입력과 정답을 함께 처리하도록 정의한 변환 |

주의할 점은 `Dataset`이 이러한 매개변수를 모두 자동으로 제공하는 것은 아니라는 것이다.

Custom Dataset에서는 필요한 매개변수를 직접 정의해야 한다.

특히 `transforms`라는 이름은 Dataset마다 지원 여부와 사용 방식이 다르다.

### 7.2 전처리가 포함된 Dataset

```python
class CustomDataset(Dataset):

    def __init__(
        self,
        features,
        labels,
        transform=None,
        target_transform=None
    ):

        self.X = torch.as_tensor(
            features,
            dtype=torch.float32
        )

        self.y = torch.as_tensor(
            labels,
            dtype=torch.long
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        # Feature 전처리
        if self.transform is not None:
            x = self.transform(x)

        # Label 전처리
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
```

여기서 전처리를 적용하는 시점에 주목해야 한다.

```text
__init__()
    │
    ▼
원본 데이터 저장
    │
    ▼
__getitem__(idx)
    │
    ▼
특정 샘플 가져오기
    │
    ▼
Transform 적용
    │
    ▼
전처리된 샘플 반환
```

즉, 모든 데이터를 미리 변환하는 대신, 필요한 샘플을 가져올 때 전처리를 적용할 수 있다.

### 7.3 입력 데이터 정규화 예제

```python
def normalize(x):
    return x / 10.0


dataset = CustomDataset(
    features,
    labels,
    transform=normalize
)

print(dataset[0])
```

출력:

```text
(tensor([0.1000, 0.2000]), tensor(0))
```

원래 입력 데이터는 다음과 같았다.

```text
[1.0, 2.0]
```

전처리 이후:

```text
[0.1, 0.2]
```

이 예제는 전처리 동작을 보여주기 위한 단순한 스케일 조정이다.

실제 정규화에서는 학습 데이터의 평균과 표준편차 등을 활용하는 경우가 많다.

### 7.4 입력과 정답을 함께 변환해야 하는 경우

이미지 분할(Segmentation)에서는 이미지와 정답 마스크에 동일한 위치의 자르기(Crop)나 뒤집기(Flip)를 적용해야 하는 경우가 있다.

이때는 입력과 정답을 동시에 전달받는 변환 함수를 사용할 수 있다.

```python
def joint_transform(x, y):
    # 입력과 정답을 함께 변환
    return x, y
```

Custom Dataset의 `__getitem__()`에서 다음과 같이 적용하도록 구현할 수 있다.

```python
if self.transforms is not None:
    x, y = self.transforms(x, y)
```

이러한 방식은 입력 데이터와 정답 사이의 공간적 대응 관계를 유지해야 하는 작업에 유용하다.

---

## 8. DataLoader 이해하기

`Dataset`이 개별 샘플을 제공한다면, `DataLoader`는 여러 샘플을 묶어 모델 학습에 적합한 형태로 제공한다.

```python
from torch.utils.data import DataLoader
```

### 8.1 DataLoader의 주요 파라미터

```python
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    collate_fn=None
)
```

각 파라미터의 의미는 다음과 같다.

| 파라미터 | 역할 |
| --- | --- |
| `dataset` | 데이터를 가져올 Dataset 객체 |
| `batch_size` | 한 번에 묶을 샘플 수 |
| `shuffle` | 샘플 순서를 무작위로 섞을지 결정 |
| `num_workers` | 데이터 로딩에 사용할 하위 프로세스 수 |
| `pin_memory` | CPU의 페이지 고정 메모리 사용 여부 |
| `drop_last` | 마지막 불완전한 배치 제거 여부 |
| `collate_fn` | 개별 샘플을 배치로 묶는 함수 |

#### batch_size

```python
batch_size=2
```

한 번에 2개의 샘플을 묶어서 반환한다.

#### shuffle

```python
shuffle=True
```

데이터를 순서대로 가져오지 않고 무작위로 섞는다.

일반적으로 학습 데이터에는 `shuffle=True`를 사용하고, 검증 및 테스트 데이터에는 `shuffle=False`를 사용하는 경우가 많다.

#### num_workers

```python
num_workers=0
```

데이터 로딩을 메인 프로세스에서 수행한다.

```python
num_workers=4
```

4개의 하위 프로세스를 사용하여 데이터 로딩을 병렬로 수행할 수 있다.

단, `num_workers`를 높인다고 항상 빨라지는 것은 아니다.

데이터 크기, 저장장치의 속도, 전처리 복잡도 등에 따라 적절한 값을 선택해야 한다.

처음 학습하거나 Jupyter Notebook에서 실습할 때는 `num_workers=0`부터 시작하는 것이 편리하다.

#### pin_memory

```python
pin_memory=True
```

주로 CUDA GPU로 데이터를 전송할 때 성능 개선을 위해 사용한다.

CPU 메모리의 특정 영역을 페이지 고정하여 데이터 전송을 효율적으로 수행할 수 있도록 한다.

GPU 메모리를 미리 예약한다는 의미는 아니다.

#### drop_last

```python
drop_last=True
```

마지막 배치가 설정한 `batch_size`보다 작으면 해당 배치를 버린다.

예를 들어 전체 샘플이 5개이고 `batch_size=2`라면:

```text
drop_last=False

Batch 1: 2개
Batch 2: 2개
Batch 3: 1개
```

반면:

```text
drop_last=True

Batch 1: 2개
Batch 2: 2개
```

마지막 샘플 1개가 제외된다.

### 8.2 DataLoader 실행 예제

앞에서 만든 Dataset을 사용한다.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for batch_idx, (X, y) in enumerate(loader):

    print(f"Batch {batch_idx}")

    print("Feature Shape:", X.shape)
    print("Label Shape:", y.shape)
```

출력:

```text
Batch 0
Feature Shape: torch.Size([2, 2])
Label Shape: torch.Size([2])

Batch 1
Feature Shape: torch.Size([2, 2])
Label Shape: torch.Size([2])

Batch 2
Feature Shape: torch.Size([1, 2])
Label Shape: torch.Size([1])
```

전체 샘플이 5개이고 배치 크기가 2이므로 총 3개의 배치가 생성된다.

각 Tensor의 Shape는 다음과 같다.

```text
Feature: [Batch Size, Feature Dimension]

Label:   [Batch Size]
```

이것이 데이터 로딩 과정에서 가장 중요한 변화 중 하나이다.

---

## 9. collate_fn이란?

`collate_fn`은 **Dataset에서 가져온 여러 샘플을 하나의 배치로 묶는 함수**이다.

일반적으로 `DataLoader`는 Dataset에서 샘플을 가져온 후 `collate_fn`을 사용해 배치를 구성한다.

### 9.1 collate_fn의 기본 동작

예를 들어 `Dataset`이 다음과 같은 샘플을 반환한다고 가정하자.

```python
dataset[0]
# (tensor([1., 2.]), tensor(0))

dataset[1]
# (tensor([3., 4.]), tensor(1))
```

`batch_size=2`이면 DataLoader는 두 샘플을 가져온 후 배치로 묶는다.

개념적으로 표현하면 다음과 같다.

```text
Dataset

Sample 0
 ├── Feature: [1.0, 2.0]
 └── Label: 0

Sample 1
 ├── Feature: [3.0, 4.0]
 └── Label: 1

          │
          ▼

       collate_fn

          │
          ▼

Mini-batch

Feature:
[[1.0, 2.0],
 [3.0, 4.0]]

Label:
[0, 1]
```

이를 Tensor의 Shape로 표현하면:

```text
개별 Feature: [2]
            ↓
배치 Feature: [2, 2]

개별 Label:   []
            ↓
배치 Label:   [2]
```

배치가 생성되면서 가장 앞에 **Batch Dimension**이 추가된다.

### 9.2 기본 collate_fn의 자료형 처리

DataLoader에서 `collate_fn=None`으로 설정하면 일반적인 자동 배치 구성 상황에서는 기본 `default_collate` 함수가 사용된다.

자료형에 따른 동작은 다음과 같다.

| 자료형 | 처리 방식 |
| --- | --- |
| Tensor | 새로운 배치 차원을 추가하여 쌓음 |
| NumPy ndarray | Tensor로 변환하여 묶음 |
| int, float | Tensor로 변환 |
| str | 문자열을 유지하여 리스트로 묶음 |
| dict | 동일한 Key의 값들을 각각 묶음 |
| list, tuple | 각 위치의 원소들을 재귀적으로 배치 처리 |

여기서 주의할 점은 **모든 자료형이 자동으로 Tensor로 변환되는 것은 아니라는 것**이다.

예를 들어 문자열은 문자열로 유지된다.

또한 PIL Image 객체는 기본 `collate_fn`에서 직접 처리할 수 없으므로 적절한 변환이 필요하다.

### 9.3 튜플을 반환했는데 리스트가 되는 이유

다음 코드를 살펴보자.

```python
import torch
from torch.utils.data import default_collate

batch = [
    (torch.tensor([1., 2.]), 0),
    (torch.tensor([3., 4.]), 1)
]

result = default_collate(batch)

print(result)
```

출력:

```text
[
    tensor([[1., 2.],
            [3., 4.]]),
    tensor([0, 1])
]
```

`Dataset`에서 개별 샘플을 튜플로 반환했지만, 기본 `collate_fn`은 일반 튜플을 리스트 형태로 묶는다.

그렇더라도 다음과 같은 언패킹(Unpacking)은 정상적으로 작동한다.

```python
X, y = result
```

따라서 DataLoader에서 흔히 사용하는 다음 코드에도 문제가 없다.

```python
for X, y in loader:
    ...
```

### 9.4 dict를 반환하는 경우

Dataset이 다음과 같이 딕셔너리를 반환한다고 가정하자.

```python
return {
    "feature": self.X[idx],
    "label": self.y[idx]
}
```

DataLoader는 같은 Key의 값들을 묶는다.

```python
{
    "feature": tensor([
        [1., 2.],
        [3., 4.]
    ]),
    "label": tensor([0, 1])
}
```

이 경우에는 다음과 같이 접근할 수 있다.

```python
for batch in loader:

    X = batch["feature"]
    y = batch["label"]
```

데이터가 여러 종류의 정보를 포함하는 경우 딕셔너리 방식이 편리할 수 있다.

---

## 10. Custom collate_fn이 필요한 경우

기본 `collate_fn`으로 처리하기 어려운 경우에는 직접 함수를 정의할 수 있다.

대표적으로 **샘플마다 데이터 길이가 다른 경우**가 있다.

예를 들어 다음과 같은 생체신호 데이터가 있다고 가정하자.

```text
Signal 1: [0.1, 0.2]

Signal 2: [0.3, 0.4, 0.5]
```

첫 번째 신호의 길이는 2이고, 두 번째 신호의 길이는 3이다.

이처럼 길이가 서로 다른 Tensor는 기본 `collate_fn`에서 `torch.stack()`으로 바로 묶을 수 없다.

따라서 짧은 신호에 Padding을 추가해 길이를 맞추는 방법을 사용할 수 있다.

### 10.1 Padding 예제

```python
import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


signals = [
    torch.tensor([0.1, 0.2]),
    torch.tensor([0.3, 0.4, 0.5])
]

labels = [0, 1]


def custom_collate_fn(batch):

    signals, labels = zip(*batch)

    # 원래 신호 길이 저장
    lengths = torch.tensor(
        [len(s) for s in signals]
    )

    # 길이가 다른 신호에 Padding 적용
    padded_signals = pad_sequence(
        signals,
        batch_first=True,
        padding_value=0.0
    )

    labels = torch.tensor(labels)

    return padded_signals, labels, lengths


dataset = list(zip(signals, labels))

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=custom_collate_fn
)


for X, y, lengths in loader:

    print(X)
    print(y)
    print(lengths)
```

출력:

```text
tensor([[0.1000, 0.2000, 0.0000],
        [0.3000, 0.4000, 0.5000]])

tensor([0, 1])

tensor([2, 3])
```

원래 길이가 2였던 신호에 `0`을 추가하여 길이를 3으로 맞춘 것이다.

또한 `lengths`를 함께 반환하므로 모델에서 실제 신호 길이와 Padding 영역을 구분하는 데 사용할 수 있다.

이러한 방식은 길이가 서로 다른 ECG, EEG 등의 생체신호나 자연어 데이터를 처리할 때 유용하다.

---

## 11. Dataset과 DataLoader의 내부 동작 흐름

앞에서 배운 내용을 하나의 과정으로 연결하면 다음과 같다.

```text
1. Raw Data 준비
         │
         ▼
2. CustomDataset 생성
         │
         └── __init__()
                데이터 초기화
         │
         ▼
3. DataLoader 생성
         │
         ├── batch_size
         ├── shuffle
         └── num_workers
         │
         ▼
4. DataLoader 반복 시작
         │
         ▼
5. 샘플 인덱스 선택
         │
         ▼
6. Dataset.__getitem__(idx)
         │
         ├── 샘플 읽기
         └── Transform 적용
         │
         ▼
7. collate_fn
         │
         └── 여러 샘플을 Mini-batch로 구성
         │
         ▼
8. Feature / Label 반환
         │
         ▼
9. Deep Learning Model
         │
         ▼
10. Loss 계산 및 학습
```

여기서 가장 중요한 점은 Dataset과 DataLoader의 역할이 분리되어 있다는 것이다.

Dataset은 개별 샘플을 어떻게 가져올지 정의한다.

DataLoader는 어떤 순서로 몇 개씩 가져와 배치를 만들지 담당한다.

이처럼 데이터 관리와 모델 학습을 분리하면 코드의 재사용성과 유지보수성이 좋아진다.

---

## 12. 자주 헷갈리는 개념

### Q1. Dataset은 반드시 Tensor를 반환해야 할까?

그렇지 않다.

Dataset은 Tensor뿐 아니라 숫자, 딕셔너리 등 다양한 자료형을 반환할 수 있다.

DataLoader의 기본 `collate_fn`은 지원되는 자료형에 대해 자동 변환 및 배치 구성을 수행한다.

하지만 모델에 입력할 Feature와 Label은 적절한 자료형의 Tensor로 준비하는 것이 일반적으로 편리하다.

### Q2. Dataset에서 바로 배치를 반환하면 안 될까?

일반적인 인덱스 기반 Dataset에서는 `__getitem__()`이 개별 샘플을 반환하도록 구현하는 것이 기본적인 방식이다.

여러 샘플을 묶는 작업은 DataLoader에 맡기는 것이 역할 분리 측면에서 적절하다.

다만 필요에 따라 배치 단위 데이터를 직접 생성하거나 여러 샘플을 한 번에 가져오는 방식도 구현할 수 있다.

### Q3. transform과 collate_fn은 무엇이 다를까?

가장 중요한 차이는 **전처리가 적용되는 단위**이다.

| 구분 | transform | collate_fn |
| --- | --- | --- |
| 적용 단위 | 개별 샘플 | 여러 샘플로 구성된 배치 |
| 호출 시점 | 일반적으로 `__getitem__()` 내부 | 배치 구성 시점 |
| 주요 용도 | 정규화, 데이터 증강 등 | Padding, Batch 구성 등 |

정리하면:

```text
개별 Sample
    │
    ▼
Transform
    │
    ▼
전처리된 Sample
    │
    ▼
collate_fn
    │
    ▼
Mini-batch
```

### Q4. Dataset과 IterableDataset은 무엇이 다를까?

PyTorch는 데이터 접근 방식에 따라 크게 두 가지 유형을 제공한다.

| 구분 | Dataset (Map-style) | IterableDataset |
| --- | --- | --- |
| 접근 방식 | 인덱스 기반 | 순차적인 데이터 반복 |
| 핵심 메서드 | `__getitem__()`, `__len__()` | `__iter__()` |
| 적합한 경우 | 이미지, CSV, 정해진 샘플 | 실시간 스트리밍, 연속 데이터 |

예를 들어 저장된 ECG 데이터 파일에서 특정 샘플을 가져오는 경우에는 일반적인 `Dataset`을 사용할 수 있다.

반면 센서에서 연속적으로 들어오는 신호를 처리한다면 `IterableDataset`이 더 적합할 수 있다.

다만 스트리밍 데이터를 여러 Worker로 읽는 경우 중복 데이터가 발생하지 않도록 별도의 처리가 필요할 수 있다.

---

## 13. 핵심 정리

| 개념 | 설명 |
| --- | --- |
| Class | 객체를 생성하기 위한 설계도 |
| Instance | 클래스로부터 생성된 객체 |
| Inheritance | 부모 클래스를 상속받는 것 |
| Overriding | 부모 클래스의 메서드를 재정의하는 것 |
| Dunder Method | Python에서 특별한 의미로 사용하는 메서드 |
| `self` | 현재 인스턴스를 참조하는 매개변수 |
| Dataset | 데이터 접근 인터페이스를 제공하는 객체 |
| DataLoader | Dataset으로부터 배치를 구성하여 제공하는 객체 |
| Transform | 개별 샘플에 적용하는 전처리 |
| collate_fn | 여러 샘플을 배치로 묶는 함수 |
| Batch Size | 하나의 배치에 포함되는 샘플 수 |
| Shuffle | 데이터 순서를 무작위로 섞는 기능 |
| Padding | 길이가 다른 데이터를 같은 길이로 맞추는 처리 |

### 최종 요약

```python
class CustomDataset(Dataset):
```

→ `Dataset`을 상속받는다.

```python
def __init__(self):
```

→ 데이터셋을 초기화한다.

```python
def __len__(self):
```

→ 전체 샘플 개수를 반환한다.

```python
def __getitem__(self, idx):
```

→ 특정 인덱스의 샘플을 반환한다.

```python
dataset = CustomDataset(...)
```

→ Dataset 인스턴스를 생성한다.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

→ 샘플을 배치 단위로 제공하도록 설정한다.

```python
for X, y in loader:
    ...
```

→ 각 배치를 가져와 모델 학습에 사용할 수 있다.

결국 PyTorch의 Dataset과 DataLoader는 **데이터를 관리하는 과정과 모델을 학습시키는 과정을 분리하기 위한 핵심 구성 요소**이다.

Dataset은 데이터 접근과 샘플 단위 전처리를 담당하고, DataLoader는 샘플 선택과 배치 구성 등을 담당한다.

이러한 구조를 이해하면 이미지, 텍스트, 시계열 데이터뿐 아니라 ECG, EEG와 같은 생체신호 데이터도 동일한 원리로 처리할 수 있다.

---

## 참고 자료

1. [PyTorch 공식 문서 — Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
2. [PyTorch 공식 문서 — torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
3. [PyTorch 공식 튜토리얼 — Writing Custom Datasets, DataLoaders and Transforms](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html)
