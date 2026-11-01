# 딥러닝을 공간 변환 관점에서 이해하기

[TOC]



---

## 딥러닝은 좌표 변환기인가?

딥러닝을 수식만으로 보면 수많은 행렬 곱셈과 활성화 함수가 반복되는 복잡한 계산처럼 보인다.

하지만 이를 기하학적으로 바라보면 조금 다른 직관을 얻을 수 있다.

딥러닝의 각 Layer는 입력 데이터를 계속 새로운 표현 공간으로 변환한다.

따라서 딥러닝을 이해하기 위한 하나의 직관으로

> **복잡한 데이터를 분류하기 쉬운 공간으로 변환하는 과정**

이라고 생각할 수 있다.

---

## 1. Linear Transformation

신경망 Layer의 기본적인 계산은 다음과 같다.

$$
\mathbf{z} = \mathbf{W}\mathbf{x}+\mathbf{b}
$$

여기서

$$
\mathbf{W}\mathbf{x}
$$

는 행렬 곱이다.

선형대수학에서 행렬은 벡터 공간을 변환하는 연산자로 해석할 수 있다.

행렬에 따라 공간을

- 회전(Rotation)
- 확대 또는 축소(Scaling)
- 기울이기(Shearing)

등의 방식으로 변환할 수 있다.

따라서 Weight Matrix $\mathbf{W}$는 입력 데이터가 존재하는 공간을 다른 형태로 변환하는 역할을 한다고 볼 수 있다.

---

### 2. Bias

다음 항은 Bias이다.

$$
+\mathbf{b}
$$

Bias는 변환된 공간을 이동시키는 역할을 한다.

따라서

$$
\mathbf{z} = \mathbf{W}\mathbf{x}+\mathbf{b}
$$

는 단순한 행렬 곱뿐 아니라 이동까지 포함한다.

엄밀하게는 $b$가 포함된 이러한 변환을 **Affine Transformation**이라고 한다.

딥러닝 프레임워크에서는 일반적으로 이러한 연산을 `Linear Layer` 또는 `Dense Layer`라는 이름으로 사용한다.

---

## 3. 하지만 Linear Transformation만으로는 부족하다

행렬 곱은 공간을

- 회전하고
- 늘리고
- 줄이고
- 기울일 수 있다.

하지만 이러한 연산을 여러 번 수행해도 전체 변환은 여전히 선형적인 성질을 가진다.

예를 들어

$$
\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1 \mathbf{x}
$$

는

$$
\mathbf{W}_{new}\mathbf{x}
$$

로 합칠 수 있다.

따라서 복잡하게 얽혀 있는 데이터를 선형 변환만으로는 원하는 형태로 충분히 변형하기 어렵다.

---

## 4. Activation Function의 역할

여기서 비선형 활성화 함수가 등장한다.

$$
\mathbf{a} = f(\mathbf{z})
$$

예를 들어 ReLU는 다음과 같다.

$$
f(z)=\max(0,z)
$$

ReLU는 음수 영역을 0으로 만들고 양수 영역은 그대로 전달한다.

이를 공간의 관점에서 생각하면 데이터 공간의 일부를 꺾거나 압축하는 비선형 변환으로 볼 수 있다.

Linear Layer가 공간을 선형적으로 변환한다면 Activation Function은 그 공간에 **비선형성(Non-linearity)**을 추가한다.

---

## 5. Layer 하나의 역할

따라서 하나의 신경망 Layer를 다음처럼 생각할 수 있다.

```text
Input Representation
        ↓
Wx + b
        ↓
Linear / Affine Transformation
        ↓
Activation Function
        ↓
Non-linear Transformation
        ↓
New Representation
```

이 과정이 여러 번 반복된다.

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

각 Layer를 통과할 때마다 데이터의 표현 방식이 변한다.

---

## 6. 구겨진 공간을 펴는 직관

비선형 분류 문제를 하나의 구겨진 종이로 생각해볼 수 있다.

원래 공간에서는 두 클래스가 복잡하게 섞여 있다고 가정한다.

```text
○ ● ○
● ○ ●
○ ● ○
```

이 상태에서는 직선 하나로 두 클래스를 나누기 어렵다.

신경망은 Layer를 통과시키면서 데이터의 표현을 계속 변화시킨다.

```text
Complex Representation
        ↓
Layer 1
        ↓
Layer 2
        ↓
Layer 3
        ↓
Simpler Representation
```

최종적으로는 다음과 같이 두 그룹이 보다 쉽게 나뉘는 표현을 얻는 것이 목표가 될 수 있다.

```text
○ ○ ○ ○ | ● ● ● ●
```

그러면 마지막 단계에서는 단순한 Linear Classifier만으로도 두 클래스를 구분할 수 있다.

---

## 7. Feature Representation

딥러닝에서 중요한 것은 원본 데이터 자체뿐만 아니라 데이터가 내부에서 **어떤 표현(Representation)**으로 변환되는가이다.

예를 들어 이미지 데이터를 생각하면 초기에는 각 Pixel의 값만 존재한다.

깊은 네트워크에서는 Layer를 거치면서 단순한 값들이 보다 의미 있는 특징으로 변환된다.

```text
Raw Input
   ↓
Low-level Feature
   ↓
Intermediate Feature
   ↓
High-level Representation
   ↓
Prediction
```

즉 신경망은 데이터를 단순히 분류하는 것뿐 아니라 **분류하기 좋은 표현 자체를 학습한다**고 볼 수 있다.

---

## 8. 딥러닝을 좌표 변환기로 보는 직관

이 관점에서 딥러닝을 다음과 같이 생각할 수 있다.

$$
\mathbf{x} \rightarrow f_1(\mathbf{x}) \rightarrow f_2(f_1(\mathbf{x})) \rightarrow \cdots \rightarrow f_L(\cdots)
$$

원래 데이터 공간의 좌표 $\mathbf{x}$를 여러 번 변환하여 새로운 표현 공간으로 Mapping하는 것이다.

따라서

> **딥러닝은 복잡한 데이터를 판단하기 쉬운 표현 공간으로 계속 Mapping하는 시스템이다.**

라는 직관을 사용할 수 있다.

다만 이것은 딥러닝의 엄밀한 정의라기보다, 신경망 내부에서 이루어지는 표현 변환을 이해하기 위한 기하학적 관점이다.

---

### 핵심 정리

신경망의 기본 연산

$$
\mathbf{z} = \mathbf{W}\mathbf{x}+\mathbf{b}
$$

에서 $\mathbf{W}$와 $\mathbf{b}$는 데이터 표현을 변환한다.

그리고

$$
\mathbf{a}=f(\mathbf{z})
$$

의 비선형 활성화 함수가 추가되면서 단순한 선형 변환을 넘어 복잡한 표현 변환이 가능해진다.

이를 여러 Layer에 걸쳐 반복하면

$$
\boxed{ \text{Complex Input Space} \rightarrow \text{Learned Representation} \rightarrow \text{Simple Decision} }
$$

이라는 관점으로 딥러닝을 이해할 수 있다.
