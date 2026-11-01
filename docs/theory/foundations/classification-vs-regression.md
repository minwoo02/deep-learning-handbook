# Classification과 Regression

[TOC]


관련 실습: [scikit-learn으로 Perceptron 구현하기](../../practice/sklearn/sklearn-perceptron.md) · [scikit-learn MLPRegressor로 California Housing 집값 예측하기](../../practice/sklearn/california-housing-regression.md) · [PyTorch 딥러닝 모델 구현 실습: 선형회귀부터 ANN 학습·평가까지](../../practice/regression/linear-to-ann-regression.md)


---

## 지도학습의 두 가지 대표 Task: Classification과 Regression

지도학습(Supervised Learning)은 입력 데이터와 함께 정답(Label)을 제공하여 모델을 학습시키는 방법이다.

지도학습에서 해결하는 대표적인 Task는 크게 두 가지로 나눌 수 있다.

1. Classification
2. Regression

두 문제는 비슷한 신경망 구조를 사용할 수 있지만 **무엇을 예측하는가**에서 차이가 있다.

---

## 1. Classification

**분류(Classification)**는 입력 데이터가 어떤 Class에 속하는지를 예측하는 문제이다.

예를 들어

```text
Email
→ Spam / Not Spam

Image
→ Dog / Cat

Tumor Data
→ Benign / Malignant
```

과 같은 문제이다.

결과가 특정 범주 중 하나로 결정되기 때문에 출력값은 이산적(Discrete)이다.

---

### Binary Classification

Class가 두 개인 경우이다.

예를 들어

```text
0 → 정상
1 → 불량
```

또는

```text
0 → Spam 아님
1 → Spam
```

과 같은 문제이다.

이를 **Binary Classification**이라고 한다.

---

### Multiclass Classification

Class가 세 개 이상인 경우이다.

예를 들어 Iris 데이터가

```text
Setosa
Versicolor
Virginica
```

세 가지 품종 중 하나를 예측하는 문제라면 Multiclass Classification이다.

---

## 2. Regression

**회귀(Regression)**는 연속적인 숫자 값을 예측하는 문제이다.

예를 들어

```text
House Data
→ 450,000,000원

Sensor Data
→ 3.27 V

Weather Data
→ 24.8 °C
```

과 같은 문제이다.

결과가 특정 Class가 아니라 연속적인 실수값이다.

---

## 3. Classification과 Regression 비교

| 구분 | Classification | Regression |
| --- | --- | --- |
| 목적 | Class 예측 | 연속적인 값 예측 |
| 출력 | Discrete | Continuous |
| 예 | 정상/불량 | 온도 |
| 예 | 개/고양이 | 전압 |
| 예 | Spam/Not Spam | 가격 |

---

## 4. MLPClassifier

scikit-learn에서는 MLP를 Classification에 사용할 수 있도록

```python
sklearn.neural_network.MLPClassifier
```

를 제공한다.

예를 들어 Iris의 세 품종을 분류하고 싶다면 `MLPClassifier`를 사용할 수 있다.

기본적인 구조는

```text
Input
↓
Hidden Layer
↓
Non-linear Activation
↓
Output
↓
Class Prediction
```

이다.

---

## 5. MLPRegressor

Regression 문제에서는

```python
sklearn.neural_network.MLPRegressor
```

를 사용할 수 있다.

예를 들어 Sensor 값으로부터 연속적인 전압을 예측한다면 결과는

```text
3.21
3.45
2.98
...
```

와 같은 숫자이다.

Hidden Layer의 기본적인 구조는 MLPClassifier와 유사하지만 최종적으로 해결하려는 Task와 Output Layer의 동작 방식이 다르다.

---

## 6. 같은 MLP인데 무엇이 다른가?

MLPClassifier와 MLPRegressor 모두 기본적으로

```text
Input
↓
Linear
↓
Activation
↓
Linear
↓
Activation
↓
Output
```

과 같은 Feedforward Neural Network 구조를 사용할 수 있다.

차이는 마지막에 원하는 결과이다.

#### Classification

$$
\text{Input} \rightarrow \text{Class}
$$

#### Regression

$$
\text{Input} \rightarrow \text{Continuous Value}
$$

따라서 문제의 Target이 무엇인지 먼저 확인해야 한다.

---

## 7. Logistic Regression

이름 때문에 혼동하기 쉬운 모델이 **Logistic Regression**이다.

이름에는 Regression이 들어가지만 대표적인 **Classification Algorithm**이다.

기본식은 먼저 Linear Combination을 계산한다.

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

그리고 이를 Sigmoid Function에 통과시킨다.

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

Sigmoid의 출력은 0과 1 사이에 존재한다.

예를 들어

$$
\sigma(z)=0.82
$$

가 나왔다면 Class 1 쪽으로 높은 값을 출력한 것으로 해석할 수 있다.

---

## 8. Logistic Regression의 Decision Boundary

Binary Classification에서 Threshold를 0.5로 사용한다고 생각해보자.

```text
Probability >= 0.5 → Class 1
Probability < 0.5  → Class 0
```

Sigmoid에서

$$
\sigma(0)=0.5
$$

이므로 Decision Boundary는

$$
\mathbf{w}^T\mathbf{x}+b=0
$$

과 연결된다.

따라서 Logistic Regression 역시 기본적으로 Linear Decision Boundary를 사용하는 분류 모델이다.

---

## 9. Perceptron과 Logistic Regression

두 모델의 기본 구조는 비슷하게 볼 수 있다.

#### Perceptron

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

$$
\hat{y} = \operatorname{step}(z)
$$

출력은 0 또는 1과 같이 바로 결정된다.

#### Logistic Regression

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

$$
\hat{y} = \sigma(z)
$$

0과 1 사이의 연속적인 값을 출력한다.

즉 Linear Combination은 비슷하지만 마지막에 사용하는 함수와 학습 방식에 차이가 있다.

---

## 10. 문제를 보고 모델을 고르는 기준

Target이 다음과 같다면 Classification 문제이다.

```text
Dog / Cat
Normal / Abnormal
Spam / Not Spam
A / B / C
```

반면 다음과 같다면 Regression 문제이다.

```text
Temperature
Voltage
Price
Distance
Lifetime
```

따라서 모델을 선택하기 전에 가장 먼저 생각할 것은

> **내가 예측하려는 Target의 형태가 무엇인가?**

이다.

---

### 핵심 정리

지도학습의 대표적인 두 Task는

$$
\boxed{ \text{Classification} \quad \text{vs} \quad \text{Regression} }
$$

이다.

Classification은 **어떤 Class인가**를 예측하고,

Regression은 **값이 얼마인가**를 예측한다.

신경망 구조가 비슷하더라도 최종 출력의 형태와 학습 목적에 따라 모델의 성격이 달라진다.
