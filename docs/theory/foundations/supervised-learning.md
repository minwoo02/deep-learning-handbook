# Supervised Learning(지도학습) 완전 정리: Classification·Regression부터 Object Detection·Segmentation까지

[TOC]


관련 실습: [PyTorch Dataset과 DataLoader 이해하기 — 클래스 상속부터 데이터 전처리까지](../../practice/pytorch-basics/datasets-and-dataloaders.md)


---

Machine Learning(머신러닝)은 컴퓨터가 데이터를 이용하여 특정 Task를 수행하는 데 필요한 패턴이나 규칙을 학습하는 기술이다.

머신러닝은 학습에 사용되는 데이터의 형태와 학습 방식에 따라 여러 종류로 구분된다.

대표적으로 다음과 같다.

- Supervised Learning(지도학습)
- Unsupervised Learning(비지도학습)
- Semi-supervised Learning(반지도학습)
- Self-supervised Learning(자기지도학습)
- Reinforcement Learning(강화학습)

그중 **Supervised Learning은 Input Data와 이에 대응하는 정답(Label)을 함께 사용하여 학습하는 방법**이다.

지도학습의 대표적인 Task는 Classification(분류)과 Regression(회귀)이며, Deep Learning에서는 Object Detection과 Segmentation 등으로 확장된다.

이 글에서는 Supervised Learning의 기본적인 개념과 학습 원리를 이해하고, 주요 Task와 대표적인 알고리즘을 정리한다.

---

## 1. Supervised Learning이란?

Supervised Learning(지도학습)은 **입력 데이터와 이에 대응하는 정답 데이터를 이용하여 입력과 출력 사이의 관계를 학습하는 머신러닝 방법**이다.

일반적으로 학습 데이터는 다음과 같은 Pair로 구성된다.

$$
\boxed{ (x_i,y_i) }
$$

여기서

- $x_i$: Input Data
- $y_i$: Label 또는 Target
- $i$: Sample Index

이다.

전체 Dataset은 다음과 같이 표현할 수 있다.

$$
\boxed{ D=\{(x_i,y_i)\}_{i=1}^{N} }
$$

여기서 $N$은 Sample의 개수이다.

즉 Supervised Learning에서는 입력과 정답이 짝을 이루는 데이터가 필요하다.

예를 들어 다음과 같은 데이터가 있다고 하자.

| Input Data | Label |
| --- | --- |
| 고양이 이미지 | Cat |
| 강아지 이미지 | Dog |
| 고양이 이미지 | Cat |
| 강아지 이미지 | Dog |

모델은 이러한 데이터를 이용하여 이미지의 특징과 Label 사이의 관계를 학습한다.

이후 새로운 이미지가 입력되면 학습한 관계를 이용하여 해당 이미지가 Cat인지 Dog인지 예측한다.

전체 과정은 다음과 같다.

```text
          Labeled Dataset
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
   Input Data           Label
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
          Model Training
                 │
                 ▼
           Trained Model
                 │
                 ▼
           New Input Data
                 │
                 ▼
              Prediction
```

핵심은 **정답이 포함된 데이터로부터 Input과 Output 사이의 Mapping을 학습한다는 것**이다.

---

## 2. Input, Feature, Label, Target

Supervised Learning을 이해하기 위해서는 관련 용어를 구분해야 한다.

### 2.1 Input Data

Input Data는 모델이 예측을 수행하기 위해 전달받는 데이터이다.

일반적으로 다음과 같이 표현한다.

$$
\boxed{ \mathbf{x} }
$$

입력은 하나의 값일 수도 있고, 여러 Feature로 구성된 Vector일 수도 있다.

예를 들어 다음과 같다.

$$
\mathbf{x} = \begin{bmatrix} x_1\\ x_2\\ x_3\\ x_4 \end{bmatrix}
$$

여기서 각 $x_j$는 하나의 Feature이다.

---

### 2.2 Feature

Feature는 입력 데이터를 표현하는 개별적인 특성 또는 변수이다.

예를 들어 Iris Dataset에서는 다음 네 가지 Feature를 사용한다.

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

하나의 Sample은 다음과 같이 표현할 수 있다.

$$
\boxed{ \mathbf{x} = (x_1,x_2,x_3,x_4) }
$$

여러 Sample을 행으로 쌓으면 다음과 같은 Feature Matrix가 된다.

$$
\boxed{ X\in\mathbb{R}^{N\times d} }
$$

여기서

- $N$: Sample 수
- $d$: Feature 수

이다.

예를 들어 150개의 Sample에 각각 4개의 Feature가 존재한다면

$$
\boxed{ X\in\mathbb{R}^{150\times4} }
$$

이다.

---

### 2.3 Label과 Target

Label 또는 Target은 모델이 예측해야 하는 정답 데이터이다.

일반적으로 다음과 같이 표현한다.

$$
\boxed{ y }
$$

Classification에서는 Target이 주로 Class를 의미한다.

예를 들어 다음과 같다.

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

Regression에서는 Target이 연속적인 수치일 수 있다.

예를 들어 다음과 같다.

```text
Input Features → Temperature: 25.3

Input Features → Blood Pressure: 120.5
```

즉 Label과 Target은 모두 학습에 사용되는 정답을 의미하지만, **Classification에서는 Class Label, Regression에서는 Numerical Target이라는 표현을 자주 사용한다.**

---

## 3. Supervised Learning의 수학적 의미

지도학습의 목적은 Input과 Output 사이의 관계를 나타내는 함수를 학습하는 것이다.

이상적인 관계를 다음과 같이 표현하자.

$$
\boxed{ y=f(x) }
$$

하지만 실제 데이터에서는 입력과 출력 사이의 정확한 함수를 알 수 없는 경우가 많다.

따라서 모델을 이용하여 이 함수를 근사한다.

$$
\boxed{ \hat{y}=f_\theta(x) }
$$

여기서

- $f_\theta$: Machine Learning Model
- $\theta$: Model Parameters
- $x$: Input
- $\hat{y}$: Predicted Output

이다.

모델 학습의 목적은 다음 관계가 성립하도록 적절한 Parameter를 찾는 것이다.

$$
\boxed{ f_\theta(x_i)\approx y_i }
$$

즉 Training Dataset에 존재하는 Input–Output 관계를 바탕으로 새로운 입력에 대해서도 적절한 출력을 생성하도록 학습한다.

이를 Function Approximation(함수 근사) 관점에서 이해할 수 있다.

---

## 4. Supervised Learning의 주요 Tasks

Supervised Learning의 대표적인 Task는 다음 두 가지이다.

1. Classification(분류)
2. Regression(회귀)

두 Task 모두 Labeled Dataset을 사용하지만 예측하는 Output의 형태가 다르다.

```text
             Supervised Learning
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
      Classification         Regression
            │                   │
            ▼                   ▼
      Discrete Label      Continuous Value
            │                   │
            ▼                   ▼
      Class Prediction    Numerical Prediction
```

---

## 5. Classification(분류)

Classification은 **입력 데이터를 미리 정의된 Class 중 하나 또는 여러 개로 분류하는 Task**이다.

Classification에서 Target은 일반적으로 Discrete(이산적인) Category이다.

예를 들어 다음과 같다.

- 이메일 → Spam / Normal
- 이미지 → Cat / Dog
- Iris Dataset → Setosa / Versicolor / Virginica
- 의료영상 → Normal / Abnormal

즉 모델은 입력 데이터에 대해 적절한 Class를 할당한다.

---

### 5.1 Classification의 수학적 표현

$K$개의 Class가 있다고 하자.

각 Class는 다음과 같이 정의할 수 있다.

$$
\boxed{ y\in\{0,1,\cdots,K-1\} }
$$

모델은 입력 $x$에 대해 각 Class의 확률을 예측할 수 있다.

$$
\boxed{ P_\theta(Y=k\mid X=x) }
$$

Multiclass Classification에서 최종 예측 Class는 다음과 같이 결정할 수 있다.

$$
\boxed{ \hat{y} = \underset{k}{\arg\max} P_\theta(Y=k\mid X=x) }
$$

즉 가장 높은 확률을 가진 Class를 선택한다.

---

### 5.2 Classification 예제

의료영상에서 네 가지 Class를 분류하는 모델을 생각하자.

```text
Classes

0 → Cancer
1 → Precancer
2 → Inflammation
3 → Normal
```

모델이 다음과 같은 확률을 출력했다고 하자.

$$
\boxed{ \mathbf{p} = (0.70,0.15,0.10,0.05) }
$$

각 확률은 다음을 의미한다.

| Class | Predicted Probability |
| --- | --- |
| Cancer | 0.70 |
| Precancer | 0.15 |
| Inflammation | 0.10 |
| Normal | 0.05 |

이때 가장 높은 확률을 가진 Class는 Cancer이다.

따라서

$$
\boxed{ \hat y=\text{Cancer} }
$$

로 예측할 수 있다.

여기서 0.70은 모델이 출력한 예측 확률이다.

다만 **모델이 0.70을 출력했다고 해서 실제로 정답일 확률이 정확히 70%라고 단정할 수는 없다.**

예측 확률을 신뢰도로 해석하려면 모델의 Calibration(확률 보정 상태)도 고려해야 한다.

---

## 6. Classification의 종류

Classification은 Output Label의 구조에 따라 여러 종류로 구분할 수 있다.

대표적으로 다음과 같다.

### 6.1 Binary Classification

Binary Classification은 두 개의 Class 중 하나를 예측하는 문제이다.

$$
\boxed{ y\in\{0,1\} }
$$

예를 들어 다음과 같다.

```text
Input
  │
  ▼
Model
  │
  ├── Normal (0)
  │
  └── Abnormal (1)
```

대표적인 예시는 정상·비정상 분류, 스팸 탐지 등이 있다.

---

### 6.2 Multiclass Classification

Multiclass Classification은 세 개 이상의 Class 중 하나를 선택하는 문제이다.

예를 들어 Iris Dataset은 세 가지 Class를 가진다.

$$
\boxed{ y\in\{0,1,2\} }
$$

```text
Input
  │
  ▼
Model
  │
  ├── Setosa
  ├── Versicolor
  └── Virginica
```

각 Sample에는 하나의 Class가 할당된다.

---

### 6.3 Multilabel Classification

Multilabel Classification은 하나의 Sample에 여러 Class가 동시에 할당될 수 있는 문제이다.

예를 들어 하나의 이미지에 다음 객체가 존재한다고 하자.

```text
Image
  │
  ├── Person
  ├── Bicycle
  └── Car
```

이 경우 여러 Label이 동시에 참일 수 있다.

따라서 다음과 같이 표현할 수 있다.

$$
\boxed{ \mathbf{y}=(1,1,1,0) }
$$

즉 Multiclass Classification과 달리 Label 간의 동시 선택이 가능하다.

---

### 6.4 Multioutput Classification

Multioutput Classification은 하나의 입력에 대해 여러 Target을 예측하는 문제이다.

예를 들어 하나의 의료영상에서 다음 정보를 동시에 예측한다고 하자.

```text
Input Image
     │
     ▼
    Model
     │
     ├── Disease Type
     │
     ├── Severity Level
     │
     └── Image Quality
```

각 Output이 서로 다른 Class 집합을 가질 수 있다.

따라서 Multilabel Classification과 Multioutput Classification은 구분할 필요가 있다.

---

## 7. Regression(회귀)

Regression은 **입력 데이터에 대해 연속적인 수치값을 예측하는 Task**이다.

Classification이 Class를 예측한다면 Regression은 Numerical Value를 예측한다.

대표적인 예시는 다음과 같다.

- 온도 예측
- 주택 가격 예측
- 혈압 추정
- 센서 출력값 예측
- 연속적인 생체신호 추정

즉 Regression의 Target은 일반적으로 다음과 같다.

$$
\boxed{ y\in\mathbb{R} }
$$

여러 개의 값을 동시에 예측하는 경우에는 Vector 형태의 출력을 가질 수도 있다.

---

### 7.1 Regression의 수학적 표현

모델은 입력 $x$에 대해 연속적인 값을 예측한다.

$$
\boxed{ \hat y=f_\theta(x) }
$$

예를 들어 다음과 같은 데이터가 있다고 하자.

| Input | Target |
| --- | --- |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |

이 데이터는 다음 관계를 따른다.

$$
\boxed{ y=2x+1 }
$$

모델이 이 관계를 학습한다면 새로운 입력 $x=5$에 대해 다음 값을 예측할 수 있다.

$$
\hat y=2(5)+1
$$

따라서

$$
\boxed{ \hat y=11 }
$$

이다.

이처럼 Input과 Output 사이의 수치적 관계를 학습하는 것이 Regression이다.

---

## 8. Linear Regression(선형회귀)

Regression의 대표적인 방법은 Linear Regression이다.

Linear Regression은 입력과 출력 사이의 선형적인 관계를 모델링한다.

가장 단순한 형태는 다음과 같다.

$$
\boxed{ \hat y=wx+b }
$$

여기서

- $w$: Weight
- $b$: Bias
- $x$: Input
- $\hat y$: Prediction

이다.

입력 Feature가 여러 개라면 다음과 같이 표현할 수 있다.

$$
\boxed{ \hat y=\mathbf{w}^{T}\mathbf{x}+b }
$$

즉 여러 Feature의 가중합을 이용하여 연속적인 값을 예측한다.

---

### 8.1 Regression과 Loss Function

Regression에서는 실제 Target과 Prediction 사이의 오차를 계산해야 한다.

대표적인 Loss Function은 MSE(Mean Squared Error)이다.

$$
\boxed{ L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i-\hat y_i)^2 }
$$

모델은 이 Loss를 최소화하는 Parameter를 찾는다.

$$
\boxed{ \hat\theta = \underset{\theta}{\arg\min} L(\theta) }
$$

즉 Training Dataset에서 관측된 Input과 Target 사이의 관계를 잘 설명하는 Parameter를 학습한다.

하지만 학습 데이터에서 오차가 작다고 해서 새로운 데이터에서도 좋은 예측을 수행하는 것은 아니다.

따라서 새로운 데이터에 대한 Generalization Performance도 평가해야 한다.

---

## 9. Classification과 Regression 비교

두 Task의 가장 중요한 차이는 **예측하는 Output의 형태**이다.

| 구분 | Classification | Regression |
| --- | --- | --- |
| 목적 | Class 예측 | Numerical Value 예측 |
| Target | Discrete Label | 주로 Continuous Value |
| Output | Class 또는 Class Probability | 연속적인 수치값 |
| 대표 모델 | Logistic Regression, SVM, Decision Tree | Linear Regression, SVR |
| 대표 Loss | Cross-Entropy | MSE, MAE |
| 대표 Metrics | Accuracy, Precision, Recall, F1 | MAE, MSE, RMSE, R² |
| 예시 | 질병 여부 분류 | 혈압 수치 추정 |

예를 들어 의료 분야에서 같은 입력 데이터를 사용하더라도 문제를 어떻게 정의하느냐에 따라 Task가 달라질 수 있다.

**Classification**

```text
Biosignal
    │
    ▼
Model
    │
    ▼
Normal / Abnormal
```

**Regression**

```text
Biosignal
    │
    ▼
Model
    │
    ▼
Estimated Blood Pressure
```

따라서 어떤 모델을 사용할 것인지 결정하기 전에 **해결하려는 문제의 Output이 무엇인지 정의하는 것**이 중요하다.

---

## 10. Supervised Learning의 대표적인 알고리즘

Supervised Learning에서는 다양한 알고리즘을 사용할 수 있다.

대표적인 방법은 다음과 같다.

| Algorithm | 주요 Task | 특징 |
| --- | --- | --- |
| k-Nearest Neighbors(kNN) | Classification, Regression | 가까운 Sample을 기반으로 예측 |
| Linear Regression | Regression | 선형관계를 모델링 |
| Logistic Regression | Classification | Class Probability 기반 분류 |
| Support Vector Machine(SVM) | Classification, Regression | Decision Boundary 또는 회귀함수 학습 |
| Decision Tree | Classification, Regression | 조건에 따른 분기 구조 |
| Random Forest | Classification, Regression | 여러 Decision Tree의 Ensemble |
| Artificial Neural Network(ANN) | Classification, Regression 등 | 여러 Layer로 비선형 관계 학습 |

---

### 10.1 k-Nearest Neighbors(kNN)

kNN은 새로운 데이터가 입력되었을 때 가까운 Training Sample을 이용하여 예측한다.

Classification에서는 가까운 이웃들의 Class를 기반으로 Label을 결정한다.

Regression에서는 가까운 이웃들의 Target을 평균하는 등의 방식으로 값을 예측한다.

---

### 10.2 Linear Regression

Linear Regression은 입력 Feature와 Target 사이의 선형관계를 학습한다.

$$
\boxed{ \hat y=\mathbf{w}^{T}\mathbf{x}+b }
$$

일반적으로 연속적인 수치 예측에 사용한다.

---

### 10.3 Logistic Regression

Logistic Regression은 이름에 Regression이 포함되어 있지만 **주로 Classification에 사용하는 알고리즘**이다.

Binary Classification에서는 다음과 같이 표현할 수 있다.

$$
\boxed{ P(Y=1\mid x) = \sigma(\mathbf{w}^{T}\mathbf{x}+b) }
$$

여기서 Sigmoid Function은 다음과 같다.

$$
\boxed{ \sigma(z)=\frac{1}{1+e^{-z}} }
$$

즉 입력 Feature의 선형결합을 Class Probability로 변환하여 분류한다.

---

### 10.4 Support Vector Machine(SVM)

SVM은 Class를 구분하는 Decision Boundary를 찾는 대표적인 지도학습 알고리즘이다.

Classification에서는 Class 사이의 Margin을 최대화하는 방향으로 모델을 구성할 수 있다.

또한 Support Vector Regression(SVR)을 사용하면 Regression 문제에도 적용할 수 있다.

---

### 10.5 Decision Tree

Decision Tree는 Feature에 대한 조건을 반복적으로 적용하여 예측하는 모델이다.

예를 들어 다음과 같은 구조를 가질 수 있다.

```text
Input
  │
  ▼
Feature 1 > Threshold?
  │
  ├── Yes
  │     │
  │     ▼
  │  Class A
  │
  └── No
        │
        ▼
     Feature 2 > Threshold?
        │
        ├── Yes → Class B
        │
        └── No  → Class C
```

즉 데이터의 특징에 따라 여러 Decision Rule을 학습한다.

---

### 10.6 Random Forest

Random Forest는 여러 Decision Tree를 결합하는 Ensemble Learning 방법이다.

각 Tree의 예측을 종합하여 최종 Prediction을 생성한다.

Classification에서는 Voting을 사용하고 Regression에서는 각 Tree의 예측값을 평균하는 방식 등을 사용할 수 있다.

---

### 10.7 Artificial Neural Network(ANN)

ANN은 여러 Layer를 이용하여 입력과 출력 사이의 복잡한 관계를 학습한다.

대표적인 구조는 다음과 같다.

```text
Input Layer
     │
     ▼
Hidden Layer 1
     │
     ▼
Activation Function
     │
     ▼
Hidden Layer 2
     │
     ▼
Activation Function
     │
     ▼
Output Layer
```

ANN은 Classification과 Regression에 모두 적용할 수 있다.

또한 모든 ANN이 반드시 지도학습에만 사용되는 것은 아니다.

예를 들어 Autoencoder는 일반적으로 입력 데이터를 재구성하도록 학습하는 구조이며, 정답 Label 없이 표현을 학습하는 데 활용할 수 있다.

즉 **Supervised Learning은 학습 방식이고, ANN은 모델 구조**라는 점을 구분해야 한다.

---

## 11. Deep Learning에서의 Supervised Learning Tasks

Deep Learning에서도 Classification과 Regression은 중요한 Task이다.

하지만 이미지와 같이 구조가 복잡한 데이터를 처리할 때는 다음 Task들도 자주 사용한다.

- Image Classification
- Object Detection
- Segmentation

이들 Task는 입력으로 이미지를 사용한다는 공통점이 있지만 **어떤 정보를 출력해야 하는지**가 다르다.

---

## 12. Image Classification

Image Classification은 입력 이미지가 어떤 Class에 속하는지 예측하는 Task이다.

예를 들어 이미지에 고양이가 존재한다면 다음과 같이 예측할 수 있다.

```text
Input Image
     │
     ▼
Deep Learning Model
     │
     ▼
Class Probabilities
     │
     ├── Cat : 0.85
     ├── Dog : 0.10
     └── Bird: 0.05
     │
     ▼
Prediction: Cat
```

일반적인 Single-label Multiclass Classification에서는 이미지 하나에 대해 하나의 Class를 예측한다.

모델은 클래스별 Score 또는 Probability를 출력할 수 있다.

예를 들어 $K$개의 Class가 존재한다면 출력 Shape는 다음과 같을 수 있다.

$$
\boxed{ \mathbf{p}\in\mathbb{R}^{K} }
$$

Batch Size가 $N$이라면 다음과 같다.

$$
\boxed{ P\in\mathbb{R}^{N\times K} }
$$

즉 각 Sample에 대해 $K$개의 Class Score 또는 Probability를 생성한다.

---

## 13. Object Detection

Object Detection은 **이미지에서 객체가 무엇인지 분류하는 동시에 해당 객체의 위치를 찾는 Task**이다.

즉 Classification과 Localization이 결합된 형태로 이해할 수 있다.

예를 들어 이미지에 사람과 자동차가 존재한다고 하자.

Object Detection Model은 다음 정보를 예측할 수 있다.

```text
Input Image
     │
     ▼
Object Detection Model
     │
     ├── Object 1
     │      ├── Class: Person
     │      └── Bounding Box
     │
     └── Object 2
            ├── Class: Car
            └── Bounding Box
```

즉 모델은 각 객체의 Class와 위치를 함께 예측한다.

---

### 13.1 Bounding Box

Bounding Box는 이미지 안에서 객체가 존재하는 영역을 사각형으로 표시한 것이다.

대표적으로 다음과 같이 표현할 수 있다.

$$
\boxed{ B=(x_{\min},y_{\min},x_{\max},y_{\max}) }
$$

여기서

- $x_{\min}$: 왼쪽 경계
- $y_{\min}$: 위쪽 경계
- $x_{\max}$: 오른쪽 경계
- $y_{\max}$: 아래쪽 경계

이다.

예를 들어 Object Detection의 출력은 다음 정보를 포함할 수 있다.

```text
Class       : Person
Confidence  : 0.92
Bounding Box: [40, 30, 180, 250]
```

즉 모델은 단순히 객체의 종류를 예측하는 것이 아니라 이미지에서 객체가 어디에 위치하는지도 예측한다.

---

## 14. Segmentation

Segmentation은 **이미지를 Pixel 단위로 분류하여 객체 또는 영역을 구분하는 Task**이다.

Object Detection이 Bounding Box로 객체의 대략적인 위치를 나타낸다면, Segmentation은 각 Pixel이 어떤 영역에 속하는지 구분한다.

예를 들어 의료영상에서 종양 영역을 구분한다고 하자.

```text
Medical Image
     │
     ▼
Segmentation Model
     │
     ▼
Pixel-level Prediction
     │
     ├── Tumor
     └── Background
```

결과적으로 다음과 같은 Segmentation Mask를 생성할 수 있다.

```text
Original Image
      │
      ▼
Segmentation Model
      │
      ▼
Binary Mask

0 0 0 0 0
0 0 1 1 0
0 1 1 1 0
0 0 1 0 0
0 0 0 0 0
```

여기서

- `0`: Background
- `1`: Target Region

이다.

즉 해당 Pixel이 관심 영역에 포함되는지를 나타낸다.

---

### 14.1 Segmentation의 출력 형태

입력 이미지가 다음 크기를 가진다고 하자.

$$
\boxed{ H\times W }
$$

Binary Segmentation에서는 최종적으로 다음 Shape의 Mask를 생성할 수 있다.

$$
\boxed{ M\in\{0,1\}^{H\times W} }
$$

Multiclass Segmentation에서는 Pixel마다 하나의 Class ID를 할당할 수 있다.

$$
\boxed{ M\in\{0,1,\cdots,K-1\}^{H\times W} }
$$

딥러닝 모델은 최종 Class Mask를 생성하기 전에 각 Class에 대한 Score를 출력할 수도 있다.

대표적으로 다음과 같은 Tensor Shape를 사용한다.

```text
[N, C, H, W]
```

여기서

- $N$: Batch Size
- $C$: Class 또는 Output Channel 수
- $H$: Height
- $W$: Width

이다.

---

## 15. Semantic Segmentation과 Instance Segmentation

Segmentation은 대표적으로 다음 두 가지로 구분할 수 있다.

1. Semantic Segmentation
2. Instance Segmentation

---

### 15.1 Semantic Segmentation

Semantic Segmentation은 **각 Pixel이 어떤 Class에 속하는지**를 구분하는 Task이다.

예를 들어 이미지에 자동차 두 대가 존재한다고 하자.

Semantic Segmentation에서는 두 자동차를 모두 같은 Class로 분류한다.

```text
Car 1 → Car Class

Car 2 → Car Class
```

즉 두 자동차가 서로 다른 객체라는 사실보다 두 영역이 모두 자동차에 해당한다는 점에 초점을 둔다.

---

### 15.2 Instance Segmentation

Instance Segmentation은 같은 Class에 속하더라도 서로 다른 객체라면 구분하는 Task이다.

예를 들어 이미지에 자동차 두 대가 있다면 다음과 같이 구분한다.

```text
Car 1 → Instance 1

Car 2 → Instance 2
```

두 객체는 모두 Car Class에 속하지만 별개의 Instance로 구분된다.

따라서 다음과 같이 정리할 수 있다.

| 구분 | Semantic Segmentation | Instance Segmentation |
| --- | --- | --- |
| 분류 기준 | Pixel의 Class | Pixel의 Class + 객체 Instance |
| 동일 Class의 객체 구분 | 일반적으로 구분하지 않음 | 구분함 |
| 주요 목적 | 영역의 의미 파악 | 개별 객체 영역 구분 |

즉 Semantic Segmentation은 **무엇에 해당하는 Pixel인가?**에 초점을 둔다.

반면 Instance Segmentation은 **어떤 객체에 해당하는 Pixel인가?**까지 구분한다.

---

## 16. Classification·Detection·Segmentation 비교

세 Task를 비교하면 다음과 같다.

| Task | 예측 대상 | Output |
| --- | --- | --- |
| Image Classification | 이미지의 Class | Class Label 또는 Score |
| Object Detection | 객체의 Class와 위치 | Class + Bounding Box |
| Semantic Segmentation | 각 Pixel의 Class | Pixel-level Class Mask |
| Instance Segmentation | 개별 객체의 Pixel 영역 | Instance별 Mask 및 Class |

이를 간단히 정리하면 다음과 같다.

```text
                 Input Image
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
   Classification  Detection  Segmentation
          │           │           │
          ▼           ▼           ▼
       Class       Class +     Pixel-level
       Label      Location      Mask
```

예를 들어 의료영상 분석에서는 다음과 같이 Task를 정의할 수 있다.

**Classification**

> 이 영상에 이상 소견이 존재하는가?

**Object Detection**

> 이상 소견이 어느 위치에 존재하는가?

**Segmentation**

> 이상 영역에 해당하는 Pixel은 정확히 어디인가?

같은 이미지를 입력으로 사용하더라도 해결하려는 문제에 따라 서로 다른 Task를 정의할 수 있다.

---

## 17. Supervised Learning의 Training 과정

> 이 절에서는 지도학습의 정의를 실제 머신러닝 구현 과정과 연결해 살펴본다.

Supervised Learning의 Training Process는 일반적으로 다음과 같은 흐름을 따른다.

```text
Labeled Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Model Selection
      │
      ▼
Model Training
      │
      ▼
Validation
      │
      ▼
Model Selection / Hyperparameter Tuning
      │
      ▼
Final Evaluation
      │
      ▼
Inference
```

---

### 17.1 Training Set

Training Set은 모델의 Parameter를 학습하는 데 사용하는 데이터이다.

$$
\boxed{ D_{\mathrm{train}} }
$$

모델은 Training Set의 Input과 Label 사이의 관계를 학습한다.

딥러닝에서는 일반적으로 Loss Function과 Optimizer를 이용하여 Parameter를 갱신한다.

---

### 17.2 Validation Set

Validation Set은 학습 중 모델의 성능을 확인하거나 Hyperparameter를 선택하는 데 사용하는 데이터이다.

$$
\boxed{ D_{\mathrm{val}} }
$$

예를 들어 다음 값을 조절할 때 사용할 수 있다.

- Learning Rate
- Model Complexity
- Regularization Strength
- Number of Epochs

Validation Set은 일반적으로 직접적인 Gradient Update에는 사용하지 않는다.

---

### 17.3 Test Set

Test Set은 최종적으로 선택한 모델의 Generalization Performance를 평가하는 데 사용하는 데이터이다.

$$
\boxed{ D_{\mathrm{test}} }
$$

중요한 점은 Test Set을 모델의 Parameter나 Hyperparameter를 조정하는 데 반복적으로 사용하지 않아야 한다는 것이다.

Test Set에 맞추어 모델을 계속 수정하면 최종 성능평가 결과가 낙관적으로 편향될 수 있다.

---

## 18. Loss Function과 Evaluation Metric

Supervised Learning에서는 Model Prediction과 Target을 비교해야 한다.

이때 사용하는 개념이 Loss Function과 Evaluation Metric이다.

두 개념은 관련되어 있지만 역할이 다르다.

#### Loss Function

모델 학습에서 최적화할 목적함수이다.

예를 들어 Regression에서는 MSE를 사용할 수 있다.

$$
\boxed{ L_{MSE} = \frac1N \sum_{i=1}^{N} (y_i-\hat y_i)^2 }
$$

Classification에서는 Cross-Entropy를 사용할 수 있다.

$$
\boxed{ L_{CE} = -\frac1N \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik}\log p_{ik} }
$$

#### Evaluation Metric

학습된 모델이 실제 Task를 얼마나 잘 수행하는지 평가하는 지표이다.

대표적인 지표는 다음과 같다.

| Task | Evaluation Metrics |
| --- | --- |
| Classification | Accuracy, Precision, Recall, F1-score |
| Regression | MAE, MSE, RMSE, R² |
| Object Detection | IoU, mAP |
| Segmentation | IoU, Dice Score |

즉 Loss Function은 주로 **모델을 어떻게 학습시킬 것인가**와 관련되고, Evaluation Metric은 **모델이 얼마나 잘 수행하는가**를 확인하는 데 사용된다.

---

## 19. Generalization(일반화)

Supervised Learning에서 중요한 목표는 단순히 Training Dataset을 잘 예측하는 것이 아니다.

**학습에 사용하지 않은 새로운 데이터에서도 좋은 성능을 보이는 것**이 중요하다.

이를 Generalization이라고 한다.

예를 들어 모델이 Training Set에서는 다음과 같은 성능을 보였다고 하자.

```text
Training Accuracy   : 99%

Validation Accuracy : 70%
```

이 경우 모델이 Training Data에 과도하게 적합되었을 가능성을 고려해야 한다.

이러한 현상을 Overfitting이라고 한다.

반대로 Training과 Validation 모두에서 성능이 낮다면 Underfitting을 의심할 수 있다.

```text
                  Model Training
                        │
                        ▼
                Training Performance
                        │
                        ▼
               Validation Performance
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Good Generalization    Generalization Gap
                                   │
                                   ▼
                          Overfitting 가능성
```

따라서 지도학습에서 중요한 것은 **Training Data에 대한 오차를 최소화하면서도 새로운 데이터에 대한 예측 성능을 유지하는 것**이다.

---

## 20. Python 실습: Classification

> 추가 실습: scikit-learn을 이용하여 대표적인 지도학습 Classification 문제를 구현한다.

Iris Dataset을 이용하여 Supervised Learning 과정을 확인해 보자.

Iris Dataset은 네 가지 Feature로 세 개의 붓꽃 Class를 분류하는 문제이다.

---

### 20.1 Dataset Load

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print("X Shape:", X.shape)
print("y Shape:", y.shape)
```

출력:

```text
X Shape: (150, 4)
y Shape: (150,)
```

즉 150개의 Sample에 4개의 Feature가 존재한다.

각 Sample에는 하나의 Class Label이 할당되어 있다.

---

### 20.2 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

여기서

```python
test_size=0.2
```

는 전체 데이터의 20%를 Test Set으로 사용한다는 의미이다.

또한

```python
stratify=y
```

는 Class 비율이 Train Set과 Test Set에서 유사하게 유지되도록 분할한다.

---

### 20.3 Model Training

Logistic Regression Model을 사용해 보자.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(
    X_train,
    y_train
)
```

여기서 `fit()`은 Training Data를 이용하여 모델을 학습하는 메서드이다.

`StandardScaler`는 Training Data의 통계량을 이용해 Feature를 표준화한다.

Pipeline으로 구성했기 때문에 Test Set에 대해 별도로 평균과 표준편차를 학습하지 않는다.

---

### 20.4 Prediction

학습된 모델을 이용하여 Test Data의 Class를 예측한다.

```python
y_pred = model.predict(
    X_test
)

print(y_pred[:10])
```

이때 모델은 각 Sample에 대해 Class Index를 출력한다.

---

### 20.5 Evaluation

Accuracy를 계산해 보자.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)
```

Accuracy는 다음과 같이 정의된다.

$$
\boxed{ \operatorname{Accuracy} = \frac{\text{Correct Predictions}} {\text{Total Predictions}} }
$$

즉 전체 Test Sample 중 올바르게 분류한 비율이다.

이 과정을 통해 지도학습의 기본적인 흐름을 확인할 수 있다.

```text
Labeled Dataset
      │
      ▼
Train/Test Split
      │
      ▼
Model.fit()
      │
      ▼
Model.predict()
      │
      ▼
Accuracy Evaluation
```

---

## 21. Python 실습: Regression

이번에는 Regression Model을 구현해 보자.

간단한 가상의 데이터를 생성한다.

$$
\boxed{ y=2x+1+\epsilon }
$$

여기서 $\epsilon$은 Gaussian Noise라고 가정한다.

---

### 21.1 Dataset 생성

```python
import numpy as np

rng = np.random.default_rng(42)

X = rng.uniform(
    low=0,
    high=10,
    size=(100, 1)
)

noise = rng.normal(
    loc=0,
    scale=1,
    size=(100, 1)
)

y = 2 * X + 1 + noise
```

여기서 실제 데이터는 다음 관계를 중심으로 생성된다.

$$
\boxed{ y=2x+1 }
$$

단, Noise가 추가되므로 모든 Sample이 정확히 직선 위에 위치하지는 않는다.

---

### 21.2 Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
```

`LinearRegression()`은 입력과 Target 사이의 선형관계를 학습한다.

---

### 21.3 Parameter 확인

```python
print(
    "Weight:",
    model.coef_
)

print(
    "Bias:",
    model.intercept_
)
```

충분한 데이터가 주어진다면 학습된 Weight와 Bias가 각각 2와 1에 가까워지는 경향을 확인할 수 있다.

다만 Noise가 존재하므로 정확히 일치하지는 않을 수 있다.

---

### 21.4 새로운 값 예측

```python
X_new = np.array([
    [5.0]
])

prediction = model.predict(
    X_new
)

print(prediction)
```

이론적인 관계에서는

$$
y=2(5)+1=11
$$

이다.

학습된 모델은 이 값에 가까운 연속적인 수치를 출력할 수 있다.

즉 Classification과 달리 Regression에서는 Class가 아니라 Numerical Value를 예측한다.

---

## 22. Supervised Learning과 Unsupervised Learning의 차이

Supervised Learning에서는 Input과 Label이 함께 주어진다.

반면 Unsupervised Learning에서는 일반적으로 정답 Label 없이 데이터의 구조나 패턴을 학습한다.

| 구분 | Supervised Learning | Unsupervised Learning |
| --- | --- | --- |
| Label | 사용 | 일반적으로 사용하지 않음 |
| 주요 목적 | 입력에 대한 정답 예측 | 데이터의 구조 및 패턴 발견 |
| 대표적인 Task | Classification, Regression | Clustering, Dimensionality Reduction |
| 대표적인 알고리즘 | Logistic Regression, SVM, Random Forest | k-means, PCA |

하지만 특정 Task가 항상 하나의 학습 방식으로만 해결되는 것은 아니다.

예를 들어 Dimensionality Reduction은 비지도학습으로 수행하는 경우가 많지만, Label을 활용하여 차원을 축소하는 지도학습 방법도 존재한다.

즉 **학습 방식은 정답 데이터가 어떻게 사용되는지에 따라 구분하고, Task는 무엇을 해결하려는지에 따라 구분**하는 것이 중요하다.

---

## 23. Supervised Learning의 장단점

### 23.1 장점

**명확한 학습 목표**

Input과 Target이 존재하므로 모델이 어떤 출력을 생성해야 하는지 정의할 수 있다.

**정량적인 성능평가**

Prediction과 Ground Truth를 비교하여 Accuracy, MAE 등의 지표를 계산할 수 있다.

**다양한 문제에 적용 가능**

Classification과 Regression뿐 아니라 Object Detection, Segmentation 등 여러 Task에 적용할 수 있다.

---

### 23.2 한계

**Labeled Data가 필요함**

지도학습에서는 정답이 포함된 Training Dataset이 필요하다.

**Labeling Cost**

특히 의료영상 Segmentation처럼 전문적인 지식이 필요한 Task에서는 Label을 생성하는 데 많은 시간과 비용이 필요할 수 있다.

**Label Quality에 대한 의존성**

정답 Label에 오류가 많으면 모델이 잘못된 관계를 학습할 수 있다.

**Generalization 문제**

Training Dataset과 실제 사용 환경의 데이터 분포가 다르면 모델의 예측 성능이 저하될 수 있다.

따라서 지도학습의 성능은 모델 구조뿐 아니라 Dataset의 양과 품질, Label의 정확성, 실제 환경과의 유사성에도 영향을 받는다.

---

## 24. 핵심 공식 총정리

### 24.1 Labeled Dataset

$$
\boxed{ D=\{(x_i,y_i)\}_{i=1}^{N} }
$$

### 24.2 Function Approximation

$$
\boxed{ \hat y=f_\theta(x) }
$$

### 24.3 Supervised Learning Objective

$$
\boxed{ \hat\theta = \underset{\theta}{\arg\min} \frac1N \sum_{i=1}^{N} \ell(f_\theta(x_i),y_i) }
$$

여기서 $\ell$은 Loss Function이다.

### 24.4 Classification

$$
\boxed{ \hat y = \underset{k}{\arg\max} P_\theta(Y=k\mid X=x) }
$$

### 24.5 Linear Regression

$$
\boxed{ \hat y=\mathbf{w}^{T}\mathbf{x}+b }
$$

### 24.6 Mean Squared Error

$$
\boxed{ L_{MSE} = \frac1N \sum_{i=1}^{N} (y_i-\hat y_i)^2 }
$$

### 24.7 Categorical Cross-Entropy

$$
\boxed{ L_{CE} = -\frac1N \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik}\log p_{ik} }
$$

---

## 25. 최종 정리

Supervised Learning은 Input Data와 이에 대응하는 Label 또는 Target을 이용하여 모델을 학습하는 머신러닝 방법이다.

지도학습의 핵심은 입력과 출력 사이의 관계를 학습하여 새로운 데이터에 대해 적절한 Prediction을 생성하는 것이다.

이를 수학적으로 표현하면 다음과 같다.

$$
\boxed{ \hat y=f_\theta(x) }
$$

대표적인 Task는 Classification과 Regression이다.

Classification은 이산적인 Class를 예측한다.

Regression은 연속적인 수치값을 예측한다.

따라서 두 Task는 Output의 형태에 따라 구분된다.

또한 Supervised Learning에서는 kNN, Linear Regression, Logistic Regression, SVM, Decision Tree, Random Forest, ANN 등 다양한 알고리즘을 사용할 수 있다.

Deep Learning에서는 이러한 Task가 이미지 분석으로 확장되어 Classification, Object Detection, Segmentation 등으로 구현된다.

Classification은 이미지의 Class를 예측한다.

Object Detection은 객체의 Class와 위치를 예측한다.

Segmentation은 Pixel 단위로 영역을 분류한다.

따라서 같은 Input Data를 사용하더라도 해결하려는 문제에 따라 서로 다른 Task가 정의될 수 있다.

마지막으로 지도학습에서 중요한 것은 Training Dataset에 대한 성능만 높이는 것이 아니다.

**학습에 사용되지 않은 새로운 데이터에서도 좋은 예측 성능을 유지하는 Generalization이 중요하다.**

---

### 핵심 요약

> Supervised Learning은 Input과 Label로 구성된 Labeled Dataset을 이용하여 입력과 출력 사이의 Mapping Function을 학습하는 방법이다. 대표적인 Task는 Classification과 Regression이며, Deep Learning에서는 Object Detection과 Segmentation 등으로 확장된다. 지도학습의 최종 목표는 학습 데이터뿐 아니라 새로운 데이터에서도 적절한 예측을 수행하는 모델을 만드는 것이다.

---

## 참고 자료

#### 주요 학습 자료

- [Supervised Learning — Dsaint31](https://dsaint31.me/mkdocs_site/ML/ch00/ch00_20_supervised/)
- [Categories of Machine Learning — Dsaint31](https://dsaint31.me/mkdocs_site/ML/ch00/ch00_12_type_of_ml/)

#### 추가 참고

- [Scikit-learn — Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Scikit-learn — Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn — Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
