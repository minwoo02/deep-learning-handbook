# scikit-learn으로 Perceptron 구현하기

[TOC]


## 실습 구성표

| 항목 | 실습 구성 |
| --- | --- |
| Task | Iris multiclass classification |
| Model | scikit-learn Perceptron |
| nn.Module | 미사용 |
| Autograd | PyTorch 미사용 |
| Loss | 코드에 별도 loss 지정 없음 |
| Optimizer | Perceptron의 fit · eta0=1.0 |
| Activation | 코드에 별도 activation 지정 없음 |



관련 이론: [Classification과 Regression](../../theory/foundations/classification-vs-regression.md) · [TLU와 Perceptron의 기하학적 이해](../../theory/architectures/perceptron-geometry.md) · [머신러닝 분류 성능평가 완전 정리: Confusion Matrix부터 Accuracy·Precision·Recall·F1-score까지](../../theory/evaluation/classification-metrics.md)


---

## scikit-learn으로 Perceptron 구현하기

앞에서 Perceptron의 수학적 구조를 살펴보았다.

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

그리고 Decision Boundary는

$$
\mathbf{w}^T\mathbf{x}+b=0
$$

으로 표현할 수 있다.

이번에는 이를 Python의 `scikit-learn`을 사용하여 실제로 구현해본다.

---

## 1. 전체 코드

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. dataset load
X, y = load_iris(return_X_y=True)

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. model 생성
model = make_pipeline(
    StandardScaler(),
    Perceptron(
        max_iter=1000,
        eta0=1.0,
        random_state=42,
    ),
)

# 4. training
model.fit(X_train, y_train)

# 5. prediction
y_pred = model.predict(X_test)

# 6. evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

전체 흐름은 다음과 같다.

```text
Dataset
   ↓
Train / Test Split
   ↓
StandardScaler
   ↓
Perceptron
   ↓
Training
   ↓
Prediction
   ↓
Evaluation
```

---

## 2. Import

```python
from sklearn.datasets import load_iris
```

`sklearn.datasets`에서 Iris Dataset을 불러오기 위한 함수이다.

- Dataset: 데이터셋
- Iris: 붓꽃
- load: 불러오기

---

```python
from sklearn.linear_model import Perceptron
```

scikit-learn의 Linear Model 모듈에서 Perceptron Class를 가져온다.

Perceptron은 Linear Decision Boundary를 사용하는 분류 모델이다.

---

```python
from sklearn.metrics import accuracy_score
```

모델의 예측 정확도(Accuracy)를 계산하기 위해 사용한다.

---

```python
from sklearn.model_selection import train_test_split
```

전체 데이터를 Training Data와 Test Data로 나누는 함수이다.

---

```python
from sklearn.pipeline import make_pipeline
```

여러 데이터 처리 단계를 하나의 Pipeline으로 연결한다.

이번 코드에서는

```text
StandardScaler
      ↓
Perceptron
```

을 하나의 모델처럼 묶는다.

---

```python
from sklearn.preprocessing import StandardScaler
```

Feature들을 표준화(Standardization)하기 위한 클래스이다.

---

## 3. Dataset Load

```python
X, y = load_iris(return_X_y=True)
```

Iris Dataset을 불러온다.

여기서

```text
X → Features
y → Target / Label
```

이다.

Iris Dataset에는 네 개의 Feature가 있다.

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

`y`에는 각 데이터가 어떤 Iris 품종에 해당하는지가 저장된다.

---

## 4. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

전체 Dataset을 Training Set과 Test Set으로 나눈다.

---

### test_size

```python
test_size=0.2
```

전체 데이터의 20%를 Test Set으로 사용한다.

따라서 나머지 80%가 Training Set이 된다.

---

### random_state

```python
random_state=42
```

데이터를 나눌 때 사용되는 Random State를 고정한다.

같은 코드를 반복 실행했을 때 동일한 데이터 분할을 재현하기 위해 사용한다.

---

### stratify

```python
stratify=y
```

Training Set과 Test Set의 Class 비율이 원본 Dataset의 Class 비율과 비슷하게 유지되도록 한다.

이를 **Stratified Sampling**이라고 한다.

---

## 5. StandardScaler

```python
StandardScaler()
```

각 Feature의 Scale을 조정한다.

일반적으로 평균이 0이고 표준편차가 1에 가까운 형태로 변환한다.

$$
x' = \frac{x-\mu}{\sigma}
$$

여기서

- $\mu$ : 평균
- $\sigma$ : 표준편차

이다.

Feature들의 수치 범위가 크게 다를 경우 Scale을 맞춰주면 학습 과정이 보다 안정적으로 진행될 수 있다.

---

## 6. Perceptron

```python
Perceptron(
    max_iter=1000,
    eta0=1.0,
    random_state=42,
)
```

Perceptron 객체를 생성한다.

---

### max_iter

```python
max_iter=1000
```

Training Data를 반복하여 학습할 수 있는 최대 반복 횟수를 지정한다.

---

### eta0

```python
eta0=1.0
```

초기 Learning Rate와 관련된 값이다.

학습 과정에서 Weight를 어느 정도 크기로 수정할지를 결정하는 데 사용된다.

---

### random_state

```python
random_state=42
```

알고리즘 내부의 Randomness를 재현 가능하도록 고정한다.

---

## 7. Pipeline

```python
model = make_pipeline(
    StandardScaler(),
    Perceptron(...),
)
```

이 코드는 다음의 연산을 하나의 Pipeline으로 연결한다.

```text
Raw Input
   ↓
StandardScaler
   ↓
Scaled Input
   ↓
Perceptron
   ↓
Prediction
```

따라서 이후에는 `model` 하나만 사용하면 된다.

---

## 8. Training

```python
model.fit(X_train, y_train)
```

`fit()`은 Training Data를 사용하여 모델을 학습한다.

Perceptron은 학습을 통해 적절한 Weight와 Bias를 찾는다.

즉 수식에서

$$
\mathbf{w},b
$$

에 해당하는 Parameter가 Training 과정에서 조정된다.

---

## 9. Prediction

```python
y_pred = model.predict(X_test)
```

학습이 끝난 모델에 Test Feature를 입력한다.

결과는 모델이 예측한 Class이다.

```text
X_test
   ↓
model
   ↓
y_pred
```

---

## 10. Evaluation

```python
accuracy = accuracy_score(y_test, y_pred)
```

실제 정답

```python
y_test
```

과 예측값

```python
y_pred
```

을 비교한다.

Accuracy는 다음과 같이 생각할 수 있다.

$$
Accuracy = \frac{\text{Correct Predictions}} {\text{Total Predictions}}
$$

---

## 11. 이론과 코드 연결

앞에서 배운 개념을 코드와 연결하면 다음과 같다.

| 이론 | 코드 |
| --- | --- |
| Dataset | `load_iris()` |
| Feature $X$ | `X` |
| Label $y$ | `y` |
| Training | `fit()` |
| Prediction | `predict()` |
| Perceptron | `Perceptron()` |
| Learning Rate | `eta0` |
| 반복 횟수 | `max_iter` |
| Evaluation | `accuracy_score()` |

---

### 핵심 정리

scikit-learn에서는 Perceptron의 복잡한 학습 과정을 직접 구현할 필요가 없다.

```python
model.fit(X_train, y_train)
```

한 줄 안에서 모델이 Training Data를 이용하여 내부 Parameter를 학습한다.

따라서 전체 흐름을 다음처럼 기억할 수 있다.

```text
Load Data
↓
Split Data
↓
Preprocessing
↓
Create Model
↓
fit()
↓
predict()
↓
Evaluation
```

이 구조는 이후 `MLPClassifier`와 같은 다른 머신러닝 모델을 사용할 때도 거의 동일하게 반복된다.

## 12. 주석 설명 포함 전체 코드

```python
# sklearn = scikit-learn
# Python에서 Machine Learning(머신러닝)을 구현할 때 많이 사용하는 라이브러리

# datasets = 데이터셋 관련 모듈
# load_iris = Iris Dataset(붓꽃 데이터셋)을 불러오는 함수
from sklearn.datasets import load_iris

# linear_model = Linear Model(선형 모델) 관련 모듈
# Perceptron = 퍼셉트론, 입력 특성들의 가중합을 이용해 분류하는 기본적인 선형 분류 모델
from sklearn.linear_model import Perceptron

# metrics = 평가 지표 관련 모듈
# accuracy_score = Accuracy(정확도)를 계산하는 함수
# 정확도 = 전체 예측 중 정답으로 맞힌 비율
from sklearn.metrics import accuracy_score

# model_selection = 데이터 분할 및 모델 선택 관련 모듈
# train_test_split = 데이터를 Training Set(훈련 세트)과 Test Set(테스트 세트)으로 분리하는 함수
from sklearn.model_selection import train_test_split

# pipeline = 여러 전처리 과정과 모델을 하나의 처리 흐름으로 연결하는 모듈
# make_pipeline = Pipeline(파이프라인)을 간단하게 생성하는 함수
from sklearn.pipeline import make_pipeline

# preprocessing = Data Preprocessing(데이터 전처리) 관련 모듈
# StandardScaler = Standardization(표준화)을 수행하는 클래스
# 각 Feature(특성)를 평균 0, 표준편차 1에 가깝게 변환
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Dataset Load (데이터셋 불러오기)
# ============================================================

# X = Feature(특성, 입력 데이터)
# y = Target / Label(정답, 목표값)

# load_iris() = Iris Dataset(붓꽃 데이터셋)을 불러옴
# return_X_y=True
# → 데이터 전체 객체가 아니라 X와 y만 바로 반환

# Iris Dataset의 X에는 4개의 Feature가 존재
# 1. sepal length = 꽃받침 길이
# 2. sepal width  = 꽃받침 너비
# 3. petal length = 꽃잎 길이
# 4. petal width  = 꽃잎 너비

# y에는 붓꽃의 Class(클래스, 품종)가 숫자로 저장됨
# 0 = Setosa
# 1 = Versicolor
# 2 = Virginica

X, y = load_iris(return_X_y=True)


# ============================================================
# 2. Train/Test Split (훈련/테스트 데이터 분할)
# ============================================================

# Training Set(훈련 세트)
# → 모델이 학습할 때 사용하는 데이터

# Test Set(테스트 세트)
# → 학습이 끝난 모델의 성능을 평가할 때 사용하는 데이터

X_train, X_test, y_train, y_test = train_test_split(
    X,                  # Feature(입력 데이터)
    y,                  # Target / Label(정답 데이터)

    test_size=0.2,      # 전체 데이터의 20%를 Test Set으로 사용
                        # 나머지 80%는 Training Set으로 사용

    random_state=42,    # Random Seed(난수 시드)
                        # 실행할 때마다 동일하게 데이터가 분할되도록 고정
                        # 42라는 숫자 자체에는 특별한 의미 없음

    stratify=y,         # Stratified Split(계층적 분할)
                        # 각 Class의 비율이 Train/Test에 비슷하게 유지되도록 분할
)


# ============================================================
# 3. Model Creation (모델 생성)
# ============================================================

# Pipeline(파이프라인)
# → 데이터 전처리와 머신러닝 모델을 하나의 처리 과정으로 연결

model = make_pipeline(

    # StandardScaler = Standardization(표준화)
    # Feature마다 값의 크기와 단위가 다르므로 스케일을 맞춰줌
    #
    # 기본 형태:
    # z = (x - mean) / standard deviation
    #
    # mean = 평균
    # standard deviation = 표준편차
    #
    # 결과적으로 각 Feature는 대체로
    # 평균(mean) ≈ 0
    # 표준편차(standard deviation) ≈ 1
    StandardScaler(),

    # Perceptron = 퍼셉트론
    # Linear Classifier(선형 분류기)
    #
    # Feature들의 Weighted Sum(가중합)을 계산하여 Class를 분류
    #
    # 기본 개념:
    # z = w1*x1 + w2*x2 + ... + wn*xn + b
    #
    # w = Weight(가중치)
    # x = Feature(입력 특성)
    # b = Bias(편향)
    Perceptron(

        # max_iter = Maximum Iterations(최대 반복 횟수)
        # 학습 데이터를 최대 1000번 반복하면서 Weight를 업데이트
        max_iter=1000,

        # eta0 = Initial Learning Rate(초기 학습률)
        # Weight를 한 번 업데이트할 때 얼마나 크게 수정할지 결정
        eta0=1.0,

        # random_state = Random Seed(난수 시드)
        # 모델 내부의 무작위 동작을 동일하게 재현하기 위해 고정
        random_state=42,
    ),
)


# ============================================================
# 4. Training (학습)
# ============================================================

# fit = 학습하다 / 모델을 데이터에 맞추다
#
# X_train = Training Feature(훈련용 입력 데이터)
# y_train = Training Label(훈련용 정답 데이터)
#
# StandardScaler가 먼저 X_train을 표준화한 후
# Perceptron이 표준화된 데이터를 이용해 Weight와 Bias를 학습
model.fit(X_train, y_train)


# ============================================================
# 5. Prediction (예측)
# ============================================================

# predict = 예측하다
#
# X_test = 모델이 학습할 때 사용하지 않은 Test Feature
#
# Pipeline에 의해
# 1. X_test를 StandardScaler로 변환
# 2. 학습된 Perceptron으로 Class 예측
#
# y_pred = Predicted Label(모델이 예측한 정답)
y_pred = model.predict(X_test)


# ============================================================
# 6. Evaluation (평가)
# ============================================================

# accuracy_score = Accuracy(정확도) 계산
#
# y_test = Actual / True Label(실제 정답)
# y_pred = Predicted Label(예측값)
#
# Accuracy = 맞힌 개수 / 전체 데이터 개수
accuracy = accuracy_score(y_test, y_pred)


# 모델의 Accuracy(정확도) 출력
print("Accuracy:", accuracy)
```
