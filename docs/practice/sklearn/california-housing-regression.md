# scikit-learn MLPRegressor로 California Housing 집값 예측하기

[TOC]


## 실습 구성표

| 항목 | 실습 구성 |
| --- | --- |
| Task | California Housing regression |
| Model | MLPRegressor · hidden=(128,64) |
| nn.Module | 미사용 |
| Autograd | PyTorch 미사용 |
| Loss | 코드에 별도 loss 지정 없음 |
| Optimizer | solver="adam" |
| Activation | activation="relu" |



관련 이론: [Classification과 Regression](../../theory/foundations/classification-vs-regression.md) · [머신러닝 회귀 성능평가 완전 정리: MAE·MSE·RMSE·MAPE부터 결정계수 R²까지](../../theory/evaluation/regression-metrics.md)


---

## scikit-learn MLPRegressor로 California Housing 집값 예측하기

앞에서는 `MLPClassifier`를 사용하여 데이터가 어떤 Class에 속하는지 예측하는 **Classification** 문제를 살펴보았다.

이번에는 같은 Multi-Layer Perceptron 구조를 이용하여 연속적인 값을 예측하는 **Regression** 문제를 구현한다.

사용할 모델은 scikit-learn의

```python
MLPRegressor
```

이며, 데이터는 California Housing Dataset을 사용한다.

전체적인 흐름은 다음과 같다.

```text
California Housing Dataset
          ↓
Train / Test Split
          ↓
StandardScaler
          ↓
MLPRegressor
          ↓
Training
          ↓
Prediction
          ↓
MSE / RMSE / MAE / R²
```

---

## 1. 전체 코드

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# 1. Dataset Load
dataset = fetch_california_housing(as_frame=True)

X = dataset.data
y = dataset.target


# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 3. Pipeline + MLPRegressor
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            ),
        ),
    ]
)


# 4. Training
model.fit(X_train, y_train)


# 5. Prediction
y_pred = model.predict(X_test)


# 6. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")
```

---

## 2. 이번 문제는 Regression이다

먼저 Classification과 Regression의 차이를 다시 보면 다음과 같다.

#### Classification

입력이 어떤 Class에 속하는지를 예측한다.

```text
Input
  ↓
Dog / Cat

Input
  ↓
Normal / Abnormal
```

출력은 이산적인 Class이다.

---

#### Regression

입력으로부터 연속적인 숫자를 예측한다.

```text
Housing Data
      ↓
House Value
```

예를 들어

```text
2.31
3.82
1.57
...
```

과 같이 연속적인 실수값을 출력한다.

따라서 이번 문제에서는 `MLPClassifier`가 아니라

```python
MLPRegressor
```

를 사용한다.

---

## 3. 필요한 Library Import

```python
from sklearn.datasets import fetch_california_housing
```

California Housing Dataset을 불러오기 위한 함수이다.

---

```python
from sklearn.model_selection import train_test_split
```

전체 Dataset을 Training Set과 Test Set으로 나누기 위해 사용한다.

---

```python
from sklearn.pipeline import Pipeline
```

여러 처리 과정을 하나의 Pipeline으로 연결한다.

이번 코드에서는

```text
StandardScaler
      ↓
MLPRegressor
```

를 하나의 Pipeline으로 구성한다.

---

```python
from sklearn.preprocessing import StandardScaler
```

각 Feature의 Scale을 표준화하기 위해 사용한다.

---

```python
from sklearn.neural_network import MLPRegressor
```

Multi-Layer Perceptron을 이용하여 Regression을 수행하는 모델이다.

---

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
```

Regression 모델의 성능을 평가하기 위한 Metric들이다.

이번 코드에서는

- MSE
- RMSE
- MAE
- R²

를 사용한다.

---

## 4. California Housing Dataset 불러오기

```python
dataset = fetch_california_housing(
    as_frame=True
)
```

`fetch_california_housing()`을 이용하여 Dataset을 불러온다.

```python
as_frame=True
```

를 사용하면 데이터가 pandas의 DataFrame / Series 형태로 반환된다.

따라서 데이터를 확인하기도 편하다.

```python
print(dataset.data.head())
print(dataset.target.head())
```

---

## 5. Feature와 Target 분리

```python
X = dataset.data
y = dataset.target
```

머신러닝에서는 일반적으로

```text
X
→ Feature / Input

y
→ Target / Label
```

로 표현한다.

이번 문제에서는

```text
주택과 지역에 대한 여러 Feature
             ↓
             X
             ↓
            MLP
             ↓
주택 가격과 관련된 Target
             ↓
             y
```

의 관계를 학습한다.

Regression 문제이므로 `y`는 Class 번호가 아니라 연속적인 숫자이다.

---

## 6. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
```

Dataset을 Training Set과 Test Set으로 나눈다.

---

### test_size

```python
test_size=0.2
```

전체 Dataset의 20%를 Test Set으로 사용한다.

따라서

```text
Training Set → 80%
Test Set     → 20%
```

이 된다.

---

### random_state

```python
random_state=42
```

데이터를 나누는 Randomness를 고정한다.

같은 코드를 다시 실행했을 때 같은 Train/Test Split을 얻기 위해 사용한다.

---

## 7. 왜 Regression에서는 stratify를 사용하지 않았는가?

앞에서 Classification 코드를 작성할 때는 다음과 같은 옵션을 사용했다.

```python
stratify=y
```

Classification에서는 각 Class의 비율을 Train/Test Dataset에 비슷하게 유지하기 위해 사용할 수 있다.

하지만 이번 `y`는

```text
Class 0
Class 1
Class 2
```

같은 범주가 아니라 연속적인 실수이다.

따라서 일반적인 Regression 문제에서는 위와 같은 방식의 `stratify=y`를 그대로 사용하지 않는다.

---

## 8. Pipeline 구성

이번 코드의 핵심 부분은 다음과 같다.

```python
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(...)
        ),
    ]
)
```

Pipeline 내부의 데이터 흐름은 다음과 같다.

```text
Raw Input
    ↓
StandardScaler
    ↓
Scaled Input
    ↓
MLPRegressor
    ↓
Prediction
```

즉 하나의 `model` 안에

```text
Preprocessing
+
Neural Network
```

을 함께 넣은 것이다.

---

## 9. StandardScaler

```python
("scaler", StandardScaler())
```

MLP를 Training하기 전에 Feature들의 Scale을 맞춘다.

StandardScaler는 각 Feature를 일반적으로 다음과 같이 변환한다.

$$
x' = \frac{x-\mu}{\sigma}
$$

여기서

- $\mu$ : Training Data의 평균
- $\sigma$ : Training Data의 표준편차

이다.

결과적으로 각 Feature는 대략

```text
평균 ≈ 0
표준편차 ≈ 1
```

인 Scale을 가지게 된다.

---

## 10. Neural Network에서 Scaling이 중요한 이유

예를 들어 두 개의 Feature가 있다고 하자.

```text
Feature A → 0 ~ 10
Feature B → 0 ~ 100000
```

Scale 차이가 매우 크다.

Neural Network에서는 이러한 입력값들이 Weight와 곱해지고 Gradient 계산에도 영향을 미치기 때문에 Feature Scale이 크게 다르면 Optimization이 어려워질 수 있다.

따라서 MLP에서는 Input Feature를 Standardization하는 것이 일반적으로 중요하다.

---

## 11. Pipeline을 사용하는 장점

Pipeline을 사용하면 `fit()`을 호출할 때

```python
model.fit(
    X_train,
    y_train
)
```

내부적으로

```text
X_train
   ↓
StandardScaler.fit()
   ↓
StandardScaler.transform()
   ↓
MLPRegressor.fit()
```

순서로 처리된다.

Test Data를 예측할 때는

```python
model.predict(X_test)
```

만 호출하면

```text
X_test
   ↓
Training Data에서 학습한 Scaling 기준 적용
   ↓
MLPRegressor
   ↓
Prediction
```

이 자동으로 이루어진다.

따라서 Training과 Test Data에 동일한 Preprocessing 과정을 적용하기 쉽다.

---

## 12. MLPRegressor

```python
MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)
```

이번 모델에서 가장 중요한 부분이다.

각 Hyperparameter를 하나씩 살펴본다.

---

## 13. hidden_layer_sizes

```python
hidden_layer_sizes=(128, 64)
```

Hidden Layer 구조를 설정한다.

`(128, 64)`는

```text
첫 번째 Hidden Layer
→ 128 Neurons

두 번째 Hidden Layer
→ 64 Neurons
```

을 의미한다.

따라서 Network 구조는 다음과 같다.

```text
Input Layer
     ↓
128 Neurons
     ↓
64 Neurons
     ↓
Output Layer
```

Hidden Layer가 두 개이므로 단순한 선형 관계뿐 아니라 더 복잡한 Non-linear 관계를 학습할 수 있다.

---

## 14. 각 Layer의 계산

첫 번째 Hidden Layer에서는 개념적으로

$$
\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1
$$

를 계산한다.

그리고 ReLU를 적용한다.

$$
\mathbf{a}_1 = ReLU(\mathbf{z}_1)
$$

두 번째 Hidden Layer에서는

$$
\mathbf{z}_2 = \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2
$$

$$
\mathbf{a}_2 = ReLU(\mathbf{z}_2)
$$

를 계산한다.

마지막 Output Layer에서 Regression 결과를 출력한다.

전체적으로 보면 다음과 같다.

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
Regression Output
```

---

## 15. activation

```python
activation="relu"
```

Hidden Layer의 Activation Function으로 ReLU를 사용한다.

$$
ReLU(x) = \max(0,x)
$$

즉,

```text
x < 0
→ 0

x > 0
→ x
```

이다.

Non-linear Activation Function을 사용하기 때문에 MLP는 단순한 Linear Regression보다 복잡한 관계를 표현할 수 있다.

---

## 16. solver

```python
solver="adam"
```

Weight와 Bias를 Update하기 위한 Optimization Algorithm으로 Adam을 사용한다.

Training 과정은 개념적으로 다음과 같다.

```text
Forward Propagation
        ↓
Loss Calculation
        ↓
Backpropagation
        ↓
Gradient
        ↓
Adam
        ↓
Weight / Bias Update
```

scikit-learn에서는 이 과정이 `fit()` 내부에서 자동으로 수행된다.

---

## 17. alpha

```python
alpha=0.001
```

`alpha`는 Weight에 대한 **L2 Regularization의 강도**를 결정하는 Hyperparameter이다.

Neural Network의 Weight가 지나치게 커지는 것을 억제하여 Overfitting을 줄이는 데 사용한다.

개념적으로 Loss에 Weight 크기에 대한 Penalty가 추가된다고 생각할 수 있다.

$$
Loss_{total} = Loss_{data} + \lambda ||W||^2
$$

여기에서 `alpha`가 Regularization 강도와 관련된 역할을 한다.

```text
alpha가 작음
→ Regularization 약함

alpha가 큼
→ Regularization 강함
```

---

## 18. learning_rate_init

```python
learning_rate_init=0.001
```

초기 Learning Rate를 설정한다.

Parameter Update를 단순화해서 표현하면

$$
W_{new} = W_{old} - \eta \frac{\partial L}{\partial W}
$$

이다.

여기서

$$
\eta
$$

가 Learning Rate이다.

Learning Rate가 너무 크면 Training이 불안정해질 수 있고, 너무 작으면 학습 속도가 지나치게 느려질 수 있다.

---

## 19. max_iter

```python
max_iter=1000
```

Training을 수행할 최대 반복 횟수를 설정한다.

`adam`과 같은 Stochastic Solver를 사용할 때는 Training Data 전체를 반복해서 학습하는 최대 Epoch 수와 연결해서 이해할 수 있다.

하지만 이번 코드에서는

```python
early_stopping=True
```

를 사용하고 있기 때문에 반드시 1000회까지 Training하는 것은 아니다.

---

## 20. Early Stopping

```python
early_stopping=True
```

Training을 무조건 최대 반복 횟수까지 진행하지 않고, Validation 성능이 더 이상 좋아지지 않으면 Training을 일찍 종료하는 기능이다.

개념적으로 다음과 같다.

```text
Training
   ↓
Validation Score 확인
   ↓
계속 좋아짐?
   ├─ Yes → Training 계속
   │
   └─ No → 일정 기간 기다림
               ↓
            개선 없음
               ↓
              Stop
```

Early Stopping은 불필요하게 Training을 오래 진행하는 것을 줄이고 Overfitting을 완화하는 데 도움이 될 수 있다.

---

## 21. validation_fraction

```python
validation_fraction=0.1
```

`early_stopping=True`일 때 Training Data 중 일부를 Validation Set으로 사용한다.

현재 설정에서는 Training Data의 10%를 내부 Validation Data로 사용한다.

전체 데이터의 구조를 개념적으로 보면 다음과 같다.

```text
Original Dataset
        ↓
Train / Test Split
        ↓
┌──────────────────┬──────────────┐
│   Training 80%   │   Test 20%   │
└──────────────────┴──────────────┘
        ↓
Early Stopping용 내부 분리
        ↓
Training Data
+
Validation Data
```

중요한 점은 **Test Set을 Early Stopping 판단에 사용하지 않는다는 것**이다.

Test Set은 마지막 모델 평가를 위해 남겨둔다.

---

## 22. n_iter_no_change

```python
n_iter_no_change=20
```

Validation 성능이 개선되지 않는 상태를 몇 번까지 기다릴지 설정한다.

현재는

```text
20번 연속 의미 있는 개선이 없음
```

상태가 지속되면 Training을 종료할 수 있도록 설정한 것이다.

---

## 23. random_state

```python
random_state=42
```

Weight Initialization이나 Validation Split 등에 포함되는 Randomness를 고정하여 실행 결과를 재현하기 쉽게 만든다.

---

## 24. Hyperparameter 정리

이번 `MLPRegressor`에서 사용한 주요 Hyperparameter를 정리하면 다음과 같다.

| Hyperparameter | 의미 |
| --- | --- |
| `hidden_layer_sizes=(128, 64)` | Hidden Layer 2개, 각각 128 / 64 Neurons |
| `activation="relu"` | Hidden Layer의 Activation Function |
| `solver="adam"` | Optimizer |
| `alpha=0.001` | L2 Regularization 강도 |
| `learning_rate_init=0.001` | 초기 Learning Rate |
| `max_iter=1000` | 최대 Training 반복 횟수 |
| `early_stopping=True` | Validation 성능을 이용한 조기 종료 |
| `validation_fraction=0.1` | Training Data 중 Validation에 사용할 비율 |
| `n_iter_no_change=20` | 개선이 없을 때 기다릴 반복 횟수 |
| `random_state=42` | Randomness 재현 |

---

## 25. Training

```python
model.fit(
    X_train,
    y_train
)
```

이 한 줄에서 실제 Training이 이루어진다.

Pipeline을 사용했기 때문에 내부적으로는 먼저 StandardScaler가 적용되고, 그 다음 MLP가 Training된다.

개념적으로는 다음 과정이 반복된다.

```text
Input
  ↓
Standardization
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Adam
  ↓
Weight / Bias Update
  ↓
Repeat
```

---

## 26. Prediction

```python
y_pred = model.predict(
    X_test
)
```

Training이 끝난 모델에 Test Data를 입력하여 연속적인 값을 예측한다.

Classification이라면

```text
Class 0
Class 1
Class 2
```

와 같은 결과가 나오지만,

Regression에서는

```text
2.34
1.82
4.01
...
```

와 같이 연속적인 값이 출력된다.

---

## 27. Regression은 Accuracy를 사용하지 않는다

Classification에서는 다음과 같은 Accuracy를 사용할 수 있다.

$$
Accuracy = \frac{\text{Correct Predictions}} {\text{Total Predictions}}
$$

하지만 Regression에서는 Prediction과 Target이 연속적인 숫자이므로

```text
예측값 == 정답
```

인지를 기준으로 평가하기 어렵다.

예를 들어

```text
Target     = 3.14
Prediction = 3.13
```

이라면 완전히 같은 숫자는 아니지만 좋은 Prediction이라고 볼 수 있다.

따라서 Regression에서는 **Prediction과 실제 Target 사이의 오차 크기**를 평가하는 Metric을 사용한다.

---

## 28. MSE

```python
mse = mean_squared_error(
    y_test,
    y_pred
)
```

MSE는 **Mean Squared Error**, 평균 제곱 오차이다.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2
$$

Prediction Error를 제곱한 뒤 평균을 계산한다.

오차를 제곱하기 때문에 큰 Error에 상대적으로 큰 Penalty를 준다.

```text
Error = 1
→ 1² = 1

Error = 3
→ 3² = 9

Error = 10
→ 10² = 100
```

따라서 큰 Prediction Error에 민감하다.

일반적으로

```text
MSE ↓
→ Prediction Error ↓
→ 좋은 모델
```

로 해석한다.

---

## 29. RMSE

```python
rmse = mse ** 0.5
```

RMSE는 **Root Mean Squared Error**이다.

MSE에 Square Root를 적용한다.

$$
RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2 }
$$

MSE는 Error를 제곱하기 때문에 원래 Target과 단위가 달라진다.

RMSE는 다시 Square Root를 취하기 때문에 Target과 같은 단위로 Error를 해석할 수 있다는 장점이 있다.

```text
MSE
→ Error²

RMSE
→ Error
```

RMSE 역시 작을수록 좋다.

---

## 30. MAE

```python
mae = mean_absolute_error(
    y_test,
    y_pred
)
```

MAE는 **Mean Absolute Error**, 평균 절대 오차이다.

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i-\hat{y}_i|
$$

Prediction Error의 절댓값을 구하고 평균을 계산한다.

예를 들어 Error가

```text
-2
+3
-1
```

이라면

```text
2
3
1
```

로 만든 뒤 평균을 구한다.

따라서 MAE는

> **Prediction이 실제값에서 평균적으로 얼마나 떨어져 있는가**

를 직관적으로 해석하기 좋다.

---

## 31. MSE와 MAE의 차이

두 Metric의 가장 큰 차이는 큰 Error를 얼마나 강하게 반영하는가이다.

예를 들어 Error가

```text
1
1
10
```

이라고 하자.

MAE에서는

```text
1
1
10
```

그대로 반영한다.

MSE에서는

```text
1
1
100
```

으로 반영한다.

따라서

```text
MSE / RMSE
→ 큰 Error에 더 민감

MAE
→ Error 크기를 보다 직접적으로 평균
```

이라고 이해할 수 있다.

---

## 32. R² Score

```python
r2 = r2_score(
    y_test,
    y_pred
)
```

R²는 **Coefficient of Determination**, 결정계수이다.

모델이 Target의 변동을 얼마나 잘 설명하는지를 나타내는 지표이다.

일반적으로

```text
R² = 1
→ Prediction이 실제값과 완벽하게 일치

R² ≈ 0
→ 단순히 평균값으로 예측하는 수준과 비슷

R² < 0
→ 평균값으로 예측하는 것보다도 좋지 않을 수 있음
```

으로 해석할 수 있다.

따라서 R²는 MSE, RMSE, MAE와 달리 **클수록 좋은 Metric**이다.

---

## 33. 네 가지 평가 Metric 비교

| Metric | 의미 | 좋은 방향 | 특징 |
| --- | --- | --- | --- |
| MSE | 평균 제곱 오차 | ↓ | 큰 Error에 민감 |
| RMSE | MSE의 제곱근 | ↓ | Target과 같은 단위로 해석 가능 |
| MAE | 평균 절대 오차 | ↓ | 평균적인 Error 크기를 직관적으로 확인 |
| R² | 결정계수 | ↑ | 모델의 설명력을 확인 |

따라서 하나의 Metric만 보는 것보다 여러 Metric을 함께 확인하면 모델의 성능을 더 다양한 관점에서 볼 수 있다.

---

## 34. 결과 출력

```python
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")
```

`.4f`는 Floating Point 값을 소수점 네 자리까지 출력하겠다는 의미이다.

예를 들어

```python
value = 0.123456

print(f"{value:.4f}")
```

의 결과는

```text
0.1235
```

가 된다.

---

## 35. MLPClassifier와 MLPRegressor 비교

앞에서 사용했던 `MLPClassifier`와 이번 `MLPRegressor`를 비교하면 차이를 더 쉽게 이해할 수 있다.

| 구분 | MLPClassifier | MLPRegressor |
| --- | --- | --- |
| Task | Classification | Regression |
| Target | Discrete Class | Continuous Value |
| 예 | Setosa / Versicolor / Virginica | 주택 가격 |
| Hidden Layer | 사용 | 사용 |
| ReLU | 사용 가능 | 사용 가능 |
| Backpropagation | 사용 | 사용 |
| Adam | 사용 가능 | 사용 가능 |
| Prediction | Class | 연속적인 숫자 |
| 대표 평가 | Accuracy | MSE, RMSE, MAE, R² |

즉 모델 내부의 기본적인 Neural Network 구조는 매우 비슷하다.

가장 큰 차이는

> **최종적으로 무엇을 예측하는가?**

이다.

---

## 36. Perceptron → MLPClassifier → MLPRegressor

지금까지의 실습을 연결하면 다음과 같다.

```text
Perceptron
│
│ Linear Classification
│
↓

MLPClassifier
│
│ Hidden Layer
│ Non-linear Activation
│ Classification
│
↓

MLPRegressor
│
│ Hidden Layer
│ Non-linear Activation
│ Regression
│
↓
Continuous Prediction
```

`MLPClassifier`와 `MLPRegressor` 모두 MLP이지만 해결하는 Task가 다르다.

---

## 37. 코드 전체를 개념과 연결하기

| 코드 | 의미 |
| --- | --- |
| `fetch_california_housing()` | Dataset Load |
| `X` | Input Features |
| `y` | Regression Target |
| `train_test_split()` | Train / Test Dataset 분리 |
| `StandardScaler()` | Feature Standardization |
| `Pipeline()` | Preprocessing + Model 연결 |
| `MLPRegressor()` | Regression용 Multi-Layer Perceptron |
| `(128, 64)` | Hidden Layer 구조 |
| `"relu"` | Non-linear Activation |
| `"adam"` | Optimizer |
| `alpha` | L2 Regularization |
| `learning_rate_init` | Learning Rate |
| `early_stopping` | Validation 성능에 따른 조기 종료 |
| `fit()` | Training |
| `predict()` | Prediction |
| `mean_squared_error()` | MSE |
| `mean_absolute_error()` | MAE |
| `r2_score()` | R² |

---

## 38. 전체 흐름

이번 코드를 하나의 흐름으로 정리하면 다음과 같다.

```text
California Housing Dataset
            ↓
     Feature / Target
            ↓
     Train / Test Split
            ↓
       StandardScaler
            ↓
       MLPRegressor
            ↓
       128 Neurons
            ↓
           ReLU
            ↓
        64 Neurons
            ↓
           ReLU
            ↓
     Regression Output
            ↓
        Prediction
            ↓
 ┌──────────┼──────────┐
 ↓          ↓          ↓
MSE       RMSE        MAE
             +
             ↓
             R²
```

---

## 핵심 정리

이번 실습에서는 scikit-learn의 `MLPRegressor`를 이용하여 연속적인 값을 예측하는 Regression 문제를 구현하였다.

모델의 구조는

```text
Input
↓
StandardScaler
↓
Hidden Layer 128
↓
ReLU
↓
Hidden Layer 64
↓
ReLU
↓
Output
```

으로 구성된다.

Training 과정에서는 Adam Optimizer가 Backpropagation으로 계산된 Gradient를 이용하여 Weight와 Bias를 업데이트한다.

또한

```python
early_stopping=True
```

를 사용하여 Validation 성능이 더 이상 개선되지 않을 경우 Training을 일찍 종료하도록 구성하였다.

마지막으로 Regression에서는 Classification의 Accuracy 대신

```text
MSE
RMSE
MAE
R²
```

와 같은 Metric을 이용하여 Prediction Error와 모델의 설명력을 평가한다.

---

## 한 문장으로 정리

`MLPRegressor`는 **Hidden Layer와 Non-linear Activation Function을 이용하여 Input Feature와 연속적인 Target 사이의 복잡한 관계를 학습하는 Multi-Layer Neural Network 기반 Regression Model**이며, `StandardScaler → MLPRegressor → MSE/RMSE/MAE/R²`의 흐름으로 전처리부터 학습과 평가까지 하나의 Pipeline으로 구성할 수 있다.
