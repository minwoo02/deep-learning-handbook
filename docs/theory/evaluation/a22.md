# Cross Validation 완전 정리: Hold-out부터 K-Fold·Stratified K-Fold와 모델 일반화 성능 평가까지

[TOC]



---

Machine Learning과 Deep Learning에서는 Training Dataset을 이용하여 모델의 Parameter를 학습한다.

그러나 Training Data에서 높은 성능을 보인다고 해서 새로운 데이터에서도 높은 성능을 보장할 수 있는 것은 아니다.

예를 들어 어떤 Classification Model이 다음과 같은 성능을 보였다고 하자.

```text
Training Accuracy   : 99%
Validation Accuracy : 73%
```

이 모델은 Training Data에는 매우 잘 적합하지만, 학습에 사용되지 않은 데이터에서는 상대적으로 낮은 성능을 보인다.

이러한 현상은 **Overfitting(과적합)** 과 관련이 있다.

따라서 모델을 평가할 때는 단순히 Training Performance를 확인하는 것이 아니라, **학습에 사용하지 않은 데이터에 대해서도 좋은 예측 성능을 보이는지** 확인해야 한다.

이와 관련된 중요한 개념이 **Generalization(일반화)** 이다.

Generalization은 모델이 학습에 사용하지 않은 새로운 데이터에서도 적절한 예측 성능을 나타내는 능력을 의미한다.

이를 평가하기 위해 사용되는 대표적인 방법이 **Cross Validation(교차검증)** 이다.

이 글에서는 다음 내용을 다룬다.

- Training·Validation·Test Set의 역할
- Hold-out Validation과 Cross Validation의 차이
- K-Fold Cross Validation의 동작 원리
- Stratified K-Fold와 Shuffle Split
- Group K-Fold와 Time Series Cross Validation
- Cross Validation과 Hyperparameter Tuning
- Data Leakage 및 Cross Validation 사용 시 주의사항

---

## 1. Model Validation이란?

Machine Learning Model의 Training은 일반적으로 다음과 같은 과정으로 진행된다.

```text
Training Data
      │
      ▼
Model Training
      │
      ▼
Parameter Optimization
      │
      ▼
Trained Model
      │
      ▼
Performance Evaluation
```

하지만 학습에 사용한 데이터로만 모델을 평가하면 실제 예측 성능을 정확하게 판단하기 어렵다.

모델은 Training Dataset에 존재하는 패턴뿐 아니라, 해당 데이터에만 존재하는 Noise나 우연한 특성까지 학습할 수 있기 때문이다.

따라서 Training Data와 별도의 데이터를 이용하여 모델을 평가하는 과정이 필요하다.

이를 위해 일반적으로 Dataset을 다음과 같이 구분한다.

| Dataset | 역할 |
| --- | --- |
| Training Set | Model Parameter 학습 |
| Validation Set | Model Selection 및 Hyperparameter Tuning |
| Test Set | 최종 Generalization Performance 평가 |

이 세 Dataset은 서로 다른 역할을 가진다.

---

### 1.1 Training Set

Training Set은 모델이 실제로 학습하는 데이터이다.

Neural Network에서는 Training Set을 이용하여 Forward Propagation, Loss Calculation, Backpropagation을 수행한다.

$$
\boxed{ \theta^* = \arg\min_\theta \mathcal{L}_{train}(\theta) }
$$

여기서 $\theta$는 Weight와 Bias를 포함하는 Model Parameter이다.

즉 Training Set은 Parameter Optimization을 수행하는 데 사용된다.

---

### 1.2 Validation Set

Validation Set은 학습한 모델의 성능을 확인하고 모델 설정을 선택하기 위해 사용하는 데이터이다.

예를 들어 다음과 같은 Hyperparameter를 비교할 수 있다.

```text
Learning Rate
Batch Size
Number of Hidden Layers
Number of Neurons
Regularization Strength
Optimizer
```

모델을 Training Set으로 학습한 뒤 Validation Set에서 성능을 비교하여 적절한 Hyperparameter를 선택한다.

따라서 Validation Set은 일반적으로 해당 학습 과정에서 직접적인 Parameter Update에 사용되지 않는다.

다만 여러 차례 Validation Performance를 확인하여 Hyperparameter를 선택하기 때문에, 모델 개발 과정 전체에서는 Validation Set에도 간접적으로 적합될 수 있다.

---

### 1.3 Test Set

Test Set은 모델 개발과 선택이 끝난 후 최종적인 Generalization Performance를 평가하기 위한 데이터이다.

Test Set을 반복적으로 이용하여 Hyperparameter를 변경하거나 모델을 선택하면 Test Set에 대한 Overfitting이 발생할 수 있다.

따라서 원칙적으로 Test Set은 최종 평가 이전에 모델 선택 과정에 사용하지 않는다.

전체 과정을 정리하면 다음과 같다.

```text
Original Dataset
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
Training / Validation          Test Set
       │                         │
       ▼                         │
Model Training                  │
       │                         │
       ▼                         │
Hyperparameter Tuning           │
       │                         │
       ▼                         │
Model Selection                 │
       │                         │
       └──────────┬──────────────┘
                  │
                  ▼
          Final Evaluation
```

---

## 2. Hold-out Validation

가장 단순한 Validation 방법은 **Hold-out Validation**이다.

Hold-out Validation은 데이터를 Training Set과 Validation Set으로 한 번 분할하여 성능을 평가하는 방법이다.

예를 들어 1,000개의 Sample이 있다고 하자.

이를 다음과 같이 나눌 수 있다.

```text
Original Data: 1,000 Samples
              │
              ▼
┌────────────────────────┬─────────────┐
│     Training Set       │ Validation  │
│        800             │     200     │
└────────────────────────┴─────────────┘
             80%               20%
```

모델은 800개의 Training Sample로 학습하고, 200개의 Validation Sample로 평가된다.

이러한 방식은 구현이 간단하고 계산 비용이 비교적 적다.

---

### 2.1 Hold-out Validation의 한계

Hold-out Validation에서는 데이터가 한 번만 분할된다.

따라서 **어떤 Sample이 Training Set과 Validation Set에 포함되는지에 따라 평가 결과가 달라질 수 있다.**

예를 들어 같은 Dataset을 서로 다른 Random Seed로 분할했다고 하자.

| Split | Validation Accuracy |
| --- | --- |
| A | 91% |
| B | 84% |
| C | 88% |

모델과 Hyperparameter가 같더라도 Validation Set의 구성에 따라 점수가 달라질 수 있다.

특히 데이터가 충분하지 않다면 이러한 문제가 더 두드러질 수 있다.

또한 Validation Set을 따로 확보하면 그만큼 Training에 사용할 수 있는 데이터가 줄어든다.

즉 Hold-out Validation에는 다음과 같은 Trade-off가 있다.

```text
Training Set을 크게 확보
           │
           ▼
Validation Set 감소
           │
           ▼
Validation Performance의 변동 가능성 증가
```

반대로 Validation Set을 크게 확보하면 한 번의 평가에 사용하는 데이터는 많아지지만 Training에 사용할 수 있는 Sample은 줄어든다.

이러한 문제를 완화하기 위한 방법 중 하나가 Cross Validation이다.

---

## 3. Cross Validation이란?

**Cross Validation(CV, 교차검증)** 은 데이터를 여러 부분으로 나누고, Training과 Validation에 사용되는 데이터를 바꿔가며 모델을 여러 번 학습하고 평가하는 방법이다.

핵심은 다음과 같다.

> 데이터를 한 번만 분할하여 평가하는 대신, 여러 번의 Training·Validation을 수행하고 그 결과를 종합하여 모델의 성능을 추정한다.

대표적인 방법은 **K-Fold Cross Validation**이다.

K-Fold에서는 Dataset을 K개의 Fold로 나눈다.

각 Fold를 한 번씩 Validation Set으로 사용하고, 나머지 K-1개의 Fold로 모델을 학습한다.

이러한 과정을 총 K번 반복한다.

```text
Dataset
   │
   ▼
K개의 Fold로 분할
   │
   ▼
Fold별 Training / Validation
   │
   ▼
K개의 독립적인 Model Training
   │
   ▼
K개의 Validation Score
   │
   ▼
평균 및 변동성 확인
```

Cross Validation은 하나의 고정된 분할 결과에만 의존하지 않고, 여러 분할에서 모델의 성능을 관찰한다는 특징이 있다.

---

## 4. K-Fold Cross Validation

K-Fold Cross Validation은 Dataset을 K개의 부분집합으로 나누는 방법이다.

각 부분집합을 **Fold**라고 한다.

예를 들어 다음과 같이 설정했다고 하자.

$$
\boxed{ K=5 }
$$

이를 **5-Fold Cross Validation**이라고 한다.

Dataset을 다음과 같이 5개의 Fold로 분할한다.

```text
Dataset
│
├── Fold 1
├── Fold 2
├── Fold 3
├── Fold 4
└── Fold 5
```

각 반복에서 하나의 Fold를 Validation Set으로 사용하고, 나머지 Fold를 Training Set으로 사용한다.

---

### 4.1 5-Fold Cross Validation의 동작

다음 표에서 `Train`은 학습에 사용하는 Fold이며, `Valid`는 검증에 사용하는 Fold이다.

| Iteration | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | Valid | Train | Train | Train | Train |
| 2 | Train | Valid | Train | Train | Train |
| 3 | Train | Train | Valid | Train | Train |
| 4 | Train | Train | Train | Valid | Train |
| 5 | Train | Train | Train | Train | Valid |

첫 번째 반복에서는 Fold 1을 Validation Set으로 사용한다.

두 번째 반복에서는 Fold 2를 Validation Set으로 사용한다.

이 과정을 반복하여 마지막에는 Fold 5를 Validation Set으로 사용한다.

따라서 5개의 Fold가 각각 한 번씩 Validation Set으로 사용된다.

---

### 4.2 K-Fold의 중요한 특징

기본 K-Fold에서는 다음과 같은 특징이 있다.

**① 각 Sample이 Validation에 한 번씩 참여한다.**

5-Fold Cross Validation에서 각 Sample은 정확히 한 번 Validation Set에 포함된다.

**② 각 Sample은 여러 번 Training에 사용된다.**

각 Sample은 자신이 Validation Set에 포함되는 한 번을 제외하고 나머지 반복에서는 Training Set에 포함된다.

따라서 5-Fold에서는 각 Sample이 4번의 Training에 참여한다.

**③ 총 K개의 모델을 학습한다.**

5-Fold Cross Validation에서는 총 5개의 Model Training이 수행된다.

**④ 결과적으로 K개의 Validation Score를 얻는다.**

각 반복에서 계산한 Validation Score를 종합하여 모델의 성능을 평가한다.

---

## 5. K-Fold에서 데이터는 얼마나 사용되는가?

전체 Sample 수를 $N$이라고 하자.

K-Fold에서 각 Fold의 Sample 수가 동일하다면 하나의 Validation Fold는 다음과 같은 크기를 가진다.

$$
\boxed{ N_{valid}=\frac{N}{K} }
$$

하나의 반복에서 Training에 사용하는 Sample 수는 다음과 같다.

$$
\boxed{ N_{train} = N\frac{K-1}{K} }
$$

예를 들어 다음과 같은 상황을 생각하자.

$$
N=1000
$$

$$
K=5
$$

그러면

$$
N_{valid} = \frac{1000}{5} = 200
$$

이다.

Training Sample 수는 다음과 같다.

$$
N_{train} = 1000\times\frac45 = 800
$$

이다.

따라서 매 반복에서

```text
Training   : 800 Samples
Validation : 200 Samples
```

을 사용한다.

하지만 각 반복에서 사용되는 Training과 Validation Sample의 구성은 달라진다.

전체적으로 보면 모든 Sample이 Validation에 한 번씩 사용되므로 한 번의 Hold-out 분할보다 다양한 데이터 구성에서 모델의 성능을 확인할 수 있다.

---

## 6. Cross Validation에서는 모델을 K번 새로 학습한다

Cross Validation을 이해할 때 중요한 부분이다.

K-Fold Cross Validation은 하나의 모델이 Fold를 바꿔가며 계속 학습하는 과정이 아니다.

**각 Fold마다 새로운 모델을 초기 상태에서 학습한다.**

예를 들어 5-Fold에서는 다음과 같이 진행된다.

```text
Fold 1 Validation
        │
        ▼
Model 1 생성 및 학습
        │
        ▼
Validation Score 1


Fold 2 Validation
        │
        ▼
Model 2 생성 및 학습
        │
        ▼
Validation Score 2


             ...
             

Fold 5 Validation
        │
        ▼
Model 5 생성 및 학습
        │
        ▼
Validation Score 5
```

각 모델은 서로 다른 Training Subset으로 학습된다.

따라서 각 모델의 Weight와 Bias도 서로 다를 수 있다.

또한 Training Random Seed나 Stochastic Optimization 등의 영향으로 결과에 추가적인 변동이 발생할 수 있다.

이 과정을 통해 하나의 분할에서만 나타나는 평가 결과가 아니라 여러 분할에서의 Performance를 관찰할 수 있다.

---

## 7. Cross Validation Score

K-Fold Cross Validation에서는 K개의 Validation Score를 얻는다.

이를 다음과 같이 나타낼 수 있다.

$$
\boxed{ s_1,s_2,\ldots,s_K }
$$

일반적으로 평균을 이용하여 성능을 요약한다.

$$
\boxed{ \bar{s} = \frac1K\sum_{i=1}^{K}s_i }
$$

예를 들어 5-Fold Cross Validation에서 Accuracy가 다음과 같았다고 하자.

| Fold | Validation Accuracy |
| --- | --- |
| Fold 1 | 80% |
| Fold 2 | 84% |
| Fold 3 | 82% |
| Fold 4 | 78% |
| Fold 5 | 86% |

평균 Accuracy는 다음과 같다.

$$
\bar{s} = \frac{0.80+0.84+0.82+0.78+0.86}{5}
$$

따라서

$$
\boxed{ \bar{s}=0.82 }
$$

이다.

즉 평균 Validation Accuracy는 82%이다.

이 값은 해당 Cross Validation 설정에서 관찰된 평균적인 모델 성능을 의미한다.

---

### 7.1 Standard Deviation

평균뿐 아니라 Fold별 Score의 변동도 확인할 수 있다.

대표적인 방법이 Standard Deviation(표준편차)이다.

$$
\boxed{ \sigma_s = \sqrt{ \frac1K \sum_{i=1}^{K} (s_i-\bar{s})^2 } }
$$

이는 Fold Score를 K개 값의 집합으로 보고 계산한 표준편차이다.

Fold별 Score가 비슷하면 Standard Deviation이 작게 나타난다.

반대로 Fold별 Score의 차이가 크면 Standard Deviation도 커질 수 있다.

다만 **Cross Validation Score의 Standard Deviation을 모델의 일반화 성능에 대한 정확한 Confidence Interval로 해석해서는 안 된다.**

각 Fold의 Training Set은 서로 상당 부분 겹치므로 평가 결과가 완전히 독립적인 관측값은 아니기 때문이다.

또한 분할 방식에 따라 실제 Performance Variability가 과소평가되거나 과대평가될 수 있다.

따라서 Fold Score의 평균과 표준편차는 모델 비교에 유용한 요약 통계량이지만, 통계적 불확실성을 완벽하게 나타내는 값은 아니다.

---

## 8. K 값은 어떻게 결정할까?

K-Fold에서 K는 Fold의 개수를 결정하는 Hyperparameter이다.

대표적으로 다음과 같은 값을 사용한다.

$$
\boxed{ K=5,\quad K=10 }
$$

K의 크기에 따라 Training과 Validation에 사용되는 Sample 비율이 달라진다.

| K | Training 비율 | Validation 비율 | Model Training 횟수 |
| --- | --- | --- | --- |
| 2 | 50% | 50% | 2 |
| 5 | 80% | 20% | 5 |
| 10 | 90% | 10% | 10 |
| 20 | 95% | 5% | 20 |

K가 커질수록 하나의 반복에서 Training에 사용하는 데이터 비율이 증가한다.

하지만 Validation Fold의 크기는 작아지고 Model Training 횟수도 증가한다.

따라서 계산 비용과 평가 특성 사이의 Trade-off가 존재한다.

K를 크게 설정한다고 항상 더 정확한 성능 추정이 가능한 것은 아니다.

모델의 특성과 데이터의 구조, 평가 지표 및 계산 자원 등을 고려하여 K를 선택해야 한다.

---

## 9. Leave-One-Out Cross Validation

K-Fold Cross Validation의 특수한 형태가 **Leave-One-Out Cross Validation(LOOCV)** 이다.

LOOCV에서는 하나의 Sample만 Validation Set으로 사용한다.

즉 전체 Sample 수가 $N$이라면

$$
\boxed{ K=N }
$$

이다.

예를 들어 5개의 Sample이 있다고 하자.

```text
Sample A
Sample B
Sample C
Sample D
Sample E
```

LOOCV는 다음과 같이 수행된다.

| Iteration | Training Set | Validation Set |
| --- | --- | --- |
| 1 | B, C, D, E | A |
| 2 | A, C, D, E | B |
| 3 | A, B, D, E | C |
| 4 | A, B, C, E | D |
| 5 | A, B, C, D | E |

각 반복에서는 $N-1$개의 Sample로 학습하고 1개의 Sample로 평가한다.

따라서 Training Data를 최대한 많이 사용할 수 있다는 특징이 있다.

하지만 $N$번의 Training이 필요하므로 Dataset이 크면 계산 비용이 매우 증가한다.

또한 하나의 Validation Result는 단 하나의 Sample에 대한 결과이므로 Fold별 Score의 변동이 클 수 있다.

따라서 LOOCV가 일반적인 5-Fold 또는 10-Fold보다 항상 유리한 것은 아니다.

---

## 10. Stratified K-Fold Cross Validation

Classification에서는 Class Distribution을 고려해야 하는 경우가 많다.

예를 들어 다음과 같은 Binary Classification Dataset이 있다고 하자.

```text
Class 0 : 900 Samples
Class 1 : 100 Samples
```

전체 데이터의 Class Distribution은 다음과 같다.

$$
P(y=0)=0.9
$$

$$
P(y=1)=0.1
$$

이다.

단순히 데이터를 무작위로 분할하면 일부 Fold에서 Class Distribution이 크게 달라질 수 있다.

특히 Minority Class의 Sample 수가 매우 적으면 특정 Fold에 해당 Class가 충분히 포함되지 않을 수 있다.

이러한 문제를 완화하기 위한 방법이 **Stratified K-Fold**이다.

---

### 10.1 Stratified Sampling

Stratified Sampling은 전체 데이터의 Class Distribution이 각 Fold에서 최대한 유지되도록 데이터를 분할하는 방법이다.

예를 들어 전체 Dataset에서 Class 0과 Class 1의 비율이 9:1이라면, 각 Fold에서도 그 비율을 가능한 한 유지하도록 분할한다.

```text
Original Dataset
Class 0 : 90%
Class 1 : 10%
        │
        ▼
Stratified K-Fold
        │
        ├── Fold 1 : 약 90% / 10%
        ├── Fold 2 : 약 90% / 10%
        ├── Fold 3 : 약 90% / 10%
        ├── Fold 4 : 약 90% / 10%
        └── Fold 5 : 약 90% / 10%
```

Class Distribution을 유지함으로써 각 Fold에서 Classification Performance를 보다 일관된 조건에서 평가할 수 있다.

단, Class별 Sample 수가 너무 적다면 모든 Fold에 해당 Class를 충분히 포함하는 것이 불가능할 수도 있다.

또한 Stratification은 Fold 간 Class 비율을 비슷하게 만들어 Metric의 변동성을 작게 보이게 할 수 있으므로, 작은 표준편차가 곧 높은 통계적 확실성을 의미하지는 않는다.

---

## 11. K-Fold와 Stratified K-Fold 비교

두 방법의 핵심 차이는 Class Distribution을 고려하는지 여부이다.

| 구분 | K-Fold | Stratified K-Fold |
| --- | --- | --- |
| 데이터 분할 | K개 Fold로 분할 | Class Distribution을 고려하여 분할 |
| 주요 목적 | 여러 분할에서 성능 평가 | Class 비율을 유지한 성능 평가 |
| 주로 사용 | Regression 및 일반적인 데이터 | Classification |
| Class Imbalance 고려 | 자동으로 보장하지 않음 | 최대한 유지 |
| Validation Fold | 서로 겹치지 않음 | 서로 겹치지 않음 |

일반적인 Classification에서는 Stratified K-Fold를 기본적인 선택지로 고려할 수 있다.

반면 Regression에서는 일반적인 K-Fold를 사용하는 경우가 많다.

다만 데이터에 시간적 순서나 그룹 의존성이 있다면 K-Fold나 Stratified K-Fold 대신 다른 분할 방식이 필요할 수 있다.

---

## 12. Shuffle Split

K-Fold는 Dataset을 서로 겹치지 않는 K개의 Validation Fold로 분할한다.

반면 **Shuffle Split**은 반복할 때마다 데이터를 무작위로 다시 나누는 방법이다.

예를 들어 100개의 Sample에서 20%를 Validation Set으로 사용한다고 하자.

```text
Iteration 1
Random Split → 80 Train / 20 Valid

Iteration 2
Random Split → 80 Train / 20 Valid

Iteration 3
Random Split → 80 Train / 20 Valid
```

각 반복에서 새로운 Random Split을 생성한다.

따라서 이전 반복에서 Validation Set에 포함되었던 Sample이 다음 반복에서도 Validation Set에 포함될 수 있다.

이 점이 K-Fold와의 중요한 차이이다.

---

### 12.1 K-Fold와 Shuffle Split 비교

| 구분 | K-Fold | Shuffle Split |
| --- | --- | --- |
| Validation 분할 | 서로 겹치지 않는 Fold | 반복마다 무작위 분할 |
| Validation Sample 중복 | 기본 K-Fold에서는 없음 | 반복 간 중복 가능 |
| 반복 횟수 | K | 사용자가 설정 |
| Validation 비율 | 약 $1/K$ | `test_size`로 지정 가능 |
| 전체 Sample의 Validation 참여 | 정확히 한 번 | 보장되지 않음 |

Shuffle Split은 Validation Set의 비율과 반복 횟수를 독립적으로 조절할 수 있다는 특징이 있다.

또한 **Stratified Shuffle Split**을 사용하면 Class Distribution까지 고려하여 무작위 분할을 수행할 수 있다.

---

## 13. Group K-Fold

지금까지 설명한 K-Fold는 데이터의 각 Sample이 서로 독립적이라는 가정과 잘 맞는다.

하지만 실제 Dataset에서는 Sample 사이에 강한 연관성이 존재하는 경우가 있다.

예를 들어 의료 데이터에서 같은 환자로부터 여러 측정값을 수집했다고 하자.

```text
Patient A
 ├── Measurement 1
 ├── Measurement 2
 └── Measurement 3

Patient B
 ├── Measurement 1
 └── Measurement 2
```

일반적인 K-Fold를 사용하면 같은 환자의 데이터 일부는 Training Set에, 나머지는 Validation Set에 포함될 수 있다.

이 경우 모델이 환자 고유의 특성을 학습하여 Validation Score가 실제 새로운 환자에 대한 성능보다 높게 나타날 수 있다.

이를 방지하기 위해 **Group K-Fold**를 사용할 수 있다.

---

### 13.1 Group K-Fold의 원리

Group K-Fold에서는 같은 Group에 속하는 Sample이 Training Set과 Validation Set에 동시에 포함되지 않도록 분할한다.

```text
Patient A ─┐
Patient B ─┼── Training
Patient C ─┘

Patient D ─┐
Patient E ─┴── Validation
```

즉 Patient ID와 같은 Group Identifier를 기준으로 분할한다.

이러한 방식은 서로 관련된 Sample의 정보가 Training과 Validation에 동시에 존재하는 것을 방지하는 데 도움이 된다.

특히 다음과 같은 Dataset에서 중요하다.

- 동일 환자의 반복 측정 데이터
- 동일 실험 대상에서 수집한 여러 Signal Segment
- 동일 장비 또는 Session에서 수집한 여러 Sample
- 동일 사용자로부터 수집한 다수의 측정값

다만 Group을 분리해야 하는 이유와 평가하려는 목표는 함께 고려해야 한다.

예를 들어 새로운 환자에 대한 예측 성능을 평가하려면 Patient 단위 분할이 적절하지만, 같은 환자의 미래 측정값을 예측하려는 문제라면 평가 설계가 달라질 수 있다.

---

## 14. Time Series Cross Validation

Time Series Dataset에는 시간적인 순서가 존재한다.

예를 들어 다음과 같은 데이터가 있다고 하자.

```text
January → February → March → April → May
```

일반적인 Random K-Fold를 적용하면 미래 데이터가 Training Set에 포함되고 과거 데이터가 Validation Set에 포함될 수 있다.

이 경우 실제 미래 예측 상황과 다른 평가가 수행될 수 있다.

따라서 시간 순서를 유지하는 Validation 전략이 필요하다.

대표적인 방법이 **Time Series Split**이다.

---

### 14.1 Time Series Split의 원리

시간 순서에 따라 과거 데이터로 모델을 학습하고 이후 데이터를 검증한다.

예를 들어 다음과 같다.

```text
Split 1
[ Train ] [ Valid ]

Split 2
[   Train   ] [ Valid ]

Split 3
[      Train      ] [ Valid ]

Split 4
[         Train         ] [ Valid ]
```

이러한 방식은 과거에서 미래로 예측하는 실제 환경을 반영한다.

다만 데이터 간 시간적 의존성이 강하다면 Training과 Validation 사이에 시간적 간격(Gap)을 둘 필요도 있다.

따라서 Time Series Cross Validation의 핵심은 단순히 여러 번 평가하는 것이 아니라 **시간 순서와 예측 시점에서 사용 가능한 정보의 범위를 유지하는 것**이다.

---

## 15. Cross Validation과 Hyperparameter Tuning

Cross Validation은 Model Performance Evaluation뿐 아니라 Hyperparameter Tuning에도 사용된다.

예를 들어 다음 Learning Rate Candidate가 있다고 하자.

```text
Learning Rate
─────────────
0.1
0.01
0.001
```

각 Learning Rate에 대해 5-Fold Cross Validation을 수행할 수 있다.

| Learning Rate | Fold 평균 Accuracy |
| --- | --- |
| 0.1 | 81% |
| 0.01 | 86% |
| 0.001 | 83% |

위 값이 예시적인 결과라면 평균 Validation Accuracy가 가장 높은 `0.01`을 선택할 수 있다.

따라서 Cross Validation은 다음 과정에 활용된다.

```text
Hyperparameter Candidates
           │
           ▼
K-Fold Cross Validation
           │
           ▼
각 설정의 평균 Validation Score
           │
           ▼
Hyperparameter Comparison
           │
           ▼
Model Selection
```

이 과정은 Hyperparameter Tuning의 기본적인 구조이다.

대표적인 자동 탐색 방법으로는 Grid Search와 Random Search가 있으며, scikit-learn에서는 `GridSearchCV`, `RandomizedSearchCV` 등의 기능을 제공한다.

---

## 16. Cross Validation과 Test Set의 관계

Cross Validation을 사용한다면 별도의 Test Set이 필요하지 않을까?

일반적인 모델 개발에서는 **별도의 Test Set을 유지하는 것이 중요하다.**

Cross Validation은 주로 Training에 사용할 수 있는 데이터 내부에서 Model Selection과 Hyperparameter Tuning을 수행하기 위해 사용된다.

예를 들어 전체 Dataset이 1,000개라고 하자.

먼저 다음과 같이 분할한다.

```text
Original Dataset: 1,000 Samples
              │
              ├── Development Set : 800
              │
              └── Test Set        : 200
```

이제 Development Set의 800개 Sample에 대해 5-Fold Cross Validation을 수행한다.

하나의 Fold에서는 다음과 같이 사용한다.

$$
N_{train}=800\times\frac45=640
$$

$$
N_{valid}=800\times\frac15=160
$$

따라서

```text
Development Set: 800
          │
          ▼
5-Fold Cross Validation
          │
          ├── 640 Train / 160 Valid
          ├── 640 Train / 160 Valid
          ├── 640 Train / 160 Valid
          ├── 640 Train / 160 Valid
          └── 640 Train / 160 Valid
          │
          ▼
Hyperparameter Selection
          │
          ▼
Final Model Training
          │
          ▼
Independent Test Set: 200
          │
          ▼
Final Performance Evaluation
```

이때 Test Set 200개는 Cross Validation과 Hyperparameter Tuning에 사용하지 않는다.

---

## 17. Cross Validation 이후 Final Model은 어떻게 만드는가?

K-Fold Cross Validation에서는 K개의 모델이 만들어진다.

그렇다면 그중 가장 높은 Validation Score를 얻은 모델을 최종 모델로 사용해야 할까?

**일반적인 목적이 모델 선택 및 Hyperparameter Tuning이라면 반드시 그렇지는 않다.**

Cross Validation의 주된 역할은 모델 구조나 Hyperparameter Configuration을 평가하는 것이다.

일반적인 절차는 다음과 같다.

1. K-Fold Cross Validation으로 후보 설정을 비교한다.
2. 적절한 Hyperparameter를 선택한다.
3. 선택한 설정으로 Development Set 전체를 사용하여 새로운 모델을 학습한다.
4. 독립적인 Test Set에서 최종 평가를 수행한다.

```text
K-Fold Cross Validation
           │
           ▼
Best Hyperparameter Selection
           │
           ▼
Development Set 전체로 재학습
           │
           ▼
Final Model
           │
           ▼
Independent Test Evaluation
```

따라서 Cross Validation은 일반적으로 최종 모델의 Weight를 직접 결정하는 과정이 아니라, 최종 모델을 구성할 설정을 선택하고 성능을 추정하는 과정이다.

물론 K개의 모델을 결합하는 Ensemble이나 Out-of-Fold Prediction처럼 다른 목적으로 Cross Validation을 활용하는 방법도 존재한다.

---

## 18. Cross Validation에서 Data Leakage 주의

Cross Validation을 수행할 때 가장 중요하게 고려해야 할 문제 중 하나는 **Data Leakage(데이터 누수)** 이다.

Data Leakage는 모델을 학습할 때 실제 Prediction 시점에 알 수 없는 정보가 Training Process에 포함되는 현상이다.

예를 들어 전체 Dataset을 먼저 Standardization한 후 K-Fold Cross Validation을 수행한다고 하자.

```text
Entire Dataset
       │
       ▼
StandardScaler.fit_transform()
       │
       ▼
K-Fold Cross Validation
```

이 경우 StandardScaler가 Validation에 들어갈 Sample까지 포함하여 평균과 표준편차를 계산한다.

따라서 Validation Set에 관한 정보가 Training 과정에 간접적으로 유입된다.

이러한 절차는 올바른 Validation Performance를 왜곡할 수 있다.

---

### 18.1 올바른 Preprocessing 순서

Preprocessing 과정에서는 Training Fold에서만 Transformation Parameter를 학습해야 한다.

```text
K-Fold Split
     │
     ├── Training Fold
     │       │
     │       ▼
     │   Scaler.fit()
     │       │
     │       ▼
     │   Scaler.transform()
     │       │
     │       ▼
     │   Model Training
     │
     └── Validation Fold
             │
             ▼
       Training Fold에서
       학습한 Scaler 사용
             │
             ▼
         Validation
```

이 원칙은 Standardization뿐 아니라 다음과 같은 작업에도 적용된다.

- Missing Value Imputation
- Feature Selection
- PCA
- Data-dependent Normalization
- Target Encoding

즉 Training 과정에서 데이터로부터 학습되는 전처리 단계는 Cross Validation의 각 Training Fold 내부에서 수행해야 한다.

scikit-learn에서는 `Pipeline`을 이용하면 이러한 실수를 줄일 수 있다.

---

## 19. Nested Cross Validation

Hyperparameter Tuning과 모델 성능평가를 동일한 Cross Validation 결과에 의존하면 선택 편향(Selection Bias)이 발생할 수 있다.

예를 들어 여러 Hyperparameter 중 가장 높은 CV Score를 얻은 설정을 선택하고, 그 점수를 그대로 최종 일반화 성능으로 보고한다고 하자.

많은 후보를 비교하는 과정에서 우연히 높은 Validation Score를 얻은 설정이 선택될 수 있다.

따라서 선택에 사용된 점수는 실제 Generalization Performance보다 낙관적일 수 있다.

이 문제를 줄이기 위한 방법이 **Nested Cross Validation**이다.

---

### 19.1 Nested CV의 기본 구조

Nested CV는 Cross Validation을 두 단계로 구성한다.

```text
Outer Cross Validation
          │
          ├── Outer Training
          │         │
          │         ▼
          │   Inner Cross Validation
          │         │
          │         ▼
          │   Hyperparameter Tuning
          │         │
          │         ▼
          │   Model Refit
          │
          └── Outer Validation
                    │
                    ▼
              Performance Evaluation
```

Inner CV는 Hyperparameter를 선택한다.

Outer CV는 Hyperparameter Tuning 과정까지 포함한 전체 모델 선택 절차의 일반화 성능을 추정한다.

따라서 Nested CV는 특히 Dataset이 작고 Hyperparameter Tuning에 따른 Selection Bias가 중요한 경우에 고려할 수 있다.

다만 계산 비용이 커진다는 단점이 있다.

---

## 20. Bootstrap Validation과의 차이

참고 자료에서는 Cross Validation과 함께 Bootstrap Sampling Validation도 소개한다.

Bootstrap Sampling은 **Sampling with Replacement(복원추출)** 을 사용하는 방법이다.

즉 Sample을 추출한 뒤 원래 Dataset에 다시 포함시킨 상태에서 다음 Sample을 추출한다.

따라서 하나의 Bootstrap Sample에는 동일한 Data Point가 여러 번 포함될 수 있다.

반면 K-Fold에서는 기본적으로 Fold가 서로 겹치지 않으며, 하나의 Sample이 Validation에 정확히 한 번 사용된다.

| 구분 | K-Fold | Bootstrap |
| --- | --- | --- |
| Sampling 방식 | Fold 분할 | 복원추출 |
| 중복 Sample | 하나의 Fold 분할 내 중복 없음 | 가능 |
| 주요 활용 | Model Validation | 불확실성 추정, 재표본화, Bagging 등 |
| 평가 데이터 | Validation Fold | Out-of-Bag Sample 등을 활용 가능 |

Bootstrap에서 선택되지 않은 Sample을 **Out-of-Bag(OOB) Sample**이라고 한다.

OOB Sample을 이용하여 별도의 Evaluation을 수행할 수도 있다.

다만 Bootstrap Sampling과 OOB Evaluation은 K-Fold Cross Validation과 동일한 절차는 아니다.

---

## 21. scikit-learn에서 Cross Validation 구현

Cross Validation의 개념을 이해한 뒤 실제 구현을 살펴보면 전체 과정이 더 명확해진다.

scikit-learn에서는 다음과 같은 도구를 제공한다.

| 기능 | 역할 |
| --- | --- |
| `KFold` | 기본 K-Fold 분할 |
| `StratifiedKFold` | Class Distribution을 고려한 분할 |
| `GroupKFold` | Group 단위 분할 |
| `ShuffleSplit` | 반복적인 Random Split |
| `StratifiedShuffleSplit` | Stratified Random Split |
| `LeaveOneOut` | LOOCV |
| `cross_val_score` | Cross Validation Score 계산 |
| `cross_validate` | 여러 Metric 및 시간 정보 계산 |

---

### 21.1 KFold 예제

```python
from sklearn.model_selection import KFold

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for train_idx, valid_idx in kfold.split(X):

    X_train = X[train_idx]
    X_valid = X[valid_idx]
```

`n_splits=5`는 5-Fold Cross Validation을 의미한다.

`shuffle=True`는 Fold를 만들기 전에 데이터를 섞도록 설정한다.

단, 시간적인 순서나 그룹 의존성을 보존해야 하는 Dataset에서는 무작위 Shuffle이 적절하지 않을 수 있다.

---

### 21.2 StratifiedKFold 예제

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for train_idx, valid_idx in skf.split(X, y):

    X_train = X[train_idx]
    X_valid = X[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]
```

여기서 `y`는 Class Label이다.

StratifiedKFold는 Class Distribution을 고려하여 Fold를 분할한다.

scikit-learn에서는 `split()`이 반환하는 두 번째 Index를 `test_idx`라고 표기하기도 한다.

하지만 별도의 최종 Test Set을 유지하는 Model Development에서는 이 Index가 실제로 Cross Validation의 **Validation Fold**에 해당한다는 점을 구분해야 한다.

---

### 21.3 cross_val_score 예제

`cross_val_score()`를 사용하면 K개의 Fold를 직접 반복하는 코드를 작성하지 않고도 Cross Validation을 수행할 수 있다.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_iris(
    return_X_y=True
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("Fold Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
```

이 코드에서 `Pipeline`을 사용했기 때문에 각 Fold에서 StandardScaler는 Training Fold의 데이터만 이용하여 학습된다.

`cross_val_score()`는 Fold마다 모델을 별도로 학습하고 Validation Score를 반환한다.

따라서 `scores`에는 5개의 Accuracy가 저장된다.

---

### 21.4 cross_validate와 차이

`cross_validate()`는 여러 평가 지표와 Training Time 등의 정보를 얻고 싶을 때 사용할 수 있다.

예를 들어 다음과 같다.

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=[
        "accuracy",
        "f1_macro"
    ],
    return_train_score=True
)

print(
    results["test_accuracy"]
)

print(
    results["test_f1_macro"]
)
```

이때 `test_accuracy`라는 Key는 각 Fold의 Validation Score를 의미한다.

따라서 별도의 최종 Test Set에 대한 Performance와 혼동하지 않아야 한다.

---

## 22. Cross Validation의 장점과 한계

### 22.1 장점

**여러 데이터 분할에서 모델 평가**

한 번의 Hold-out 분할에만 의존하지 않고 여러 분할에서 모델 성능을 관찰할 수 있다.

**제한된 데이터를 반복적으로 활용**

각 Sample을 서로 다른 반복에서 Training과 Validation에 사용할 수 있다.

**Hyperparameter Tuning에 활용**

여러 Hyperparameter Configuration을 비교하는 데 사용할 수 있다.

**Model Stability 확인**

Fold별 Score의 변동을 살펴보면서 데이터 분할에 따른 Performance Variation을 관찰할 수 있다.

---

### 22.2 한계

**계산 비용 증가**

K-Fold에서는 K번의 Model Training이 필요하다.

**완전한 독립 평가가 아님**

각 Fold의 Training Set은 서로 중복되므로 Fold Score가 완전히 독립적인 것은 아니다.

**데이터 의존성을 고려해야 함**

환자, 장비, Session, 시간 등 데이터의 의존 구조에 따라 적절한 분할 방법을 선택해야 한다.

**Data Leakage 가능성**

Preprocessing이나 Hyperparameter Tuning 과정에서 Validation 정보가 학습에 유입될 수 있다.

**높은 CV Score가 최종 성능을 보장하지 않음**

Cross Validation 역시 주어진 데이터와 평가 설정에서 계산한 Performance Estimate일 뿐이다.

따라서 새로운 환경에서의 성능을 완전히 보장할 수는 없다.

---

## 23. Cross Validation 방법 선택 기준

| 데이터 특성 | 고려할 방법 |
| --- | --- |
| 일반적인 Regression Dataset | K-Fold |
| 일반적인 Classification Dataset | Stratified K-Fold |
| Class Imbalance가 존재하는 Classification | Stratified K-Fold |
| 동일 환자·사용자의 반복 측정 | Group K-Fold |
| Group 분리와 Class 비율 유지가 모두 중요 | Stratified Group K-Fold |
| 시간 순서가 중요한 데이터 | Time Series Split |
| 매우 작은 Dataset에서 LOOCV가 적합한 경우 | Leave-One-Out |
| 반복적인 Random Split 비교 | Shuffle Split |
| Hyperparameter Tuning까지 포함한 성능 추정 | Nested Cross Validation |

중요한 것은 Cross Validation 방법을 단순히 데이터의 양만으로 결정하지 않는 것이다.

**실제 모델이 적용될 환경에서 어떤 데이터가 새로운 데이터로 간주되는지** 먼저 정의해야 한다.

예를 들어 새로운 환자에게 사용할 모델이라면 같은 환자의 데이터가 Training과 Validation에 동시에 포함되지 않도록 평가하는 것이 중요할 수 있다.

반면 미래 시점을 예측하는 모델이라면 시간 순서를 보존해야 한다.

따라서 Cross Validation의 설계는 Model Architecture뿐 아니라 최종 예측 목표와도 연결된다.

---

## 24. 핵심 개념 비교

| 개념 | 의미 |
| --- | --- |
| Generalization | 새로운 데이터에 대한 예측 성능 |
| Training Set | Model Parameter 학습 |
| Validation Set | Model Selection과 Hyperparameter Tuning |
| Test Set | 최종 Generalization Performance 평가 |
| Hold-out Validation | 한 번의 데이터 분할을 이용한 평가 |
| Cross Validation | 여러 데이터 분할을 활용한 반복 평가 |
| K-Fold | K개의 Fold를 번갈아 Validation에 사용하는 방법 |
| Stratified K-Fold | Class Distribution을 고려한 K-Fold |
| Shuffle Split | 반복마다 무작위로 Training·Validation 분할 |
| Group K-Fold | 동일 Group이 Training과 Validation에 동시에 포함되지 않도록 분할 |
| Time Series Split | 시간 순서를 고려한 Validation |
| LOOCV | Sample 하나씩 Validation에 사용하는 방법 |
| Nested CV | Model Selection과 Performance Estimation을 분리하는 이중 CV |
| Data Leakage | 평가 데이터의 정보가 학습 과정에 유입되는 문제 |

---

## 25. 최종 정리

Cross Validation은 Machine Learning Model의 Generalization Performance를 추정하기 위해 데이터를 여러 부분으로 나누고 반복적으로 Training과 Validation을 수행하는 방법이다.

가장 기본적인 Hold-out Validation은 Dataset을 한 번만 분할하여 모델을 평가한다.

그러나 데이터의 양이 적거나 어떤 Sample이 Validation Set에 포함되는지에 따라 평가 결과의 변동이 큰 경우에는 한 번의 분할만으로 모델 성능을 판단하기 어려울 수 있다.

이를 보완하기 위한 대표적인 방법이 K-Fold Cross Validation이다.

K-Fold Cross Validation에서는 Dataset을 K개의 Fold로 나누고 각 Fold를 한 번씩 Validation Set으로 사용한다.

따라서 총 K번의 Model Training과 Evaluation을 수행하며, 각 Sample은 Validation에 정확히 한 번 참여한다.

이 과정에서 계산된 K개의 Validation Score를 평균하여 모델의 성능을 요약할 수 있다.

하지만 Cross Validation을 사용한다고 해서 모든 평가 문제가 해결되는 것은 아니다.

데이터에 Class Imbalance가 존재한다면 Stratified K-Fold를 고려할 수 있고, 동일 환자나 사용자처럼 서로 연관된 Sample이 존재한다면 Group K-Fold를 사용할 수 있다.

또한 시간 순서가 중요한 Dataset에서는 Time Series Split과 같은 방법이 필요하다.

Hyperparameter Tuning에서는 Cross Validation을 이용하여 여러 Model Configuration을 비교하고, 적절한 설정을 선택할 수 있다.

이후 선택한 설정으로 Development Set 전체를 사용하여 모델을 다시 학습하고, 별도로 보관한 Test Set에서 최종 평가를 수행하는 것이 일반적인 절차이다.

전체 학습 흐름은 다음과 같이 정리할 수 있다.

```text
Original Dataset
       │
       ▼
Independent Test Set 확보
       │
       ▼
Development Dataset
       │
       ▼
Cross Validation
       │
       ├── K-Fold
       ├── Stratified K-Fold
       ├── Group K-Fold
       └── Time Series Split
       │
       ▼
Validation Score Comparison
       │
       ▼
Hyperparameter Selection
       │
       ▼
Final Model Training
       │
       ▼
Independent Test Evaluation
       │
       ▼
Generalization Performance
```

따라서 Cross Validation의 본질은 단순히 데이터를 여러 번 나누는 것이 아니다.

**학습 데이터와 평가 데이터의 역할을 구분하면서, 여러 데이터 구성에 대해 모델의 예측 성능을 확인하여 Generalization Performance를 보다 신뢰성 있게 추정하는 것**이다.

---

### 핵심 요약

> Cross Validation은 데이터를 여러 부분으로 나누어 Training과 Validation을 반복하는 모델 평가 방법이다. K-Fold Cross Validation에서는 K개의 Fold가 각각 한 번씩 Validation Set으로 사용되며, K번의 독립적인 모델 학습을 통해 여러 Validation Score를 얻는다. 이를 이용하여 모델 성능을 추정하고 Hyperparameter를 비교할 수 있다. 다만 데이터의 Class Distribution, Group Structure, Time Dependency 및 Data Leakage를 고려하여 적절한 Validation Strategy를 선택해야 한다.

---

## 참고 자료

### 주요 학습 자료

- [Cross Validation — Dsaint31](https://dsaint31.me/mkdocs_site/ML/ch01/ch01_52/)

### scikit-learn 공식 문서

- [Cross-validation: Evaluating Estimator Performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [Common Pitfalls and Recommended Practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Nested versus Non-nested Cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
