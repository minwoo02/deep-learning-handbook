# Regularization 완전 정리: Overfitting·L1/L2부터 Dropout·Early Stopping·Data Augmentation까지

[TOC]


관련 실습: [AdamW 완전 정리: Adam의 Moment Estimation부터 L2 Regularization·Weight Decay와 PyTorch 구현까지](../../practice/tech-optimizers/adamw.md)


---

Machine Learning과 Deep Learning에서 모델을 학습하는 목적은 단순히 Training Dataset의 Loss를 최소화하는 것이 아니다.

궁극적인 목표는 **학습에 사용하지 않은 새로운 데이터에서도 좋은 예측 성능을 나타내는 것**이다.

이를 Generalization(일반화)이라고 한다.

그러나 Model Complexity가 지나치게 높거나 Training Data에 존재하는 Noise까지 학습하면, Training Performance는 높지만 새로운 데이터에서는 성능이 떨어지는 현상이 발생할 수 있다.

이러한 현상을 Overfitting(과적합)이라고 한다.

Overfitting을 완화하고 모델의 Generalization Performance를 개선하기 위해 사용하는 대표적인 방법이 **Regularization(정규화 또는 정칙화)** 이다.

Regularization은 단순히 Weight를 작게 만드는 것만을 의미하지 않는다. Model Complexity를 제어하거나, 학습 데이터에 다양성을 부여하거나, 학습 과정에 제약을 적용하는 등 다양한 방법이 포함된다.

이번 글에서는 다음 내용을 살펴본다.

- Overfitting과 Generalization
- Regularization의 수학적 의미
- L1·L2 Regularization과 Elastic Net
- Early Stopping
- Dropout
- Data Augmentation
- Batch Normalization의 간접적인 Regularization 효과
- Weight Constraint와 Noise Injection
- Label Smoothing과 Soft Labeling
- Regularization Strength와 Hyperparameter Tuning

---

### 1. Regularization이란?

Regularization은 모델이 Training Data에 지나치게 적합되는 것을 방지하고, 새로운 데이터에 대한 일반화 성능을 개선하기 위해 사용하는 기법이다.

일반적인 Machine Learning Model은 다음 Loss Function을 최소화한다.

$$
\boxed{ \theta^*= \arg\min_{\theta} \mathcal{L}_{data}(\theta) }
$$

여기서

- $\theta$: Model Parameter
- $\mathcal{L}_{data}$: 데이터에 대한 Loss Function
- $\theta^*$: 최적화된 Parameter

이다.

그러나 Training Loss만 최소화하면 모델이 Training Dataset의 세부적인 패턴이나 Noise까지 학습할 수 있다.

이를 방지하기 위해 Parameter의 크기나 모델의 학습 방식에 제약을 추가할 수 있다.

대표적인 형태는 다음과 같다.

$$
\boxed{ \mathcal{L}_{total}(\theta) = \mathcal{L}_{data}(\theta) + \lambda\mathcal{R}(\theta) }
$$

여기서

- $\mathcal{L}_{total}$: Regularization을 포함한 전체 Objective
- $\mathcal{L}_{data}$: 원래의 Data Loss
- $\mathcal{R}(\theta)$: Regularization Term
- $\lambda$: Regularization Strength

이다.

이 식의 핵심은 모델이 Training Data에 잘 맞으면서도 Parameter가 지나치게 복잡해지지 않도록 균형을 맞추는 것이다.

다만 모든 Regularization 기법이 실제로 Loss Function에 Penalty Term을 추가하는 것은 아니다.

예를 들어 Dropout은 학습 중 일부 Neuron의 출력을 무작위로 제거하고, Early Stopping은 학습을 조기에 중단하는 방식으로 작동한다.

따라서 Regularization을 **모델의 과도한 적합을 제한하기 위한 여러 전략의 총칭**으로 이해하는 것이 적절하다.

---

### 2. Overfitting과 Underfitting

Regularization을 이해하기 전에 Overfitting과 Underfitting의 차이를 살펴볼 필요가 있다.

#### 2.1 Underfitting

Underfitting(과소적합)은 모델이 Training Data의 주요 패턴조차 충분히 학습하지 못한 상태이다.

일반적으로 다음과 같은 특성이 나타날 수 있다.

- Training Loss가 높음
- Validation Loss도 높음
- Model Capacity가 부족하거나 Optimization이 충분하지 않을 수 있음

예를 들어 비선형적인 데이터를 Linear Model로 학습한다면 데이터의 복잡한 관계를 충분히 표현하지 못할 수 있다.

#### 2.2 Overfitting

Overfitting은 모델이 Training Data에 지나치게 적합된 상태이다.

일반적으로 다음과 같은 특성이 나타날 수 있다.

- Training Loss는 매우 낮음
- Validation Loss는 상대적으로 높음
- 새로운 데이터에 대한 Generalization Performance가 낮음

예를 들어 Training Dataset에 존재하는 Noise까지 학습하면, 학습 데이터에서는 높은 Accuracy를 보이더라도 새로운 데이터에서는 성능이 크게 저하될 수 있다.

#### 2.3 적절한 학습 상태

적절하게 학습된 모델은 Training Dataset의 중요한 패턴을 학습하면서도 새로운 데이터에서 좋은 성능을 나타내야 한다.

| 구분 | Training Performance | Validation Performance |
| --- | --- | --- |
| Underfitting | 낮음 | 낮음 |
| 적절한 학습 | 높음 | 높음 |
| Overfitting | 매우 높음 | 상대적으로 낮음 |

다만 실제 학습에서는 Training과 Validation 성능의 차이만으로 모든 문제를 판단할 수 없다.

Dataset의 구성, Noise, Model Capacity, Optimization 상태 등을 함께 확인해야 한다.

---

### 3. Regularization과 Bias-Variance Trade-off

Regularization은 일반적으로 Bias-Variance Trade-off와 연결하여 이해할 수 있다.

**Bias**는 모델이 데이터의 실제 관계를 충분히 표현하지 못하면서 발생하는 체계적인 오차와 관련이 있다.

**Variance**는 Training Dataset의 변화에 따라 학습 결과가 크게 달라지는 민감도와 관련이 있다.

모델이 지나치게 복잡하면 Training Data의 작은 변화에도 예측 결과가 크게 달라질 수 있다.

반대로 모델을 지나치게 단순하게 제한하면 중요한 패턴까지 학습하지 못할 수 있다.

Regularization은 이러한 균형을 조절하는 방법 중 하나이다.

```text
Regularization Strength 증가
             │
             ▼
Model Flexibility 제한
             │
             ▼
복잡한 패턴 및 Noise에 대한 적합 감소
             │
             ▼
Overfitting 완화 가능
```

그러나 Regularization Strength가 너무 크면 모델이 충분한 패턴을 학습하지 못해 Underfitting이 발생할 수 있다.

따라서 Regularization은 강하게 적용할수록 항상 좋은 것이 아니다.

**Training Data에 대한 적합도와 새로운 데이터에 대한 Generalization Performance 사이에서 적절한 균형을 찾는 것이 중요하다.**

---

## Part 1. Parameter에 Penalty를 적용하는 Regularization

Regularization의 대표적인 방법은 Model Parameter의 크기에 Penalty를 부여하여 지나치게 큰 Weight가 학습되는 것을 억제하는 것이다.

이러한 방법을 이해하려면 먼저 **Norm(노름)** 이라는 수학적 개념을 알아야 한다.

Norm은 Vector의 크기를 나타내는 개념이며, L1·L2 Regularization은 각각 서로 다른 Norm을 이용하여 Parameter에 Penalty를 부여한다.

---

### 4. Norm(노름)이란?

**Norm(노름)** 은 Vector의 크기(Length 또는 Magnitude)를 나타내는 수학적 개념이다.

일반적으로 Vector가 원점으로부터 얼마나 떨어져 있는지를 측정하는 데 사용된다.

예를 들어 3차원 Vector가 다음과 같다고 하자.

$$
\mathbf{a}=(a_1,a_2,a_3)
$$

이 Vector의 Euclidean Norm은 다음과 같다.

$$
\boxed{ \|\mathbf{a}\|_2 = \sqrt{a_1^2+a_2^2+a_3^2} }
$$

이는 3차원 공간에서 Vector의 일반적인 길이를 의미한다.

#### 4.1 일반적인 Lp Norm

$n$차원 Vector에 대한 Lp Norm은 다음과 같이 정의한다.

$$
\boxed{ \|\mathbf{a}\|_p = \left( \sum_{i=1}^{n}|a_i|^p \right)^{1/p} }
$$

여기서

- $\mathbf{a}$: Vector
- $a_i$: Vector의 i번째 Component
- $p$: Norm의 차수를 결정하는 값
- $p\geq1$: Norm을 만족하는 범위

이다.

대표적인 Norm은 다음과 같다.

| 종류 | 정의 | 특징 |
| --- | --- | --- |
| L1 Norm | $\displaystyle|\mathbf{a}|_1=\sum_i | a_i |
| L2 Norm | $\displaystyle\|\mathbf{a}\|_2=\sqrt{\sum_i a_i^2}$ | Euclidean Length |
| L∞ Norm | $\displaystyle|\mathbf{a}|_\infty=\max_i | a_i |

예를 들어 다음 Vector를 생각해 보자.

$$
\boxed{ \mathbf{a}=(3,4) }
$$

**L1 Norm**

$$
\|\mathbf{a}\|_1 = |3|+|4|
$$

$$
\boxed{ \|\mathbf{a}\|_1=7 }
$$

**L2 Norm**

$$
\|\mathbf{a}\|_2 = \sqrt{3^2+4^2}
$$

$$
\boxed{ \|\mathbf{a}\|_2=5 }
$$

**L∞ Norm**

$$
\|\mathbf{a}\|_\infty = \max(|3|,|4|)
$$

$$
\boxed{ \|\mathbf{a}\|_\infty=4 }
$$

동일한 Vector라도 어떤 Norm을 사용하는지에 따라 계산되는 크기가 달라진다.

---

#### 4.2 Norm의 기본 성질

Norm은 다음 세 가지 성질을 만족해야 한다.

**① Non-negativity & Definiteness**

$$
\boxed{ \|\mathbf{a}\|_p\geq0 }
$$

$$
\boxed{ \|\mathbf{a}\|_p=0 \iff \mathbf{a}=\mathbf{0} }
$$

Norm은 항상 0 이상이며, Zero Vector일 때만 0이 된다.

**② Absolute Homogeneity(절대 동차성)**

$$
\boxed{ \|c\mathbf{a}\|_p = |c|\|\mathbf{a}\|_p }
$$

여기서 $c$는 Scalar이다.

Vector에 상수를 곱하면 Norm은 해당 상수의 절댓값만큼 Scaling된다.

**③ Triangle Inequality(삼각부등식)**

$$
\boxed{ \|\mathbf{a}+\mathbf{b}\|_p \leq \|\mathbf{a}\|_p+\|\mathbf{b}\|_p }
$$

두 Vector의 합에 대한 Norm은 각 Vector의 Norm을 더한 값보다 클 수 없다.

이 세 가지 성질을 만족하는 함수가 Norm이다.

---

#### 4.3 Euclidean Norm과 Inner Product

L2 Norm은 Inner Product(내적)로도 표현할 수 있다.

$$
\boxed{ \|\mathbf{a}\|_2 = \sqrt{\mathbf{a}\cdot\mathbf{a}} }
$$

Matrix Notation을 사용하면 다음과 같다.

$$
\boxed{ \|\mathbf{a}\|_2 = \sqrt{\mathbf{a}^{T}\mathbf{a}} }
$$

또한 Inner Product와 L2 Norm 사이에는 Cauchy–Schwarz Inequality가 성립한다.

$$
\boxed{ |\mathbf{a}\cdot\mathbf{b}| \leq \|\mathbf{a}\|_2\|\mathbf{b}\|_2 }
$$

이는 두 Vector의 Inner Product가 각 Vector의 크기와 어떤 관계를 가지는지 나타내는 중요한 부등식이다.

단, Cauchy–Schwarz Inequality는 Norm의 정의를 이루는 세 가지 공리가 아니라, Inner Product와 Euclidean Norm 사이에서 성립하는 별도의 성질이다.

---

#### 4.4 Norm과 Regularization의 관계

Machine Learning에서는 Model Parameter를 Vector로 표현할 수 있다.

예를 들어 다음과 같이 정의하자.

$$
\boxed{ \mathbf{w}=[w_1,w_2,\ldots,w_n]^T }
$$

이때 Weight Vector의 L1 Norm은 다음과 같다.

$$
\boxed{ \|\mathbf{w}\|_1 = \sum_{i=1}^{n}|w_i| }
$$

L2 Norm은 다음과 같다.

$$
\boxed{ \|\mathbf{w}\|_2 = \sqrt{\sum_{i=1}^{n}w_i^2} }
$$

여기서 주의할 점이 있다.

**일반적으로 L2 Regularization에서 사용하는 Penalty는 L2 Norm 자체가 아니라 L2 Norm의 제곱이다.**

$$
\boxed{ \|\mathbf{w}\|_2^2 = \sum_{i=1}^{n}w_i^2 }
$$

따라서 대표적인 Regularization을 다음과 같이 표현할 수 있다.

**L1 Regularization**

$$
\boxed{ \mathcal{L}_{L1} = \mathcal{L}_{data} + \lambda\|\mathbf{w}\|_1 }
$$

**L2 Regularization**

$$
\boxed{ \mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda\|\mathbf{w}\|_2^2 }
$$

두 방법은 모두 Weight에 Penalty를 적용하지만, 사용하는 Norm의 형태가 다르기 때문에 학습 결과에도 차이가 발생한다.

---

### 5. L1 Regularization

L1 Regularization은 Weight의 절댓값 합을 Loss Function에 추가하는 방법이다.

$$
\boxed{ \mathcal{L}_{L1} = \mathcal{L}_{data} + \lambda\sum_{i=1}^{n}|w_i| }
$$

또는 L1 Norm을 사용하여 다음과 같이 표현할 수 있다.

$$
\boxed{ \mathcal{L}_{L1} = \mathcal{L}_{data} + \lambda\|\mathbf{w}\|_1 }
$$

여기서

- $w_i$: i번째 Weight
- $\lambda$: Regularization Strength
- $n$: Regularization을 적용하는 Weight 수

이다.

L1 Regularization은 Weight의 절댓값 합이 커질수록 더 큰 Penalty를 부여한다.

따라서 Optimization 과정에서 Weight를 축소시키는 방향으로 작용한다.

특히 L1 Regularization은 **일부 Weight를 정확히 0으로 만드는 Sparse Solution을 유도할 수 있다**는 특징이 있다.

---

#### 5.1 Sparsity란?

Sparsity(희소성)는 Vector의 많은 Component가 0인 특성을 의미한다.

예를 들어 다음 Weight Vector가 있다고 하자.

$$
\mathbf{w} = [0.8,-0.4,0.03,0.01,0.5]
$$

L1 Regularization을 적용한 후 다음과 같은 Solution을 얻을 수 있다.

$$
\mathbf{w} = [0.7,-0.3,0,0,0.4]
$$

이는 L1의 작동 경향을 보여주기 위한 예시이다.

일부 Weight가 0이 되면 해당 Feature의 기여가 제거될 수 있다.

따라서 L1 Regularization은 Linear Model에서 Feature Selection 효과를 제공할 수 있다.

---

#### 5.2 L1 Regularization은 왜 Weight를 0으로 만드는가?

L1 Penalty는 다음과 같다.

$$
\boxed{ \mathcal{R}_{L1}(w)=|w| }
$$

이 함수는 $w=0$에서 미분 불가능한 지점을 가진다.

반면 $w\neq0$에서는 다음과 같이 미분된다.

$$
\frac{d|w|}{dw} = \begin{cases} +1,&w>0\\ -1,&w<0 \end{cases}
$$

즉 Weight가 양수이면 감소시키는 방향으로, 음수이면 증가시키는 방향으로 Penalty Gradient가 작용한다.

이러한 특성과 $w=0$에서의 비매끄러운 구조 때문에 L1 Regularization은 특정 최적화 문제에서 Weight가 정확히 0이 되는 Solution을 유도할 수 있다.

다만 모든 Weight가 반드시 0으로 수렴하는 것은 아니며, 실제 결과는 Data Loss와 Regularization Strength 등에 따라 달라진다.

---

#### 5.3 Lasso Regression

Linear Regression에서 L1 Regularization을 적용한 대표적인 모델이 Lasso Regression이다.

Lasso는 다음의 약자이다.

**Least Absolute Shrinkage and Selection Operator**

이름에서 알 수 있듯이 Lasso는 다음 두 가지 특징을 가진다.

- Parameter Shrinkage
- Feature Selection

즉 L1 Regularization의 핵심은 다음과 같다.

> L1 Regularization은 Weight의 절댓값 합을 제한하여 일부 Parameter를 0으로 만들 수 있으며, 이를 통해 Sparse Model을 유도한다.

---

### 6. L2 Regularization

L2 Regularization은 Weight의 제곱합을 Loss Function에 추가하는 방법이다.

$$
\boxed{ \mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda\sum_{i=1}^{n}w_i^2 }
$$

이를 L2 Norm으로 표현하면 다음과 같다.

$$
\boxed{ \mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda\|\mathbf{w}\|_2^2 }
$$

L2 Regularization은 큰 Weight에 상대적으로 더 큰 Penalty를 부여한다.

예를 들어 다음과 같은 Weight가 있다고 하자.

$$
w_1=1,\qquad w_2=3
$$

각 Weight의 제곱은 다음과 같다.

$$
w_1^2=1
$$

$$
w_2^2=9
$$

즉 Weight의 절댓값이 3배 커지면 L2 Penalty에 대한 해당 Weight의 기여는 9배가 된다.

이러한 특성으로 인해 L2 Regularization은 지나치게 큰 Weight가 발생하는 것을 억제하는 방향으로 작용한다.

---

#### 6.1 L2 Regularization의 Gradient

L2 Regularization이 실제 Parameter Update에 어떤 영향을 미치는지 살펴보자.

다음 Loss Function을 생각하자.

$$
\boxed{ \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda\sum_iw_i^2 }
$$

특정 Weight $w_i$에 대해 미분하면 다음과 같다.

$$
\boxed{ \frac{\partial\mathcal{L}_{total}}{\partial w_i} = \frac{\partial\mathcal{L}_{data}}{\partial w_i} + 2\lambda w_i }
$$

여기서

$$
2\lambda w_i
$$

가 Regularization Gradient이다.

Gradient Descent의 기본 Update Equation은 다음과 같다.

$$
\boxed{ w_i^{(t+1)} = w_i^{(t)} - \eta \frac{\partial\mathcal{L}_{total}}{\partial w_i} }
$$

여기에 L2 Gradient를 대입하면

$$
w_i^{(t+1)} = w_i^{(t)} - \eta \left( \frac{\partial\mathcal{L}_{data}}{\partial w_i} + 2\lambda w_i^{(t)} \right)
$$

이다.

전개하면

$$
w_i^{(t+1)} = w_i^{(t)} - \eta\frac{\partial\mathcal{L}_{data}}{\partial w_i} - 2\eta\lambda w_i^{(t)}
$$

이다.

따라서

$$
\boxed{ w_i^{(t+1)} = (1-2\eta\lambda)w_i^{(t)} - \eta \frac{\partial\mathcal{L}_{data}}{\partial w_i} }
$$

로 정리할 수 있다.

여기서

$$
1-2\eta\lambda
$$

는 일반적인 작은 양의 Learning Rate와 Regularization Strength 조건에서 Weight를 축소시키는 요소로 작용한다.

참고로 L2 Penalty를 다음과 같이 정의하는 경우도 많다.

$$
\frac{\lambda}{2}\|\mathbf{w}\|_2^2
$$

이 경우 Regularization Gradient는 $\lambda w_i$가 된다.

이는 계수 정의의 차이이며, L2 Regularization의 기본적인 의미는 동일하다.

---

#### 6.2 L2 Regularization 예제

간단한 Loss Function을 생각해 보자.

$$
\boxed{ \mathcal{L}_{data}(w)=(w-4)^2 }
$$

Regularization이 없는 경우 최적 Weight는 다음과 같다.

$$
\boxed{ w^*=4 }
$$

이제 L2 Regularization을 적용한다.

$$
\boxed{ \mathcal{L}_{total}(w) = (w-4)^2+\lambda w^2 }
$$

이를 미분하면

$$
\frac{d\mathcal{L}_{total}}{dw} = 2(w-4)+2\lambda w
$$

이다.

최적점에서 Gradient가 0이라는 조건을 적용하면

$$
2(w-4)+2\lambda w=0
$$

따라서

$$
w(1+\lambda)=4
$$

이고

$$
\boxed{ w^*=\frac4{1+\lambda} }
$$

를 얻는다.

Regularization Strength에 따라 Weight는 다음과 같이 변화한다.

| $\lambda$ | 최적 Weight |
| --- | --- |
| 0 | 4 |
| 1 | 2 |
| 3 | 1 |
| 9 | 0.4 |

$\lambda$가 증가할수록 최적 Weight는 0에 가까워진다.

이 예제는 L2 Regularization의 Parameter Shrinkage 효과를 보여준다.

하지만 Weight가 작아졌다고 Generalization Performance가 반드시 개선되는 것은 아니다.

---

#### 6.3 Ridge Regression

Linear Regression에서 L2 Regularization을 적용한 대표적인 모델이 Ridge Regression이다.

Ridge Regression은 Weight의 크기를 제한하여 지나치게 큰 Regression Coefficient가 발생하는 것을 억제한다.

특히 Feature 사이의 상관관계가 높은 상황에서 계수 추정의 불안정성을 완화하는 데 도움이 될 수 있다.

L1 Regularization이 일부 Weight를 정확히 0으로 만드는 경향이 있다면, L2 Regularization은 여러 Weight를 작게 유지하는 경향이 있다는 점에서 차이가 있다.

---

### 7. L1과 L2 Regularization 비교

L1과 L2는 모두 Weight의 크기에 Penalty를 부여하지만, 사용하는 Norm과 선호하는 Solution의 특성이 다르다.

| 구분 | L1 Regularization | L2 Regularization |
| --- | --- | --- |
| Penalty | $\|\mathbf{w}\|_1$ | $\|\mathbf{w}\|_2^2$ |
| 수식 | $\sum_i | w_i |
| Weight 변화 | 일부 Weight가 정확히 0이 될 수 있음 | 여러 Weight를 연속적으로 축소 |
| Sparsity | 유도 가능 | 일반적으로 정확한 0은 잘 만들지 않음 |
| 대표 모델 | Lasso Regression | Ridge Regression |
| 주요 특징 | Feature Selection | Weight Shrinkage 및 계수 안정화 |

L1 Regularization은 Sparse Solution을 선호할 수 있다.

반면 L2 Regularization은 여러 Weight를 작은 값으로 유지하는 경향이 있다.

이러한 차이는 각 Penalty의 수학적 형태에서 발생한다.

**L1 Penalty**

$$
\boxed{ \mathcal{R}_{L1}(\mathbf{w}) = \|\mathbf{w}\|_1 }
$$

**L2 Penalty**

$$
\boxed{ \mathcal{R}_{L2}(\mathbf{w}) = \|\mathbf{w}\|_2^2 }
$$

즉 L1과 L2의 차이는 단순히 계산 방법이 다른 것이 아니라, **Optimization 과정에서 어떤 Parameter Solution을 선호하는지에도 영향을 준다는 것**이다.

---

### 8. Elastic Net

Elastic Net은 L1 Regularization과 L2 Regularization을 결합한 방법이다.

일반적으로 다음과 같이 표현할 수 있다.

$$
\boxed{ \mathcal{L}_{ElasticNet} = \mathcal{L}_{data} + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2 }
$$

이를 전개하면 다음과 같다.

$$
\boxed{ \mathcal{L}_{ElasticNet} = \mathcal{L}_{data} + \lambda_1\sum_i|w_i| + \lambda_2\sum_iw_i^2 }
$$

여기서

- $\lambda_1$: L1 Regularization Strength
- $\lambda_2$: L2 Regularization Strength

이다.

Elastic Net은 두 Regularization의 특성을 함께 활용한다.

L1에 의한 Sparse Solution을 유도하면서, L2에 의한 Weight Shrinkage 효과도 적용할 수 있다.

특히 서로 높은 상관관계를 가지는 Feature가 여러 개 존재하는 데이터에서 고려할 수 있다.

대표적으로 Linear Regression 계열에서 활용된다.

---

### 9. L1·L2 Regularization을 Bias에도 적용할까?

L1과 L2 Regularization은 일반적으로 Weight에 적용하고, Bias에는 적용하지 않는 경우가 많다.

Linear Model을 생각하면 다음과 같다.

$$
\boxed{ y=\mathbf{w}^{T}\mathbf{x}+b }
$$

여기서

- $\mathbf{w}$: Weight Vector
- $\mathbf{x}$: Input Feature Vector
- $b$: Bias

이다.

Weight는 각 Feature가 Prediction에 미치는 영향을 결정한다.

반면 Bias는 Prediction의 기준 위치를 조절한다.

Bias까지 강하게 제한하면 Model의 기준값 조정이 불필요하게 어려워질 수 있다.

따라서 L2 Regularization Term을 다음과 같이 Weight에 대해서만 구성할 수 있다.

$$
\boxed{ \mathcal{R}(\mathbf{w}) = \|\mathbf{w}\|_2^2 }
$$

전체 Loss Function은 다음과 같다.

$$
\boxed{ \mathcal{L}_{total} = \mathcal{L}_{data}(\mathbf{w},b) + \lambda\|\mathbf{w}\|_2^2 }
$$

이 경우 Data Loss는 Weight와 Bias 모두에 의존하지만, Regularization Penalty는 Weight에만 적용된다.

다만 Bias를 반드시 제외해야 하는 것은 아니다.

실제 적용 대상은 Model Architecture, Optimizer 및 Training Objective에 따라 달라질 수 있다.

---

#### Part 1 핵심 정리

Norm은 Vector의 크기를 측정하는 수학적 개념이며, L1·L2 Regularization은 이러한 Norm을 이용하여 Weight에 Penalty를 부여한다.

L1 Regularization은 Weight의 절댓값 합에 Penalty를 부여한다.

$$
\boxed{ \mathcal{L}_{L1} = \mathcal{L}_{data} + \lambda\|\mathbf{w}\|_1 }
$$

L2 Regularization은 Weight의 제곱합에 Penalty를 부여한다.

$$
\boxed{ \mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda\|\mathbf{w}\|_2^2 }
$$

L1은 일부 Weight를 0으로 만드는 Sparse Solution을 유도할 수 있으며, L2는 큰 Weight를 억제하고 여러 Weight를 작게 유지하는 경향이 있다.

Elastic Net은 두 방법을 결합하여 Sparse Solution과 Weight Shrinkage의 특성을 함께 활용한다.

전체적인 관계는 다음과 같다.

```text
Norm
  │
  ├── L1 Norm
  │      │
  │      ▼
  │  L1 Regularization
  │      │
  │      └── Lasso / Sparsity
  │
  └── L2 Norm
         │
         ▼
     Squared L2 Norm
         │
         ▼
     L2 Regularization
         │
         └── Ridge / Weight Shrinkage

L1 + L2
   │
   ▼
Elastic Net
```

### 핵심은 L1과 L2가 서로 다른 형태의 Penalty를 사용하며, 이 차이가 Optimization 과정에서 선호하는 Parameter의 형태에도 영향을 준다는 것이다.

## Part 2. Training Process를 제어하는 Regularization

### 9. Early Stopping

Early Stopping은 Validation Performance가 더 이상 개선되지 않을 때 Training을 중단하는 방법이다.

Neural Network는 Epoch가 증가할수록 Training Loss가 계속 감소할 수 있다.

하지만 Validation Loss도 계속 감소하는 것은 아니다.

어느 시점부터는 Training Loss는 감소하지만 Validation Loss가 다시 증가할 수 있다.

이는 Overfitting의 징후일 수 있다.

```text
Training 진행
      │
      ▼
Training Loss 감소
      │
      ▼
Validation Loss 측정
      │
      ├── 개선됨
      │     └── Training 지속
      │
      └── 개선되지 않음
            │
            ▼
       Patience 확인
            │
            ▼
       Early Stopping
```

Early Stopping의 핵심은 단순히 Epoch 수를 줄이는 것이 아니다.

**Validation Performance가 좋은 시점을 선택하여 불필요한 추가 학습을 제한하는 것**이다.

---

#### 9.1 Patience

Early Stopping에서는 일반적으로 `patience`라는 Hyperparameter를 사용한다.

예를 들어

$$
patience=5
$$

라고 설정했다고 하자.

Validation Loss가 개선되지 않더라도 5번의 확인 기회를 허용하고, 그 이후에도 개선되지 않으면 Training을 중단할 수 있다.

실제 구현에서는 Minimum Improvement를 의미하는 `min_delta` 등을 함께 사용할 수도 있다.

또한 일반적으로 가장 좋은 Validation Performance를 나타낸 시점의 Model Parameter를 저장하거나 복원한다.

따라서 Early Stopping은 Training Duration과 Model Selection을 함께 제어하는 Regularization 방법으로 이해할 수 있다.

---

### 10. Dropout

Dropout은 Neural Network의 Training 과정에서 일부 Neuron의 출력을 무작위로 0으로 만드는 방법이다.

Dropout을 적용하면 모델이 특정 Neuron이나 Feature Representation에 지나치게 의존하지 않도록 학습을 유도할 수 있다.

예를 들어 다음과 같은 Hidden Layer가 있다고 하자.

```text
Hidden Layer

○ ○ ○ ○ ○ ○
```

Dropout을 적용하면 각 Training Step에서 일부 Neuron의 출력이 비활성화될 수 있다.

```text
Training Step A

○ × ○ ○ × ○

Training Step B

× ○ ○ × ○ ○

Training Step C

○ ○ × ○ ○ ×
```

여기서 `×`는 해당 Step에서 출력이 비활성화된 Neuron을 의미한다.

중요한 것은 Dropout이 Neuron을 영구적으로 제거하는 것이 아니라는 점이다.

다음 Training Step에서는 다른 Neuron이 비활성화될 수 있다.

---

#### 10.1 Dropout의 수학적 표현

Dropout Probability를 $p$라고 하자.

Neuron의 출력이 $h_i$라면 Dropout Mask를 다음과 같이 정의할 수 있다.

$$
m_i\sim\operatorname{Bernoulli}(1-p)
$$

여기서 $m_i$는 다음 값을 가진다.

$$
m_i= \begin{cases} 1,&\text{Neuron 유지}\\ 0,&\text{Neuron 비활성화} \end{cases}
$$

PyTorch에서 일반적으로 사용하는 Inverted Dropout에서는 Training 시 다음과 같이 Scaling한다.

$$
\boxed{ \tilde h_i = \frac{m_i h_i}{1-p} }
$$

이를 통해 Training 시 출력의 기대값을 유지한다.

예를 들어

$$
p=0.2
$$

라면 각 출력이 Training Step마다 20%의 확률로 0이 된다.

Dropout은 Neuron 간 Co-adaptation을 줄이는 데 도움이 될 수 있다.

여기서 Co-adaptation은 특정 Neuron들이 서로의 존재에 과도하게 의존하는 현상으로 이해할 수 있다.

Dropout은 이와 같은 의존성을 완화하여 모델이 더 다양한 Feature 조합을 활용하도록 유도한다.

---

#### 10.2 Training과 Inference의 차이

일반적인 Dropout은 Training 과정에서만 적용한다.

Training Mode에서는 Dropout Mask가 무작위로 생성된다.

반면 Evaluation 또는 Inference Mode에서는 일반적으로 Dropout을 비활성화한다.

```text
Training
    │
    ▼
Random Dropout 적용
    │
    ▼
Parameter Update


Inference
    │
    ▼
Dropout 비활성화
    │
    ▼
전체 Network를 이용한 Prediction
```

PyTorch에서는 `model.train()`과 `model.eval()`을 통해 이러한 동작을 구분할 수 있다.

다만 MC Dropout처럼 추론 시에도 Dropout을 사용하여 예측 불확실성을 추정하는 별도의 접근도 존재한다.

이는 일반적인 Dropout Inference와 구분해야 한다.

---

## Part 3. 데이터를 활용하는 Regularization

### 11. Data Augmentation

Data Augmentation(데이터 증강)은 기존 Training Data에 적절한 변환을 적용하여 학습 데이터의 다양성을 늘리는 방법이다.

Regularization은 Model Parameter를 직접 제한하는 방식뿐 아니라, 학습에 사용되는 데이터의 구성을 변화시키는 방법으로도 수행할 수 있다.

예를 들어 Image Classification에서는 다음과 같은 Augmentation을 활용할 수 있다.

- Rotation
- Random Crop
- Horizontal Flip
- Scaling
- Color Jitter

이러한 변환을 통해 모델이 특정 위치, 방향, 조명 조건 등에 지나치게 의존하지 않도록 유도할 수 있다.

---

#### 11.1 Data Augmentation이 Regularization인 이유

예를 들어 고양이 이미지가 항상 화면 중앙에만 존재하는 Dataset을 학습한다고 하자.

모델은 고양이의 형태뿐 아니라 화면 중앙에 객체가 존재한다는 특징까지 학습할 수 있다.

이 경우 새로운 이미지에서 고양이가 화면 가장자리에 위치하면 Prediction Performance가 떨어질 수 있다.

Random Crop이나 Translation 등을 적용하면 모델이 다양한 위치의 객체를 학습하도록 만들 수 있다.

즉 Data Augmentation의 핵심은 다음과 같다.

> Label의 의미를 유지하는 적절한 변환을 적용하여 모델이 학습 데이터의 우연한 특성에 지나치게 의존하지 않도록 유도한다.

---

#### 11.2 Data Augmentation의 주의사항

모든 변환이 항상 적절한 것은 아니다.

예를 들어 숫자 이미지를 분류할 때 이미지를 특정 각도로 회전하면 숫자 자체가 다른 숫자처럼 보일 수도 있다.

또한 의료 영상이나 생체신호에서는 특정 패턴이 실제 Label과 밀접하게 연결될 수 있다.

따라서 변환 후에도 원래의 Label이 유지되는지 확인해야 한다.

또한 Data Augmentation은 일반적으로 Training Set에 적용하며, Validation 및 Test Set에서는 일관된 평가 조건을 유지해야 한다.

**특히 원본 Sample에서 생성한 Augmented Sample이 Training Set과 Validation Set에 동시에 포함되어 Data Leakage가 발생하지 않도록 주의해야 한다.**

---

### 12. Noise Injection

Noise Injection은 Training 과정에서 Input Data 또는 Model의 일부 요소에 Random Noise를 추가하는 방법이다.

예를 들어 Input Data를 다음과 같이 표현할 수 있다.

$$
\boxed{ \tilde{\mathbf{x}} = \mathbf{x}+\boldsymbol{\epsilon} }
$$

여기서 $\boldsymbol{\epsilon}$은 Random Noise이다.

Gaussian Noise를 사용하는 경우 다음과 같이 나타낼 수 있다.

$$
\boxed{ \boldsymbol{\epsilon} \sim\mathcal{N}(0,\sigma^2I) }
$$

Noise Injection을 적용하면 모델이 작은 입력 변화에 지나치게 민감하지 않도록 학습을 유도할 수 있다.

즉 입력의 작은 Perturbation에 대해 Prediction이 안정적으로 유지되도록 만드는 데 도움이 될 수 있다.

---

#### 12.1 Data Augmentation과 Noise Injection의 차이

Noise Injection은 넓은 의미에서 Data Augmentation의 한 형태로 볼 수도 있다.

하지만 관점에는 차이가 있다.

| 구분 | Data Augmentation | Noise Injection |
| --- | --- | --- |
| 주요 방법 | 의미를 유지하는 데이터 변환 | Random Noise 추가 |
| 대표 예시 | Crop, Rotation, Flip | Gaussian Noise |
| 주요 목적 | 데이터 다양성 확보 | 작은 Perturbation에 대한 Robustness |
| 적용 대상 | 주로 Input Data | Input, Activation, Weight 등 |

참고 자료에서는 이미지 분류의 경우 단순 White Noise보다 Crop, Flip, Rotation, Color Jitter 등이 실제 데이터에서 발생하는 의미 있는 변화를 더 잘 반영할 수 있다고 설명한다.

따라서 Noise Injection이 항상 효과적인 것은 아니며, Dataset의 특성에 맞는 Noise Model을 선택해야 한다.

---

## Part 4. 모델 구조와 Parameter에 제약을 적용하는 기법

### 13. Batch Normalization

Batch Normalization은 Neural Network의 중간 Activation을 정규화하여 Training Stability를 개선하는 기법이다.

일반적으로 Mini-batch Statistics를 이용하여 다음과 같이 정규화한다.

$$
\boxed{ \hat{x}_i= \frac{x_i-\mu_B} {\sqrt{\sigma_B^2+\epsilon}} }
$$

이후 Learnable Parameter를 사용하여 Scale과 Shift를 적용한다.

$$
\boxed{ y_i=\gamma\hat{x}_i+\beta }
$$

Batch Normalization의 주요 목적은 Optimization을 안정화하고 효율적으로 수행하도록 돕는 것이다.

그런데 Training 과정에서는 Mini-batch마다 계산되는 Mean과 Variance가 달라질 수 있다.

이러한 Batch Statistics의 변동이 학습에 영향을 주어 간접적인 Regularization 효과를 나타낼 수 있다.

다만 주의해야 할 부분이 있다.

**Batch Normalization은 본래 L1·L2 Regularization처럼 Parameter의 크기를 제한하기 위한 기법이 아니다.**

또한 Batch Normalization을 적용한다고 해서 반드시 Generalization Performance가 개선되는 것은 아니다.

따라서 Batch Normalization은 Training Stability를 개선하는 기법이며, 상황에 따라 부수적인 Regularization 효과를 나타낼 수 있다고 이해하는 것이 적절하다.

---

### 14. Weight Constraint

Weight Constraint는 Model Parameter가 특정 조건을 만족하도록 직접 제한하는 방법이다.

L2 Regularization은 Weight의 크기에 비례하는 Penalty를 Loss Function에 추가한다.

반면 Weight Constraint는 Weight가 정해진 범위 밖으로 벗어나지 않도록 제한한다.

예를 들어 Weight Vector의 L2 Norm을 일정한 값 이하로 유지하도록 설정할 수 있다.

$$
\boxed{ \|\mathbf{w}\|_2\le c }
$$

여기서

- $\mathbf{w}$: Weight Vector
- $c$: 허용되는 Maximum Norm

이다.

일반적으로

$$
\|\mathbf{w}\|_2 = \sqrt{\sum_iw_i^2}
$$

이다.

Optimization 이후 Weight가 허용 범위를 벗어나면 Constraint를 만족하도록 조정할 수 있다.

따라서 L2 Regularization과 Weight Constraint의 차이는 다음과 같다.

| 구분 | L2 Regularization | Weight Constraint |
| --- | --- | --- |
| 방식 | Loss에 Penalty 추가 | Parameter의 허용 범위를 직접 제한 |
| Weight 크기 | 큰 Weight에 비용 부여 | 지정된 Constraint 내로 제한 |
| 표현 | $\lambda\sum_iw_i^2$ | $\|\mathbf{w}\|_2\le c$ |

두 방식은 모두 Weight를 제어할 수 있지만, 수학적 적용 방법은 다르다.

---

## Part 5. Target Label을 조정하는 Regularization

### 15. Label Smoothing

Label Smoothing은 Classification에서 사용하는 Target Label을 조정하는 방법이다.

일반적인 Multiclass Classification에서는 One-Hot Encoding을 사용한다.

예를 들어 Class가 세 개라면 다음과 같이 표현할 수 있다.

$$
\boxed{ \mathbf{y}=[1,0,0] }
$$

이 경우 정답 Class의 Target Probability는 1이고 나머지 Class는 0이다.

Label Smoothing은 이러한 Hard Label을 조금 부드럽게 조정한다.

대표적으로 다음 식을 사용할 수 있다.

$$
\boxed{ \tilde{\mathbf{y}} = (1-\epsilon)\mathbf{y} + \frac{\epsilon}{K}\mathbf{1} }
$$

여기서

- $\mathbf{y}$: 기존 One-Hot Label
- $\tilde{\mathbf{y}}$: Smoothed Label
- $\epsilon$: Smoothing Strength
- $K$: Class 개수
- $\mathbf{1}$: 모든 성분이 1인 Vector

이다.

---

#### 15.1 Label Smoothing 예제

Binary Classification에서 One-Hot Label을 다음과 같이 가정하자.

$$
\mathbf{y}=[1,0]
$$

이때

$$
K=2,\qquad\epsilon=0.2
$$

라고 설정하면

$$
\tilde{\mathbf{y}} = 0.8[1,0]+0.1[1,1]
$$

이다.

따라서

$$
\boxed{ \tilde{\mathbf{y}}=[0.9,0.1] }
$$

이다.

기존 Target Probability가 1과 0으로 구성되어 있었다면, Label Smoothing 이후에는 0.9와 0.1로 변한다.

이러한 방법은 모델이 특정 Class에 지나치게 높은 확신을 가지도록 학습되는 현상을 완화할 수 있다.

다만 Label Smoothing을 적용한다고 모델의 Confidence Calibration이 반드시 개선되거나 Accuracy가 항상 증가하는 것은 아니다.

---

### 16. Soft Labeling

Soft Labeling은 Target Label을 하나의 정답 Class에만 집중시키지 않고, 여러 Class에 대한 Probability Distribution 형태로 표현하는 방법이다.

예를 들어 세 가지 Class가 있다고 하자.

일반적인 Hard Label은 다음과 같다.

$$
\boxed{ \mathbf{y}=[1,0,0] }
$$

Soft Label은 다음과 같은 형태일 수 있다.

$$
\boxed{ \tilde{\mathbf{y}}=[0.7,0.2,0.1] }
$$

이 값은 Class 사이의 Semantic Similarity나 Annotation Uncertainty 등을 반영할 수 있다.

즉 첫 번째 Class가 가장 가능성이 높지만 다른 Class도 일부 가능성을 가진다고 표현할 수 있다.

---

#### 16.1 Label Smoothing과 Soft Labeling의 차이

Label Smoothing은 일반적으로 정해진 규칙에 따라 Hard Label을 부드럽게 조정한다.

반면 Soft Labeling은 Class 간 유사성이나 Label의 불확실성 등 의미 있는 정보를 반영할 수 있다.

| 구분 | Label Smoothing | Soft Labeling |
| --- | --- | --- |
| Target 형태 | Probability Distribution | Probability Distribution |
| 생성 방식 | 정해진 Smoothing Rule | Class 유사성·불확실성 등 반영 가능 |
| 주요 목적 | 과도한 확신 완화 | 부드러운 Target 정보 제공 |
| 관계 | Soft Labeling의 한 형태 | 더 넓은 개념 |

따라서 Label Smoothing은 Soft Labeling의 한 종류로 이해할 수 있다.

Soft Labeling이 모든 데이터에서 적절한 것은 아니며, Soft Label이 어떤 정보에 근거해 만들어졌는지 중요하다.

---

## Part 6. Regularization의 종합적인 이해

### 17. Regularization 기법 분류

지금까지 살펴본 기법은 서로 다른 방식으로 Overfitting을 완화하거나 학습을 제어한다.

| Regularization | 주요 접근 방법 | 핵심 특징 |
| --- | --- | --- |
| L1 | Weight 절댓값에 Penalty | Sparsity 유도 |
| L2 | Weight 제곱에 Penalty | Weight Shrinkage |
| Elastic Net | L1+L2 결합 | 희소성과 축소 효과 결합 |
| Early Stopping | 학습 시점 제어 | 과도한 Training 제한 |
| Dropout | Neuron 출력 무작위 제거 | 특정 Neuron에 대한 의존성 완화 |
| Data Augmentation | 학습 데이터 변환 | 학습 데이터 다양성 확보 |
| Batch Normalization | Activation 정규화 | Training 안정화, 간접 효과 가능 |
| Weight Constraint | Parameter 크기 제한 | 직접적인 Parameter 제약 |
| Noise Injection | Random Noise 추가 | Perturbation Robustness 유도 |
| Label Smoothing | Target Distribution 조정 | 과도한 Confidence 완화 |
| Soft Labeling | Soft Target 사용 | Label 불확실성·유사성 반영 |

이들 방법은 모두 Regularization과 연관되어 있지만, 적용 대상과 작동 원리는 다르다.

---

### 18. Regularization과 Weight Decay의 관계

Regularization을 공부할 때 자주 혼동하는 개념 중 하나가 Weight Decay이다.

앞서 L2 Regularization은 다음과 같이 표현하였다.

$$
\mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda\sum_iw_i^2
$$

일반적인 SGD에서는 L2 Penalty를 추가하면 Weight Update 과정에 Weight를 축소시키는 항이 나타난다.

따라서 적절한 계수 대응 아래에서 L2 Regularization과 Multiplicative Weight Decay는 같은 형태의 업데이트를 만들 수 있다.

하지만 Adam과 같은 Adaptive Optimizer에서는 이를 동일하게 취급할 수 없다.

Adam은 Gradient의 First Moment와 Second Moment를 이용하여 Update를 조정하기 때문이다.

이때 L2 Penalty를 Gradient에 추가하는 방식과 Weight Decay를 별도로 적용하는 방식은 서로 다른 결과를 만들 수 있다.

이를 구분하기 위해 AdamW에서는 **Decoupled Weight Decay**를 사용한다.

즉 다음 관계를 기억하는 것이 중요하다.

```text
Regularization
      │
      ├── L1 Penalty
      │
      ├── L2 Penalty
      │
      └── Weight Decay
              │
              └── AdamW의 Decoupled Weight Decay
```

Weight Decay는 Weight를 제어하는 Regularization 전략이지만, 모든 Optimizer에서 L2 Loss Penalty와 수학적으로 완전히 동일한 것은 아니다.

구체적인 AdamW 수식과 구현은 별도의 AdamW 게시글에서 다루는 것이 좋다.

---

### 19. Regularization Strength는 어떻게 선택할까?

Regularization에는 일반적으로 Strength를 결정하는 Hyperparameter가 존재한다.

예를 들어 L1·L2 Regularization에서는 $\lambda$를 사용한다.

$$
\boxed{ \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda\mathcal{R}(\theta) }
$$

Regularization Strength가 작다면 Regularization Term의 영향도 작아진다.

반대로 Regularization Strength가 커지면 Penalty의 영향이 강해질 수 있다.

| Regularization Strength | 예상되는 경향 |
| --- | --- |
| 너무 작음 | Overfitting이 충분히 완화되지 않을 수 있음 |
| 적절함 | Generalization Performance 개선 가능 |
| 너무 큼 | Underfitting 발생 가능 |

하지만 Regularization Strength에 대한 절대적인 정답은 존재하지 않는다.

Dataset, Model Architecture, Loss Function, Optimizer 등에 따라 적절한 값이 달라질 수 있다.

따라서 Validation Set 또는 Cross Validation을 이용하여 Regularization Strength를 선택할 수 있다.

---

### 20. Regularization과 Hyperparameter Tuning

Regularization은 Hyperparameter Tuning과도 밀접한 관계가 있다.

예를 들어 다음과 같은 값을 탐색할 수 있다.

**L1·L2 Regularization**

$$
\lambda\in\{10^{-4},10^{-3},10^{-2},10^{-1}\}
$$

**Dropout**

$$
p\in\{0.1,0.2,0.3,0.5\}
$$

**Early Stopping**

$$
patience\in\{5,10,20\}
$$

이 값들은 탐색 방법을 설명하기 위한 예시이다.

실제 선택 과정은 다음과 같이 구성할 수 있다.

```text
Regularization Candidates
           │
           ▼
      Model Training
           │
           ▼
   Validation Performance
           │
           ▼
    Hyperparameter Selection
           │
           ▼
        Final Model
```

중요한 점은 Regularization Strength를 Training Loss만으로 선택하지 않는 것이다.

Regularization이 강해지면 Training Loss가 증가하더라도 Validation Performance가 개선될 수 있다.

반대로 Training Loss가 낮아도 Validation Performance가 나쁠 수 있다.

따라서 Regularization의 효과는 Generalization 관점에서 평가해야 한다.

---

### 21. Regularization과 Normalization의 차이

Regularization과 Normalization은 이름이 비슷해 혼동하기 쉽다.

하지만 목적은 다르다.

**Regularization**

모델의 과도한 적합을 제어하고 Generalization Performance를 개선하는 데 초점을 둔다.

**Normalization**

데이터 또는 Activation의 Scale과 Distribution을 조정하는 데 초점을 둔다.

예를 들어 Standardization은 다음과 같다.

$$
\boxed{ z=\frac{x-\mu}{\sigma} }
$$

반면 L2 Regularization은 다음과 같다.

$$
\boxed{ \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda\sum_iw_i^2 }
$$

따라서 두 방법은 수학적 목적과 적용 방식이 다르다.

Batch Normalization처럼 Normalization을 수행하면서 간접적인 Regularization 효과가 나타날 수 있는 경우도 있지만, 두 개념을 동일하게 이해해서는 안 된다.

---

### 22. Regularization을 사용할 때 주의할 점

Regularization을 적용할 때는 다음을 함께 고려해야 한다.

**첫째, Regularization은 모든 Overfitting 문제를 자동으로 해결하지 않는다.**

Data Leakage, 부적절한 Train/Validation Split, 잘못된 Label 등 다른 문제가 있다면 Regularization만으로 해결할 수 없다.

**둘째, Regularization을 지나치게 강하게 적용하면 Underfitting이 발생할 수 있다.**

모델이 Training Data의 중요한 패턴까지 충분히 학습하지 못할 수 있다.

**셋째, 서로 다른 Regularization 방법을 함께 사용할 수 있지만 효과가 단순히 더해지는 것은 아니다.**

예를 들어 Dropout과 Batch Normalization을 함께 사용할 때는 Model Architecture와 적용 위치 등에 따라 동작이 달라질 수 있다.

**넷째, Regularization은 Validation Performance를 기준으로 평가해야 한다.**

Training Performance만 확인하면 Generalization의 개선 여부를 판단하기 어렵다.

---

## 23. 핵심 공식 정리

#### Regularized Objective

$$
\boxed{ \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda\mathcal{R}(\theta) }
$$

#### L1 Regularization

$$
\boxed{ \mathcal{R}_{L1}(\mathbf{w}) = \sum_i|w_i| }
$$

#### L2 Regularization

$$
\boxed{ \mathcal{R}_{L2}(\mathbf{w}) = \sum_iw_i^2 }
$$

#### Elastic Net

$$
\boxed{ \mathcal{R}_{ElasticNet} = \lambda_1\sum_i|w_i| + \lambda_2\sum_iw_i^2 }
$$

#### L2 Gradient

$$
\boxed{ \frac{\partial\mathcal{L}_{total}}{\partial w_i} = \frac{\partial\mathcal{L}_{data}}{\partial w_i} + 2\lambda w_i }
$$

#### Inverted Dropout

$$
\boxed{ \tilde h_i=\frac{m_ih_i}{1-p} }
$$

#### Gaussian Noise Injection

$$
\boxed{ \tilde{\mathbf{x}}= \mathbf{x}+\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2I) }
$$

#### Weight Constraint

$$
\boxed{ \|\mathbf{w}\|_2\le c }
$$

#### Label Smoothing

$$
\boxed{ \tilde{\mathbf{y}} = (1-\epsilon)\mathbf{y} + \frac{\epsilon}{K}\mathbf{1} }
$$

---

## 24. 최종 정리

Regularization은 Machine Learning과 Deep Learning에서 Overfitting을 완화하고 Generalization Performance를 개선하기 위해 사용하는 기법이다.

일반적으로 모델이 Training Data에 지나치게 적합되면 새로운 데이터에 대한 Prediction Performance가 저하될 수 있다.

Regularization은 이러한 문제를 완화하기 위해 Model Complexity를 제한하거나 Training Process에 제약을 적용한다.

대표적인 방법은 L1과 L2 Regularization이다.

L1 Regularization은 Weight의 절댓값 합에 Penalty를 부여하며, 일부 Weight를 0으로 만드는 Sparse Solution을 유도할 수 있다.

반면 L2 Regularization은 Weight의 제곱합에 Penalty를 부여하여 큰 Weight가 발생하는 것을 억제한다.

이 두 방법을 결합한 방식이 Elastic Net이다.

그러나 Regularization은 Weight Penalty에만 한정되지 않는다.

Early Stopping은 Validation Performance를 기준으로 Training을 중단하는 방법이며, Dropout은 Training 과정에서 일부 Neuron을 무작위로 비활성화하여 특정 Neuron에 대한 과도한 의존성을 완화한다.

Data Augmentation은 데이터에 적절한 변환을 적용하여 Training Data의 다양성을 높이고, Noise Injection은 입력 또는 모델의 일부 요소에 Noise를 추가하여 작은 변화에 대한 Robustness를 유도한다.

Label Smoothing과 Soft Labeling은 Target Label의 Distribution을 조정하여 모델이 보다 부드러운 Prediction을 학습하도록 할 수 있다.

또한 Batch Normalization은 주로 Training Stability를 개선하는 기법이지만 상황에 따라 간접적인 Regularization 효과를 제공할 수 있다.

각 기법은 적용 목적과 방법이 서로 다르다.

```text
Regularization
      │
      ├── Parameter Penalty
      │      ├── L1
      │      ├── L2
      │      └── Elastic Net
      │
      ├── Training Control
      │      ├── Early Stopping
      │      └── Dropout
      │
      ├── Data-based Methods
      │      ├── Data Augmentation
      │      └── Noise Injection
      │
      ├── Model Constraints
      │      ├── Weight Constraint
      │      └── Batch Normalization (간접 효과)
      │
      └── Target-based Methods
             ├── Label Smoothing
             └── Soft Labeling
```

따라서 Regularization을 이해할 때 중요한 것은 단순히 여러 방법의 이름을 암기하는 것이 아니다.

**각 방법이 모델의 어떤 요소를 제한하거나 변화시키며, 그 결과 Training Performance와 Generalization Performance 사이의 균형에 어떤 영향을 주는지 이해하는 것**이다.

---

### 핵심 요약

> Regularization은 모델의 Overfitting을 완화하고 Generalization Performance를 개선하기 위한 다양한 학습 전략이다. L1·L2는 Weight에 Penalty를 적용하고, Early Stopping은 Training Duration을 제어하며, Dropout은 Neuron 사이의 과도한 의존성을 완화한다. Data Augmentation과 Noise Injection은 데이터 또는 입력의 다양성을 높이고, Label Smoothing과 Soft Labeling은 Target Distribution을 조정한다. 적절한 Regularization 방법과 Strength는 Dataset과 Model의 특성에 따라 달라지므로 Validation Performance를 기준으로 선택해야 한다.

---

## 참고 자료

### 주요 참고 자료

- [Dsaint31 — Regularization](https://dsaint31.tistory.com/848)

### 공식 문서 및 추가 학습

- [scikit-learn — Linear Models: Ridge, Lasso, Elastic Net](https://scikit-learn.org/stable/modules/linear_model.html)
- [PyTorch — Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [PyTorch — AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch — CrossEntropyLoss와 Label Smoothing](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

### 관련 학습 시리즈

- [DL_a — 딥러닝 기초 개념 정리](../../index.md)
- [DL_c — 딥러닝 구현 및 실습](../../index.md)
