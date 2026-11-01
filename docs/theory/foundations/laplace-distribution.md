# 라플라스 분포 완전 정리: PDF·통계적 특성과 MAE·L1 정규화

[TOC]



---

Laplace Distribution(라플라스 분포)은 평균을 중심으로 좌우 대칭을 이루는 연속확률분포이다.

Gaussian Distribution과 유사하게 대칭적인 형태를 가지지만, 중심에서는 더 뾰족한 Peak를 가지고 Tail은 상대적으로 두껍다는 특징이 있다.

라플라스 분포는 Double Exponential Distribution이라고도 불린다.

대표적인 활용 분야는 다음과 같다.

- Signal Processing
- Image Denoising
- Sparse Signal Modeling
- Robust Regression
- Machine Learning
- L1 Regularization

특히 라플라스 분포는 머신러닝에서 사용하는 **Mean Absolute Error(MAE)** 와 **L1 Regularization**을 확률적으로 이해하는 데 중요한 역할을 한다.

이 글에서는 Laplace Distribution의 PDF와 주요 통계적 특성을 살펴보고, Gaussian Distribution과의 차이를 비교한다.

이후 Laplace Likelihood에서 MAE를 유도하고, Laplace Prior에서 L1 Regularization이 유도되는 원리를 정리한다.

---

## 1. Laplace Distribution이란?

Laplace Distribution은 위치 매개변수(Location Parameter)를 중심으로 좌우 대칭을 이루는 연속확률분포이다.

확률변수 $X$가 Laplace Distribution을 따른다면 다음과 같이 표기한다.

$$
\boxed{ X\sim\operatorname{Laplace}(\mu,b) }
$$

여기서

- $\mu$: Location Parameter
- $b$: Scale Parameter
- $b>0$

이다.

라플라스 분포의 주요 특징은 다음과 같다.

1. $\mu$에서 뾰족한 Peak를 가진다.
2. 좌우 대칭이다.
3. Gaussian Distribution보다 상대적으로 두꺼운 Tail을 가진다.
4. 평균과 중앙값은 모두 $\mu$이다.
5. 분산은 $2b^2$이다.

---

## 2. Probability Density Function(PDF)

Laplace Distribution의 PDF는 다음과 같다.

$$
\boxed{ f(x\mid\mu,b) = \frac{1}{2b} \exp \left( -\frac{|x-\mu|}{b} \right) }
$$

여기서

- $x$: Random Variable의 값
- $\mu$: Location Parameter
- $b$: Scale Parameter

이다.

PDF는 확률밀도를 나타내며, 특정 구간의 확률은 적분하여 계산한다.

$$
\boxed{ P(a\le X\le c) = \int_a^c f(x)\,dx }
$$

---

### 2.1 PDF의 의미

PDF를 다시 살펴보자.

$$
f(x) = \frac{1}{2b} e^{-|x-\mu|/b}
$$

이 식에서 중요한 부분은 다음 항이다.

$$
\boxed{ |x-\mu| }
$$

이는 확률변수의 값이 중심 $\mu$에서 얼마나 떨어져 있는지를 나타낸다.

$x=\mu$라면

$$
|x-\mu|=0
$$

이다.

따라서

$$
f(\mu) = \frac{1}{2b}e^0
$$

이므로

$$
\boxed{ f(\mu)=\frac{1}{2b} }
$$

이다.

즉 PDF는 $x=\mu$에서 최대값을 가진다.

반면 $x$가 $\mu$에서 멀어질수록

$$
|x-\mu|
$$

가 증가하고 지수항은 감소한다.

따라서 확률밀도도 감소한다.

---

## 3. Laplace Distribution의 대칭성

Laplace Distribution은 $\mu$를 중심으로 좌우 대칭이다.

이를 확인하기 위해 다음 두 값을 생각하자.

$$
x_1=\mu-d
$$

$$
x_2=\mu+d
$$

여기서 $d>0$이다.

Laplace PDF에 대입하면

$$
f(\mu-d) = \frac{1}{2b} e^{-d/b}
$$

이다.

반면

$$
f(\mu+d) = \frac{1}{2b} e^{-d/b}
$$

이다.

따라서

$$
\boxed{ f(\mu-d)=f(\mu+d) }
$$

이다.

즉 중심에서 같은 거리만큼 떨어진 두 지점의 확률밀도는 동일하다.

---

## 4. Location Parameter

Location Parameter인 $\mu$는 Laplace Distribution의 중심을 결정한다.

라플라스 분포에서는 다음 값들이 모두 $\mu$와 같다.

$$
\boxed{ E[X]=\mu }
$$

$$
\boxed{ \operatorname{Median}(X)=\mu }
$$

$$
\boxed{ \operatorname{Mode}(X)=\mu }
$$

즉 평균, 중앙값, 최빈값이 모두 동일하다.

$\mu$가 변화하면 분포 전체가 좌우로 이동한다.

예를 들어

$$
\mu=0
$$

이면 0을 중심으로 분포가 형성된다.

반면

$$
\mu=5
$$

이면 5를 중심으로 분포가 형성된다.

---

## 5. Scale Parameter

Scale Parameter인 $b$는 분포가 얼마나 넓게 퍼져 있는지 결정한다.

$$
\boxed{ b>0 }
$$

$b$가 작으면 분포가 중심에 강하게 집중된다.

반대로 $b$가 크면 분포의 폭이 넓어진다.

Laplace Distribution의 분산은 다음과 같다.

$$
\boxed{ \operatorname{Var}(X)=2b^2 }
$$

따라서 Standard Deviation은 다음과 같다.

$$
\boxed{ \sigma=\sqrt{2}b }
$$

이다.

즉 $b$는 표준편차와 동일한 값이 아니다.

이 점은 Gaussian Distribution과 비교할 때 중요하다.

---

## 6. Laplace Distribution의 CDF

Laplace Distribution의 Cumulative Distribution Function(CDF)은 다음과 같다.

$$
\boxed{ F(x)= \begin{cases} \frac12 e^{(x-\mu)/b}, & x<\mu \\[6pt] 1-\frac12e^{-(x-\mu)/b}, & x\ge\mu \end{cases} }
$$

CDF는 확률변수 $X$가 특정 값 $x$ 이하일 확률을 나타낸다.

$$
\boxed{ F(x)=P(X\le x) }
$$

특히 중심에서의 CDF를 계산하면

$$
F(\mu) = 1-\frac12e^0
$$

이므로

$$
\boxed{ F(\mu)=0.5 }
$$

이다.

즉 Laplace Distribution에서는 중심 $\mu$를 기준으로 왼쪽과 오른쪽의 확률이 각각 50%이다.

---

## 7. Laplace Distribution의 Mean과 Variance

라플라스 분포의 주요 통계량을 정리하면 다음과 같다.

#### Mean

$$
\boxed{ E[X]=\mu }
$$

#### Median

$$
\boxed{ \operatorname{Median}(X)=\mu }
$$

#### Variance

$$
\boxed{ \operatorname{Var}(X)=2b^2 }
$$

#### Standard Deviation

$$
\boxed{ \sigma=\sqrt{2}b }
$$

#### Skewness

$$
\boxed{ \operatorname{Skewness}=0 }
$$

라플라스 분포는 좌우 대칭이기 때문에 Skewness가 0이다.

---

## 8. Kurtosis(첨도)

Kurtosis는 분포의 네 번째 표준화 적률을 나타낸다.

Laplace Distribution의 Kurtosis는 다음과 같다.

$$
\boxed{ \operatorname{Kurtosis}=6 }
$$

반면 Gaussian Distribution의 Kurtosis는 다음과 같다.

$$
\boxed{ \operatorname{Kurtosis}=3 }
$$

따라서 Laplace Distribution은 Gaussian Distribution보다 높은 Kurtosis를 가진다.

참고로 Excess Kurtosis를 사용하는 경우에는 Gaussian Distribution의 값을 0으로 정의한다.

따라서

$$
\boxed{ \text{Laplace Excess Kurtosis}=3 }
$$

이다.

이러한 차이는 Laplace Distribution이 Gaussian Distribution보다 상대적으로 두꺼운 Tail을 가진다는 특성과 연결된다.

---

## 9. Gaussian Distribution과 비교

Laplace Distribution과 Gaussian Distribution은 모두 대칭적인 연속확률분포이다.

하지만 PDF의 형태와 Tail의 감소 방식이 다르다.

#### Gaussian Distribution

$$
\boxed{ f_G(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x-\mu)^2}{2\sigma^2} \right) }
$$

#### Laplace Distribution

$$
\boxed{ f_L(x) = \frac{1}{2b} \exp \left( -\frac{|x-\mu|}{b} \right) }
$$

두 분포의 주요 차이는 지수함수 안에 들어가는 오차항이다.

Gaussian Distribution은 제곱 오차를 사용한다.

$$
\boxed{ (x-\mu)^2 }
$$

반면 Laplace Distribution은 절댓값 오차를 사용한다.

$$
\boxed{ |x-\mu| }
$$

따라서 Tail의 감소 방식이 다르다.

| 구분 | Gaussian | Laplace |
| --- | --- | --- |
| 형태 | 부드러운 종 모양 | 뾰족한 중심 |
| 지수항 | Squared Error | Absolute Error |
| Tail | 상대적으로 얇음 | 상대적으로 두꺼움 |
| Variance | $\sigma^2$ | $2b^2$ |
| Kurtosis | 3 | 6 |
| NLL 기반 회귀 | MSE | MAE |

---

## 10. Python 실습: Laplace와 Gaussian PDF

> SciPy 시각화 예제를 기반으로 작성하였다.

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import laplace, norm

mu = 0
b = 1

sigma = np.sqrt(2) * b

x = np.linspace(
    -8,
    8,
    1000
)

laplace_pdf = laplace.pdf(
    x,
    loc=mu,
    scale=b
)

normal_pdf = norm.pdf(
    x,
    loc=mu,
    scale=sigma
)

plt.figure(figsize=(9, 5))

plt.plot(
    x,
    laplace_pdf,
    label="Laplace"
)

plt.plot(
    x,
    normal_pdf,
    label="Gaussian"
)

plt.xlabel("x")
plt.ylabel("Probability Density")

plt.title(
    "Laplace vs Gaussian"
)

plt.legend()
plt.grid(alpha=0.3)

plt.show()
```

여기서는 두 분포의 분산을 동일하게 설정하였다.

Laplace Distribution의 분산은 다음과 같다.

$$
\operatorname{Var}(X)=2b^2
$$

따라서 Gaussian Distribution의 표준편차를

$$
\boxed{ \sigma=\sqrt2b }
$$

로 설정하면 두 분포의 분산이 같아진다.

이는 두 분포의 형태를 공정하게 비교하기 위한 설정이다.

그래프에서 Laplace Distribution은 중심이 더 뾰족하며 Tail이 상대적으로 두꺼운 특성을 확인할 수 있다.

---

## 11. Laplace Distribution과 Outlier

Outlier(이상치)는 데이터의 일반적인 분포에서 크게 벗어난 값이다.

Gaussian Distribution과 Laplace Distribution은 큰 오차에 대해 서로 다른 형태의 확률밀도를 부여한다.

Gaussian Distribution의 지수항은 다음과 같다.

$$
\exp \left( -\frac{r^2}{2\sigma^2} \right)
$$

반면 Laplace Distribution의 지수항은 다음과 같다.

$$
\exp \left( -\frac{|r|}{b} \right)
$$

여기서

$$
r=y-\hat y
$$

는 오차이다.

Gaussian Distribution은 오차의 제곱을 사용하기 때문에 큰 오차에서 확률밀도가 더 빠르게 감소한다.

반면 Laplace Distribution은 절댓값 오차를 사용하므로 Tail이 상대적으로 두껍다.

이러한 차이는 Regression에서 사용하는 Loss Function과 연결된다.

---

## 12. Laplace Likelihood에서 MAE 유도

Regression 모델이 입력 $x$에 대해 예측값을 생성한다고 하자.

$$
\boxed{ \hat y=\mu_\theta(x) }
$$

실제 Target이 예측값을 중심으로 Laplace Distribution을 따른다고 가정한다.

$$
\boxed{ Y\mid X=x \sim \operatorname{Laplace} \left( \mu_\theta(x),b \right) }
$$

이때 조건부 PDF는 다음과 같다.

$$
q_\theta(y\mid x,b) = \frac{1}{2b} \exp \left( -\frac{|y-\mu_\theta(x)|}{b} \right)
$$

이다.

---

### 12.1 Negative Log-Likelihood

NLL을 계산하면 다음과 같다.

$$
\ell_{NLL} = -\log q_\theta(y\mid x,b)
$$

따라서

$$
\ell_{NLL} = -\log \left[ \frac{1}{2b} \exp \left( -\frac{|y-\mu_\theta(x)|}{b} \right) \right]
$$

이다.

로그의 성질을 이용하면

$$
\boxed{ \ell_{NLL} = \log(2b) + \frac{|y-\mu_\theta(x)|}{b} }
$$

이다.

---

### 12.2 Scale Parameter가 고정된 경우

여기서 Scale Parameter $b$가 고정되어 있다고 가정하자.

그러면

$$
\log(2b)
$$

는 상수이다.

또한

$$
\frac1b
$$

도 상수이다.

따라서 모델 Parameter에 따라 변화하는 부분은 다음과 같다.

$$
\boxed{ |y-\mu_\theta(x)| }
$$

이다.

전체 데이터에 대한 평균을 계산하면

$$
\boxed{ L_{MAE} = \frac1N \sum_{i=1}^{N} |y_i-\mu_\theta(x_i)| }
$$

이다.

따라서

$$
\boxed{ \text{Laplace NLL Minimization} \Longleftrightarrow \text{MAE Minimization} }
$$

이다.

단, 이 대응관계는 Scale Parameter가 고정된 조건에서 성립한다.

---

## 13. MAE와 MSE의 비교

MSE와 MAE는 오차에 대해 서로 다른 Penalty를 부여한다.

#### Mean Squared Error

$$
\boxed{ L_{MSE} = \frac1N \sum_{i=1}^{N} (y_i-\hat y_i)^2 }
$$

#### Mean Absolute Error

$$
\boxed{ L_{MAE} = \frac1N \sum_{i=1}^{N} |y_i-\hat y_i| }
$$

예를 들어 다음 오차를 생각하자.

| Error | Absolute Error | Squared Error |
| --- | --- | --- |
| 1 | 1 | 1 |
| 2 | 2 | 4 |
| 3 | 3 | 9 |
| 5 | 5 | 25 |

오차가 커질수록 MSE는 더 빠르게 증가한다.

반면 MAE는 선형적으로 증가한다.

따라서 큰 이상치가 존재하는 상황에서 MAE가 MSE보다 상대적으로 강건한 특성을 가질 수 있다.

---

## 14. PyTorch에서 MAE 사용

PyTorch에서는 `nn.L1Loss`로 MAE를 계산할 수 있다.

```python
import torch
import torch.nn as nn

predictions = torch.tensor([
    1.0,
    2.0,
    3.0
])

targets = torch.tensor([
    1.5,
    2.5,
    10.0
])

mae = nn.L1Loss()

mse = nn.MSELoss()

print(
    "MAE:",
    mae(predictions, targets).item()
)

print(
    "MSE:",
    mse(predictions, targets).item()
)
```

이 예제에서 오차는 다음과 같다.

$$
|1.0-1.5|=0.5
$$

$$
|2.0-2.5|=0.5
$$

$$
|3.0-10.0|=7.0
$$

따라서 MAE는 다음과 같다.

$$
\boxed{ L_{MAE} = \frac{0.5+0.5+7}{3} = \frac83 }
$$

이다.

반면 MSE는 다음과 같다.

$$
\boxed{ L_{MSE} = \frac{0.25+0.25+49}{3} = 16.5 }
$$

이다.

즉 큰 오차에 대해 두 Loss Function이 서로 다른 Penalty를 부여하는 것을 확인할 수 있다.

---

## 15. Laplace Prior와 L1 Regularization

지금까지는 Laplace Distribution을 Regression의 Error Distribution으로 사용하였다.

하지만 Laplace Distribution은 Model Parameter의 Prior Distribution으로도 사용할 수 있다.

이 경우 L1 Regularization과 연결된다.

모델의 Parameter를 다음과 같이 정의하자.

$$
\boldsymbol{\theta} = (\theta_1,\theta_2,\cdots,\theta_d)
$$

각 Parameter가 평균 0인 Laplace Distribution을 따른다고 가정한다.

$$
\boxed{ \theta_j \sim \operatorname{Laplace}(0,b) }
$$

---

### 15.1 Laplace Prior의 PDF

각 Parameter의 Prior는 다음과 같다.

$$
\boxed{ P(\theta_j) = \frac{1}{2b} \exp \left( -\frac{|\theta_j|}{b} \right) }
$$

이다.

Parameter들이 독립이라고 가정하면 전체 Prior는 다음과 같다.

$$
P(\boldsymbol{\theta}) = \prod_{j=1}^{d} P(\theta_j)
$$

이다.

---

### 15.2 Negative Log-Prior

Negative Logarithm을 적용하면

$$
-\log P(\boldsymbol{\theta}) = -\sum_{j=1}^{d} \log P(\theta_j)
$$

이다.

Laplace Prior를 대입하면

$$
-\log P(\boldsymbol{\theta}) = \sum_{j=1}^{d} \left[ \log(2b) + \frac{|\theta_j|}{b} \right]
$$

이다.

따라서

$$
\boxed{ -\log P(\boldsymbol{\theta}) = \frac1b \sum_{j=1}^{d} |\theta_j| + C }
$$

이다.

여기서 $C$는 Parameter에 의존하지 않는 상수이다.

L1 Norm은 다음과 같다.

$$
\boxed{ \|\boldsymbol{\theta}\|_1 = \sum_{j=1}^{d}|\theta_j| }
$$

따라서

$$
\boxed{ -\log P(\boldsymbol{\theta}) = \frac1b \|\boldsymbol{\theta}\|_1 + C }
$$

이다.

---

## 16. MAP Estimation과 L1 Regularization

Maximum a Posteriori Estimation(MAP)은 Posterior Distribution을 최대화하는 Parameter를 찾는다.

$$
\boxed{ \hat\theta_{MAP} = \underset{\theta}{\arg\max} P(D\mid\theta)P(\theta) }
$$

여기에 Negative Logarithm을 적용하면

$$
\boxed{ J_{MAP}(\theta) = -\log P(D\mid\theta) - \log P(\theta) }
$$

이다.

Laplace Prior를 사용하면 다음 형태를 얻는다.

$$
\boxed{ J_{MAP}(\theta) = L_{NLL}(\theta) + \lambda\|\theta\|_1 }
$$

이다.

따라서

$$
\boxed{ \text{Laplace Prior} \Longrightarrow \text{L1 Regularization} }
$$

이다.

L1 Regularization은 Model Parameter의 절댓값의 합에 Penalty를 부여한다.

이는 일부 Parameter가 0이 되는 Sparse Solution을 유도할 수 있다.

이러한 특성 때문에 LASSO Regression에서도 사용된다.

---

## 17. Observation Distribution과 Parameter Prior

여기서 중요한 차이를 정리할 필요가 있다.

같은 Laplace Distribution을 사용하더라도 적용 대상에 따라 의미가 다르다.

#### Observation Distribution

Target과 예측값 사이의 오차를 Laplace Distribution으로 모델링한다.

$$
\boxed{ P(y\mid x,\theta) }
$$

고정 Scale 조건에서 MAE Loss와 연결된다.

#### Parameter Prior

모델 Parameter가 Laplace Distribution을 따른다고 가정한다.

$$
\boxed{ P(\theta) }
$$

L1 Regularization과 연결된다.

이를 비교하면 다음과 같다.

| 구분 | Laplace Observation Distribution | Laplace Parameter Prior |
| --- | --- | --- |
| 적용 대상 | Prediction Error | Model Parameters |
| 확률분포 | $P(y\mid x,\theta)$ | $P(\theta)$ |
| 연결 개념 | MAE | L1 Regularization |
| 추정 방법 | MLE | MAP |

따라서 같은 Laplace Distribution이라도 어디에 적용하는지에 따라 다른 목적함수를 유도한다.

---

## 18. Laplacian Kernel

Laplace Distribution은 L1 Norm을 사용하는 Exponential Similarity와도 관련된다.

대표적인 형태는 다음과 같다.

$$
\boxed{ K(\mathbf{x},\mathbf{y}) = \exp \left( -\gamma \|\mathbf{x}-\mathbf{y}\|_1 \right) }
$$

여기서

- $\mathbf{x},\mathbf{y}$: Feature Vectors
- $\gamma$: Scale Parameter
- $\|\cdot\|_1$: L1 Norm

이다.

L1 Distance가 작으면 Similarity가 커진다.

반면 두 Vector 사이의 L1 Distance가 커지면 Similarity는 지수적으로 감소한다.

이러한 형태는 Laplacian Kernel이라고 한다.

Gaussian RBF Kernel에서는 일반적으로 Squared Euclidean Distance를 사용한다.

$$
\boxed{ K_{RBF}(\mathbf{x},\mathbf{y}) = \exp \left( -\gamma \|\mathbf{x}-\mathbf{y}\|_2^2 \right) }
$$

즉 두 Kernel은 서로 다른 Distance Measure를 사용한다.

---

## 19. Laplace Smoothing과의 구분

Laplace Distribution과 Laplace Smoothing은 이름이 비슷하지만 다른 개념이다.

**Laplace Distribution**

연속확률분포이며 PDF는 다음과 같다.

$$
f(x) = \frac{1}{2b} e^{-|x-\mu|/b}
$$

**Laplace Smoothing**

확률 추정에서 관측되지 않은 사건의 확률이 0이 되는 문제 등을 완화하기 위한 Smoothing 기법이다.

예를 들어 Add-One Smoothing에서는 다음과 같은 형태를 사용할 수 있다.

$$
\boxed{ P(x_i) = \frac{n_i+1}{N+K} }
$$

여기서

- $n_i$: 해당 Category의 관측 횟수
- $N$: 전체 관측 횟수
- $K$: Category 개수

이다.

따라서 Laplace Smoothing은 Laplace Distribution의 PDF를 직접 적용하는 방법이 아니다.

---

## 20. 최종 정리

Laplace Distribution은 Location Parameter $\mu$와 Scale Parameter $b$를 가지는 연속확률분포이다.

PDF는 다음과 같다.

$$
\boxed{ f(x\mid\mu,b) = \frac{1}{2b} \exp \left( -\frac{|x-\mu|}{b} \right) }
$$

이다.

분포는 $\mu$를 중심으로 좌우 대칭이며 Gaussian Distribution보다 뾰족한 중심과 상대적으로 두꺼운 Tail을 가진다.

평균과 중앙값은 모두 $\mu$이다.

$$
\boxed{ E[X]=\mu }
$$

분산은 다음과 같다.

$$
\boxed{ \operatorname{Var}(X)=2b^2 }
$$

Laplace Distribution은 머신러닝의 Loss Function 및 Regularization과도 연결된다.

Regression에서 Target Error가 Laplace Distribution을 따른다고 가정하고 Scale Parameter가 고정되어 있다면 NLL 최소화는 MAE 최소화와 연결된다.

반면 Model Parameter에 Laplace Prior를 적용하면 MAP Estimation을 통해 L1 Regularization이 유도된다.

따라서 다음 두 관계를 구분하는 것이 중요하다.

$$
\boxed{ \text{Laplace Likelihood} \rightarrow \text{MAE} }
$$

$$
\boxed{ \text{Laplace Prior} \rightarrow \text{L1 Regularization} }
$$

이러한 관계를 이해하면 MAE와 L1 Regularization이 단순한 계산 공식이 아니라 확률분포에 대한 가정에서 유도될 수 있다는 사실을 알 수 있다.

---

### 핵심 요약

> Laplace Distribution은 뾰족한 중심과 두꺼운 Tail을 가지는 연속확률분포이다. PDF는 $\frac{1}{2b}e^{-|x-\mu|/b}$이며, 평균은 $\mu$, 분산은 $2b^2$이다. 고정 Scale Laplace Likelihood의 NLL은 MAE와 연결되고, Laplace Prior의 Negative Log는 L1 Regularization과 연결된다.

---

## 참고 자료

- [Laplace Distribution — Dsaint31](https://dsaint31.tistory.com/882)
- [Probability Distribution — Dsaint31](https://dsaint31.tistory.com/707)
- [MLE와 Negative Log-Likelihood — Dsaint31](https://dsaint31.tistory.com/991)
- [SciPy — Laplace Distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html)
- [PyTorch — L1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
