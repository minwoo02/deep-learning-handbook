# Perceptron에서 MLP와 Backpropagation까지: 신경망은 어떻게 학습하는가?

[TOC]


관련 실습: [scikit-learn MLPClassifier 이해하기](../../practice/sklearn/sklearn-mlp-classifier.md) · [PyTorch Custom 모델 구현: nn.Module 상속부터 모델 관리까지](../../practice/pytorch-basics/custom-nn-module.md) · [PyTorch로 MLP 직접 구현하기](../../practice/classification/iris-mlp-training.md) · [PyTorch로 XOR 해결하기: MLP 클래스 구조 분석](../../practice/classification/xor-mlp.md) · [PyTorch로 Iris Dataset 다중 분류 구현하기: Dataset부터 ANN 학습 및 평가까지](../../practice/classification/iris-classification-pipeline.md)


---

## Perceptron에서 MLP와 Backpropagation까지: 신경망은 어떻게 학습하는가?

인공신경망의 기본 구조를 이해하려면 먼저 **Perceptron 하나가 어떤 계산을 수행하는지** 살펴볼 필요가 있다.

Perceptron에서 시작하면 이후 등장하는

- Weight와 Bias
- Activation Function
- Hidden Layer
- MLP
- Backpropagation
- Gradient
- Training

이 서로 별개의 개념이 아니라 하나의 흐름으로 연결되어 있다는 것을 알 수 있다.

전체 흐름을 먼저 보면 다음과 같다.

```text
Perceptron
    ↓
Linear Combination
    ↓
Activation Function
    ↓
단일 선형 모델의 한계
    ↓
Hidden Layer를 추가한 MLP
    ↓
Backpropagation
    ↓
Weight / Bias Update
    ↓
Training
```

---

## 1. Perceptron

Perceptron은 초기 인공신경망의 가장 기본적인 모델 중 하나이다.

하나의 인공 뉴런은 입력값을 받아 다음과 같은 계산을 수행한다.

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

그리고 계산된 $z$를 활성화 함수(Activation Function)에 통과시킨다.

$$
\hat{y}=f(z)
$$

Perceptron에서는 주로 **Step Function**을 사용한다.

$$
\hat{y} = \operatorname{step} (\mathbf{w}^T\mathbf{x}+b)
$$

---

### 1.1 Input

입력 데이터를 벡터로 나타내면 다음과 같다.

$$
\mathbf{x} = \begin{bmatrix} x_1\\ x_2\\ \vdots\\ x_n \end{bmatrix}
$$

각 $x_i$는 하나의 Feature를 의미한다.

예를 들어 어떤 데이터를 분류하기 위해

- 길이
- 너비
- 온도
- 전압

등을 사용한다면 각각이 하나의 입력 Feature가 될 수 있다.

---

### 1.2 Weight

각 입력값에는 Weight가 곱해진다.

$$
w_1x_1+w_2x_2+\cdots+w_nx_n
$$

Weight는 각 입력이 모델의 판단에 얼마나 큰 영향을 미치는지를 조절한다.

Weight들을 벡터로 표현하면

$$
\mathbf{w} = \begin{bmatrix} w_1\\ w_2\\ \vdots\\ w_n \end{bmatrix}
$$

이고,

입력과 Weight의 계산은 내적(Dot Product)을 이용하여

$$
\mathbf{w}^T\mathbf{x}
$$

로 표현할 수 있다.

즉,

$$
\mathbf{w}^T\mathbf{x} = w_1x_1+w_2x_2+\cdots+w_nx_n
$$

이다.

---

### 1.3 Bias

Perceptron은 여기에 Bias를 추가한다.

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

Bias는 모델의 판단 기준을 이동시키는 역할을 한다.

Weight가 입력값의 영향력을 조절한다면, Bias는 전체 기준점을 이동시킨다고 볼 수 있다.

---

### 1.4 Parameter

딥러닝에서 Weight와 Bias처럼 **Training을 통해 모델이 스스로 학습하는 값**을 Parameter라고 한다.

따라서

$$
\boxed{ \text{Parameter} = \text{Weight} + \text{Bias} }
$$

로 이해할 수 있다.

신경망의 Training은 결국 적절한 $W$와 $b$를 찾는 과정이다.

---

## 2. Perceptron의 Activation Function

Perceptron에서는 계산된

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

를 Step Function에 통과시킨다.

대표적인 형태는 다음과 같다.

$$
f(z)= \begin{cases} 1 & z\ge0\\ 0 & z<0 \end{cases}
$$

즉,

```text
z < 0
→ OFF
→ 0

z ≥ 0
→ ON
→ 1
```

과 같이 동작한다.

따라서 Perceptron의 전체 계산은 다음과 같다.

```text
Input
  ↓
Weight × Input
  ↓
Summation + Bias
  ↓
Step Function
  ↓
Output
```

수식으로는

$$
\boxed{ \hat{y} = \operatorname{step} (\mathbf{w}^T\mathbf{x}+b) }
$$

이다.

---

## 3. Perceptron의 한계

단일 Perceptron은 기본적으로 **Linear Classifier**이다.

즉 하나의 선형 결정 경계를 기준으로 데이터를 두 영역으로 나눈다.

따라서 선형적으로 분리할 수 없는 문제에는 한계가 있다.

대표적인 예가 XOR 문제이다.

XOR에서는 하나의 직선만으로 두 클래스를 완전히 분리할 수 없다.

```text
Single Perceptron
        ↓
Linear Decision Boundary
        ↓
Non-linear Problem 해결 불가
```

이러한 한계에서 자연스럽게 다음 질문이 나온다.

> 하나의 Perceptron으로 해결할 수 없다면 여러 개를 연결하면 어떨까?

이 아이디어가 **Multi-Layer Perceptron, MLP**로 이어진다.

---

## 4. Multi-Layer Perceptron

MLP는 여러 개의 뉴런을 Layer 형태로 연결한 신경망이다.

기본적인 구조는 다음과 같다.

```text
Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

입력층과 출력층 사이에 존재하는 Layer를 **Hidden Layer**라고 한다.

Hidden Layer를 여러 개 사용할 수도 있다.

```text
Input
  ↓
Hidden Layer 1
  ↓
Hidden Layer 2
  ↓
...
  ↓
Output
```

---

### 4.1 PyTorch에서의 MLP

예를 들어 다음과 같은 구조를 생각할 수 있다.

```python
self.linear1 = nn.Linear(1, 10)
self.relu = nn.ReLU()
self.linear2 = nn.Linear(10, 1)
```

Forward 과정에서는

```python
x = self.linear1(x)
x = self.relu(x)
x = self.linear2(x)
```

와 같이 동작한다.

구조를 나타내면 다음과 같다.

```text
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
Output
```

여기서 중요한 것은 단순히 Linear Layer를 여러 개 쌓는 것만으로는 충분하지 않다는 점이다.

---

## 5. 왜 Non-linear Activation Function이 필요한가?

만약 Hidden Layer 사이에 Activation Function을 사용하지 않는다고 가정해보자.

첫 번째 Layer가

$$
\mathbf{y}_1 = \mathbf{W}_1\mathbf{x}
$$

이고,

두 번째 Layer가

$$
\mathbf{y}_2 = \mathbf{W}_2\mathbf{y}_1
$$

이라면,

$$
\mathbf{y}_2 = \mathbf{W}_2\mathbf{W}_1\mathbf{x}
$$

이다.

세 번째 Layer까지 사용하면

$$
\mathbf{y}_3 = \mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1 \mathbf{x}
$$

가 된다.

그런데 행렬의 곱

$$
\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1
$$

역시 하나의 행렬로 표현할 수 있다.

$$
\mathbf{W}_{new} = \mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1
$$

따라서

$$
\mathbf{y}_3 = \mathbf{W}_{new}\mathbf{x}
$$

가 된다.

즉 Linear Layer만 계속 쌓으면

$$
\boxed{ \text{Linear} \circ \text{Linear} \circ \text{Linear} = \text{Linear} }
$$

이므로 아무리 깊게 쌓아도 결국 하나의 Linear Layer와 같은 형태가 된다.

---

## 6. Non-linear Activation Function

각 Layer 사이에 ReLU나 Sigmoid 같은 Non-linear Activation Function을 넣으면 상황이 달라진다.

예를 들어

$$
\mathbf{a}_1 = f(\mathbf{W}_1\mathbf{x})
$$

$$
\mathbf{a}_2 = f(\mathbf{W}_2\mathbf{a}_1)
$$

이라면 전체 함수는

$$
\mathbf{a}_2 = f \left( \mathbf{W}_2 f \left( \mathbf{W}_1\mathbf{x} \right) \right)
$$

가 된다.

중간에 비선형 함수 $f$가 존재하기 때문에 더 이상 하나의 행렬 곱으로 합칠 수 없다.

따라서 Deep Neural Network에서는

```text
Linear Transformation
        ↓
Non-linear Activation
        ↓
Linear Transformation
        ↓
Non-linear Activation
```

이 반복된다.

이 **Non-linearity**가 MLP가 단일 Perceptron보다 복잡한 문제를 표현할 수 있게 만드는 핵심 요소이다.

---

## 7. Step Function은 왜 Deep Learning에 적합하지 않은가?

Perceptron에서는 Step Function을 사용했다.

```text
입력 < 0
→ 0

입력 ≥ 0
→ 1
```

하지만 Deep Neural Network에서는 Backpropagation을 통해 Parameter를 학습해야 한다.

Backpropagation에서는 함수의 **Gradient**가 필요하다.

Step Function의 경우

- 대부분의 구간에서 기울기가 0이고
- $z=0$에서는 함수가 불연속이므로 일반적인 의미의 미분이 정의되지 않는다.

즉 Weight를 어느 방향으로 얼마나 수정해야 하는지에 대한 유용한 Gradient 정보를 얻기 어렵다.

예를 들어 Gradient가 0이라면

$$
w_{new} = w_{old} - \eta\cdot0
$$

이므로

$$
w_{new}=w_{old}
$$

가 된다.

Weight가 변하지 않는 것이다.

---

## 8. Sigmoid

이러한 문제를 해결하기 위해 사용된 대표적인 함수가 Sigmoid이다.

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

Sigmoid는 S자 형태의 부드러운 함수이다.

Step Function과 달리 출력이 갑자기 0에서 1로 변하지 않고 연속적으로 변화한다.

```text
0
      ↗
    ↗
  ↗
       1
```

따라서 Gradient를 계산하여 Backpropagation에 사용할 수 있다.

---

## 9. Sigmoid와 Vanishing Gradient

Sigmoid 역시 완벽한 함수는 아니다.

입력이 매우 큰 양수 또는 음수가 되면 출력이 각각 1 또는 0 근처에서 포화된다.

이 영역에서는 미분값이 매우 작아진다.

Sigmoid의 미분은

$$
\sigma'(x) = \sigma(x) (1-\sigma(x))
$$

이다.

Deep Neural Network에서는 Backpropagation 과정에서 여러 Layer의 Gradient가 Chain Rule에 의해 곱해진다.

예를 들어 작은 값이 계속 곱해지면

$$
0.1 \times 0.1 \times 0.1 \times \cdots
$$

처럼 Gradient가 점점 작아질 수 있다.

그 결과 입력에 가까운 Layer의 Parameter가 거의 업데이트되지 않을 수 있다.

이를 **Vanishing Gradient Problem**이라고 한다.

---

## 10. ReLU

ReLU(Rectified Linear Unit)는 다음과 같이 정의된다.

$$
ReLU(x) = \max(0,x)
$$

즉,

$$
ReLU(x) = \begin{cases} 0 & x\le0\\ x & x>0 \end{cases}
$$

이다.

```text
입력 < 0
→ 0

입력 > 0
→ 입력 그대로
```

양수 영역에서는

$$
\frac{d}{dx}ReLU(x)=1
$$

이므로 Sigmoid처럼 양수의 큰 영역에서 Gradient가 포화되어 작아지는 현상이 없다.

따라서 깊은 신경망에서 Vanishing Gradient 문제를 완화하는 데 도움이 된다.

또한

$$
\max(0,x)
$$

라는 단순한 연산만 필요하므로 계산도 비교적 간단하다.

---

## 11. Activation Function의 흐름

Activation Function의 변화를 개념적으로 다음과 같이 정리할 수 있다.

```text
Step Function
- Perceptron에서 사용
- Gradient 기반 Deep Network Training에 부적합

        ↓

Sigmoid
- 부드럽고 미분 가능
- Backpropagation 가능
- Saturation 영역에서 Gradient가 매우 작아짐

        ↓

ReLU
- 계산이 단순함
- 양수 영역 Gradient = 1
- Vanishing Gradient 문제 완화
```

---

## 12. MLP는 어떻게 학습하는가?

MLP를 만들었다고 해서 자동으로 좋은 Weight를 얻는 것은 아니다.

예를 들어 여러 Layer가 있다면

$$
\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3, \cdots
$$

각각의 Weight를 적절한 값으로 학습해야 한다.

즉 다음 문제가 남는다.

> 최종 Output에서 발생한 Error를 이용해 Hidden Layer의 Weight를 어떻게 수정할 것인가?

이 문제를 해결하기 위한 핵심 알고리즘이 **Backpropagation**이다.

---

## 13. Training의 전체 구조

신경망의 Training은 대략 다음 과정을 반복한다.

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Calculation
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Parameter Update
  ↓
Repeat
```

결국 Training이란

$$
\boxed{ \text{Loss가 작아지도록 Weight와 Bias를 수정하는 과정} }
$$

이다.

---

## 14. Forward Propagation

먼저 현재 Parameter를 이용하여 Prediction을 계산한다.

이를 **Forward Propagation**이라고 한다.

예를 들어 MLP가

```text
Input
↓
Linear
↓
ReLU
↓
Linear
↓
Output
```

구조라면 데이터가 앞에서 뒤로 순서대로 전달된다.

수식으로는

$$
\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1
$$

$$
\mathbf{a}_1 = ReLU(\mathbf{z}_1)
$$

$$
\hat{\mathbf{y}} = \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2
$$

와 같이 진행된다.

---

## 15. Loss

모델이 만든 Prediction

$$
\hat{y}
$$

과 실제 Target

$$
y
$$

를 비교한다.

두 값의 차이를 숫자로 표현하는 것이 **Loss Function**이다.

$$
L(y,\hat{y})
$$

Training의 목적은 이 Loss를 작게 만드는 것이다.

$$
\boxed{ \min L }
$$

---

## 16. Gradient

Loss를 줄이기 위해서는 각 Parameter를 어느 방향으로 수정해야 하는지 알아야 한다.

이를 위해 Loss를 Parameter에 대해 미분한다.

예를 들어 Weight $w$에 대한 Gradient는

$$
\frac{\partial L}{\partial w}
$$

이다.

이 값은 Weight를 조금 변화시켰을 때 Loss가 어느 방향으로 얼마나 변하는지를 나타낸다.

---

## 17. Backpropagation

**Backpropagation(역전파)**은 최종 Loss에서 시작하여 Network의 뒤쪽 Layer부터 앞쪽 Layer 방향으로 Gradient를 계산하는 방법이다.

```text
Forward

Input
  →
Hidden
  →
Output
  →
Loss


Backward

Loss
  →
Output Layer
  →
Hidden Layer
  →
Input 방향
```

여러 Layer로 이루어진 함수의 미분은 Chain Rule을 이용한다.

예를 들어

$$
x \rightarrow z_1 \rightarrow a_1 \rightarrow z_2 \rightarrow L
$$

과 같은 구조가 있다면,

앞쪽 Parameter에 대한 Gradient를 구하기 위해 중간 미분값들을 Chain Rule로 연결한다.

Backpropagation의 핵심은

> **각 Parameter가 최종 Loss에 얼마나 영향을 주었는지를 Gradient 형태로 계산하는 것**

이다.

---

## 18. PyTorch의 loss.backward()

PyTorch에서는 Backpropagation을 다음 한 줄로 실행할 수 있다.

```python
loss.backward()
```

이를 실행하면 PyTorch의 Autograd가 계산 그래프를 따라 역방향으로 이동하면서 각 Parameter에 대한 Gradient를 계산한다.

예를 들어

```python
for parameter in model.parameters():
    print(parameter.grad)
```

와 같이 각 Parameter에 계산된 Gradient를 확인할 수 있다.

즉,

```python
loss.backward()
```

는 Parameter를 실제로 수정하는 명령이 아니라

> **각 Parameter의 Gradient를 계산하는 명령**

이다.

실제 Parameter 수정은 Optimizer가 수행한다.

---

## 19. Optimizer와 Parameter Update

Gradient가 계산되었다면 Parameter를 Loss가 줄어드는 방향으로 수정해야 한다.

Gradient Descent의 기본적인 형태는 다음과 같다.

$$
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
$$

여기서 $\eta$는 Learning Rate이다.

PyTorch에서는 일반적으로 다음과 같이 사용한다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

각 단계의 의미는 다음과 같다.

```text
optimizer.zero_grad()
→ 이전 Gradient 초기화

loss.backward()
→ Backpropagation으로 Gradient 계산

optimizer.step()
→ 계산된 Gradient를 이용해 Parameter Update
```

---

## 20. Perceptron의 Weight Update

초기 Perceptron은 다음과 같은 형태로 Weight를 수정할 수 있다.

$$
w_{new} = w_{old} + \eta (y-\hat{y})x
$$

여기서

- $y$ : Target
- $\hat{y}$ : Prediction
- $\eta$ : Learning Rate
- $x$ : Input

이다.

핵심은 Prediction이 틀렸을 때 Weight를 수정한다는 것이다.

---

### Case 1. 예측이 맞은 경우

$$
y=\hat{y}
$$

이면

$$
y-\hat{y}=0
$$

이다.

따라서

$$
w_{new} = w_{old}
$$

가 된다.

예측이 맞았으므로 Weight를 수정하지 않는다.

---

### Case 2. 정답은 1인데 0으로 예측한 경우

$$
y=1,\qquad\hat{y}=0
$$

이면

$$
y-\hat{y}=1
$$

이다.

따라서 입력 $x$의 방향으로 Weight를 증가시킨다.

---

### Case 3. 정답은 0인데 1로 예측한 경우

$$
y=0,\qquad\hat{y}=1
$$

이면

$$
y-\hat{y}=-1
$$

이다.

따라서 Weight가 반대 방향으로 수정된다.

---

## 21. 숫자로 직접 계산해보기

다음 조건을 사용해보자.

```text
x = 1
η = 0.1
```

Prediction은

$$
\hat{y} = step(wx)
$$

라고 가정한다.

---

### Case 1. Prediction이 맞음

```text
w = 0.5
x = 1

wx = 0.5
ŷ = 1

y = 1
```

그러면

$$
y-\hat{y}=0
$$

이므로

$$
w_{new} = 0.5 + 0.1(0)(1) = 0.5
$$

이다.

Weight는 변하지 않는다.

---

### Case 2. 정답은 1인데 0으로 Prediction

```text
w = -0.3
x = 1

wx = -0.3
ŷ = 0

y = 1
```

Weight Update는

$$
w_{new} = -0.3 + 0.1(1-0)(1)
$$

$$
= -0.2
$$

이다.

Weight가

```text
-0.3
→
-0.2
```

로 증가한다.

---

### Case 3. 정답은 0인데 1로 Prediction

```text
w = 0.3
x = 1

wx = 0.3
ŷ = 1

y = 0
```

Weight Update는

$$
w_{new} = 0.3 + 0.1(0-1)(1)
$$

$$
= 0.2
$$

이다.

Weight가

```text
0.3
→
0.2
```

로 감소한다.

---

## 22. $(y-\hat{y})$와 $x$의 의미

Perceptron Update Rule

$$
w_{new} = w_{old} + \eta(y-\hat{y})x
$$

에서

$$
y-\hat{y}
$$

는 Prediction이 어느 방향으로 잘못되었는지를 나타낸다.

| 상황 | $y-\hat{y}$ | Weight 변화 |
| --- | --- | --- |
| Prediction이 맞음 | 0 | 변화 없음 |
| 1이어야 하는데 0 | +1 | 증가 방향 |
| 0이어야 하는데 1 | -1 | 감소 방향 |

여기에 $x$를 곱하는 이유는 해당 입력이 이번 Prediction에 얼마나 관여했는지를 반영하기 위해서이다.

예를 들어

$$
x=0
$$

이었다면

$$
wx=0
$$

이므로 해당 Weight는 이번 계산에 직접적인 영향을 주지 않는다.

Update 항 역시

$$
(y-\hat{y})x=0
$$

이 된다.

---

## 23. Perceptron과 현대 Neural Network Training의 연결

Perceptron과 현대 신경망의 학습 방식은 완전히 같은 알고리즘은 아니지만, 공통적인 아이디어가 있다.

둘 다

> **Prediction의 Error를 이용하여 Parameter를 더 좋은 방향으로 수정한다.**

는 구조를 가진다.

Perceptron에서는 비교적 직접적인 Update Rule을 사용한다.

$$
w \leftarrow w+\eta(y-\hat{y})x
$$

현대의 Neural Network에서는 Loss Function을 정의하고 Gradient를 계산한다.

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

그리고 다층 Network에서는 이 Gradient를 계산하기 위해 Backpropagation을 사용한다.

---

## 24. 전체 흐름

Perceptron에서 현대적인 MLP Training까지의 흐름을 하나로 연결하면 다음과 같다.

```text
Perceptron
│
├─ Input
├─ Weight
├─ Bias
└─ Step Function
        ↓
Single Linear Classifier
        ↓
복잡한 Non-linear 문제에 한계
        ↓
Hidden Layer 추가
        ↓
MLP
        ↓
Non-linear Activation 필요
        ↓
Sigmoid / ReLU
        ↓
Forward Propagation
        ↓
Prediction
        ↓
Loss
        ↓
Backpropagation
        ↓
Gradient
        ↓
Optimizer
        ↓
Weight / Bias Update
        ↓
Training
```

---

## 핵심 정리

| 개념 | 핵심 |
| --- | --- |
| Perceptron | $step(\mathbf{w}^T\mathbf{x}+b)$ 형태의 기본 인공 뉴런 |
| Weight | 입력이 결과에 미치는 영향을 조절하는 Parameter |
| Bias | 판단 기준의 위치를 조절하는 Parameter |
| Parameter | 모델이 Training으로 학습하는 Weight와 Bias |
| Step Function | Perceptron에서 사용하는 Threshold 기반 Activation |
| Perceptron의 한계 | 하나의 Linear Decision Boundary만 표현 가능 |
| MLP | Hidden Layer를 사용하여 여러 변환을 연결한 Neural Network |
| Non-linear Activation | 여러 Linear Layer가 하나의 Linear Transform으로 합쳐지는 것을 막음 |
| Sigmoid | 미분 가능한 부드러운 Activation Function |
| Vanishing Gradient | 깊은 Network에서 Gradient가 매우 작아질 수 있는 문제 |
| ReLU | 양수 영역 Gradient가 1이며 계산이 단순한 Activation Function |
| Forward Propagation | 현재 Parameter로 Prediction을 계산하는 과정 |
| Loss | Prediction과 Target의 차이를 수치화 |
| Gradient | Parameter 변화에 따른 Loss 변화량 |
| Backpropagation | Chain Rule을 이용해 뒤에서 앞으로 Gradient를 계산 |
| `loss.backward()` | PyTorch에서 Backpropagation을 수행하여 Gradient 계산 |
| Optimizer | Gradient를 이용하여 실제 Parameter를 Update |
| Training | Loss를 줄이도록 Weight와 Bias를 반복적으로 수정하는 과정 |

---

### 한 문장으로 정리

Perceptron은 입력을 선형 결합한 뒤 Step Function으로 판단하는 단순한 인공 뉴런이고, 이를 여러 층으로 확장한 MLP에서는 Non-linear Activation Function을 사용하여 복잡한 함수를 표현하며, **Backpropagation으로 각 Parameter의 Gradient를 계산하고 Optimizer가 Weight와 Bias를 반복적으로 수정하면서 학습한다.**
