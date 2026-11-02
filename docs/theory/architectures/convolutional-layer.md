# Convolutional Layer: Kernel·Stride·Padding과 출력 크기

[TOC]

Convolutional Layer는 작은 필터를 입력 위로 이동시키며 특징을 추출하는 계층이다. 출력인 **Feature Map**에는 특징의 반응 강도와 공간적 위치가 함께 표현된다.

## 1. Kernel에서 Feature Map까지

Kernel은 입력의 일부 영역에 가중치를 적용한다. 같은 가중치를 위치마다 공유하면서 연산을 반복하면 하나의 Feature Map이 만들어진다.

여러 필터를 사용하면 서로 다른 특징을 표현하는 여러 출력 채널을 얻는다. 계층을 쌓으면 앞선 특징들을 조합해 더 복잡한 표현을 학습한다.

## 2. 입력 채널과 출력 채널

일반적인 합성곱(`groups=1`)에서는 필터 하나가 모든 입력 채널을 함께 처리한다.

RGB 이미지라면 입력 채널은 3개이다. 필터의 공간 크기가 \(3 \times 3\)이면, 출력 채널 하나를 만드는 가중치는 \(3 \times 3 \times 3\)개이다.

**입력 채널 수는 필터의 깊이와, 필터 개수는 출력 채널 수와 연결된다.** Grouped·Depthwise Convolution에서는 연결 방식이 달라진다.

## 3. Convolution과 Cross-Correlation

수학적 Convolution은 Kernel을 뒤집어 적용하지만, 딥러닝에서는 보통 뒤집지 않는 **Cross-Correlation**을 사용한다.

PyTorch의 `Conv2d`도 Cross-Correlation으로 정의된다. 따라서 계층 이름이 Convolution이라고 해서 실제 계산에서 Kernel을 반드시 뒤집는 것은 아니다.

## 4. 주요 Hyperparameter

| 설정 | 역할 |
| --- | --- |
| Kernel Size | 한 번에 처리하는 공간 영역의 크기 |
| Stride | Kernel이 이동하는 간격 |
| Padding | 입력 가장자리에 추가하는 영역 |
| Dilation | Kernel 원소 사이의 간격 |
| Output Channels | 출력 특징 채널의 개수 |

### Kernel Size

다른 조건이 같으면 Kernel이 커질수록 한 출력이 직접 참조하는 영역과 가중치 수가 증가한다.

Kernel 크기가 반드시 홀수여야 하는 것은 아니다.

### Stride

Stride는 **이동 간격**이다. 1이면 한 칸씩, 2이면 두 칸씩 이동한다.

다른 조건이 같으면 Stride를 키울수록 출력의 공간 크기가 작아진다. 인접한 연산 영역은 서로 겹칠 수 있다.

### Padding

Padding은 가장자리의 처리와 출력 크기를 조절한다. Zero Padding은 주변을 0으로 채우는 방식이다.

TensorFlow의 `padding="same"`은 **Stride가 1일 때** 입력과 같은 공간 크기를 유지한다. Stride가 커지면 같은 크기가 보장되지 않는다.

## 5. 출력 크기 계산

높이와 너비에 각각 다음 식을 적용한다. 양쪽 Padding 크기가 같은 경우이다.

\[
O =
\left\lfloor
\frac{I + 2P - D(K-1) - 1}{S}
\right\rfloor + 1
\]

| 기호 | 의미 |
| --- | --- |
| \(I\) | 입력 크기 |
| \(O\) | 출력 크기 |
| \(K\) | Kernel 크기 |
| \(P\) | 한쪽 Padding 크기 |
| \(S\) | Stride |
| \(D\) | Dilation |

Dilation이 1이면 다음처럼 단순해진다.

\[
O =
\left\lfloor
\frac{I + 2P - K}{S}
\right\rfloor + 1
\]

### 계산 예시

입력이 \(7 \times 7\)이고 \(K=3\), \(P=0\), \(S=2\), \(D=1\)이면 다음과 같다.

\[
O = \left\lfloor\frac{7-3}{2}\right\rfloor + 1 = 3
\]

따라서 출력의 공간 크기는 \(3 \times 3\)이다. 출력 채널을 2개로 설정하면 배치 차원을 제외한 PyTorch 출력 형태는 **(2, 3, 3)**이다.

이 예시는 출력 크기 식을 설명하기 위한 계산 예시이며, 학습 실행 결과는 아니다.

## 6. 여러 형태의 Convolution

| 종류 | 핵심 차이 |
| --- | --- |
| Depthwise | 입력 채널별로 독립적으로 처리한다 |
| Pointwise | \(1 \times 1\) Kernel을 사용한다 |
| Depthwise Separable | Depthwise와 Pointwise를 결합한다 |
| Dilated | Kernel 원소 간격을 늘린다 |
| Transposed | 학습 가능한 공간 해상도 확대 등에 사용한다 |

**Transposed Convolution은 원래 입력을 정확히 복원하는 합성곱의 역연산이 아니다.**

## 핵심 정리

Convolutional Layer를 이해할 때는 **공간 크기와 채널 수를 분리해서 생각하는 것**이 중요하다.

공간 크기는 Kernel·Stride·Padding·Dilation으로 결정되고, 출력 채널 수는 필터 개수로 결정된다.

## 참고자료

- [dsaint31 — Convolutional Layer](https://dsaint31.me/mkdocs_site/ML/ch14_cnn/CNN_convolutional_layer/)
- [PyTorch — Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — 채널·연산 방식·출력 크기 식 확인자료
- [TensorFlow — Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) — `same` Padding의 조건 확인자료
- [PyTorch — ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) — Transposed Convolution의 의미 확인자료
