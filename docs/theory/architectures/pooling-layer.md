# Pooling Layer: 특징을 요약하고 공간 크기를 줄이는 방법

[TOC]

Pooling Layer는 Feature Map의 일정 영역을 대표값으로 요약하는 계층이다. 보통 공간 크기를 줄이는 **Downsampling**에 사용한다.

## 1. Pooling을 사용하는 이유

Feature Map의 공간 크기를 줄이면 이후 계층이 처리할 데이터가 작아진다. 계산량과 메모리 사용량을 줄이는 데 도움이 되지만, 세밀한 공간 정보도 함께 잃는다.

따라서 Pooling은 **정보를 얼마나 유지하고 얼마나 압축할지 결정하는 과정**이다.

## 2. Max Pooling과 Average Pooling

| 종류 | 계산 방법 | 남기는 정보 |
| --- | --- | --- |
| Max Pooling | 영역의 최댓값 선택 | 가장 강한 반응 |
| Average Pooling | 영역의 평균 계산 | 영역 전체의 평균 반응 |

예를 들어 다음 영역을 하나의 값으로 요약한다고 하자.

\[
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}
\]

- Max Pooling 결과는 **4**이다.
- Average Pooling 결과는 **2.5**이다.

이 예시는 연산을 설명하기 위한 계산 예시이다.

일반적인 두 연산에는 **학습하는 가중치가 없다.** 채널마다 독립적으로 적용하므로 채널 수는 유지된다.

## 3. 주요 Hyperparameter

### Kernel Size

한 번에 요약하는 영역의 크기이다.

영역이 커지면 더 넓은 범위를 하나의 값으로 압축한다. 세부 위치 정보가 중요하다면 지나친 요약이 불리할 수 있다.

### Stride

Pooling 영역이 이동하는 간격이다.

| 설정 | 영역 사이의 관계 |
| --- | --- |
| \(S < K\) | 서로 겹친다 |
| \(S = K\) | 서로 겹치지 않는다 |
| \(S > K\) | 처리 영역 사이에 빈 구간이 생길 수 있다 |

여기서 \(K\)는 Kernel 크기, \(S\)는 Stride이다. 위 비교는 Dilation이 1인 경우이다.

대표적인 설정은 **Kernel Size 2, Stride 2**이다. PyTorch의 `MaxPool2d`와 `AvgPool2d`는 Stride를 생략하면 Kernel Size를 사용한다.

### Padding

Pooling에도 Padding을 지정할 수 있다.

PyTorch에서는 기본값이 0이다. Padding을 사용하는 경우 Max Pooling은 음의 무한대, Average Pooling은 0으로 확장한다.

Average Pooling은 Padding의 0을 평균 계산에 포함하는지에 따라서도 결과가 달라질 수 있다.

## 4. 출력 크기 계산

Dilation이 1이고, 양쪽 Padding이 같으며, `ceil_mode=False`일 때 높이와 너비에 각각 다음 식을 적용한다.

\[
O =
\left\lfloor
\frac{I + 2P - K}{S}
\right\rfloor + 1
\]

| 기호 | 의미 |
| --- | --- |
| \(I\) | 입력 크기 |
| \(O\) | 출력 크기 |
| \(P\) | 한쪽 Padding 크기 |
| \(K\) | Kernel 크기 |
| \(S\) | Stride |

예를 들어 \(I=4\), \(K=2\), \(S=2\), \(P=0\)이면 다음과 같다.

\[
O = \left\lfloor\frac{4-2}{2}\right\rfloor + 1 = 2
\]

따라서 공간 크기는 \(4 \times 4\)에서 \(2 \times 2\)가 된다. 채널 수는 유지된다.

## 5. Global Average Pooling과 Flatten

Global Average Pooling은 **채널별 공간 전체의 평균**을 구한다.

입력 형태가 \((N,C,H,W)\)라면, 공간 전체를 평균한 결과는 \((N,C,1,1)\)로 표현할 수 있다. 이를 펼치면 샘플마다 \(C\)개의 값이 남는다.

PyTorch에서는 `AdaptiveAvgPool2d((1, 1))`로 이러한 공간 요약을 수행할 수 있다.

| 구분 | Flatten | Global Average Pooling |
| --- | --- | --- |
| 처리 | 차원을 펼친다 | 공간 전체를 평균한다 |
| 샘플당 값의 개수 | \(C \times H \times W\) | \(C\) |
| 공간별 값 | 순서에 따라 유지된다 | 채널별 평균으로 요약된다 |

**Flatten 자체는 값을 요약하지 않는다.** Global Average Pooling과 역할이 다르다.

## 6. 위치 변화와 정보 손실

Pooling은 작은 위치 변화에 덜 민감한 표현을 만드는 데 도움이 될 수 있다. 그러나 **입력이 이동해도 출력이 항상 같다는 뜻은 아니다.**

특징이 Pooling 영역의 경계를 넘으면 결과가 달라질 수 있다. Global Average Pooling 역시 특징의 세밀한 공간 배치를 구분하기 어렵게 만든다.

따라서 위치와 형태를 정밀하게 유지해야 하는 문제에서는 Downsampling의 정도를 신중하게 선택해야 한다. Kernel을 크게 만든다고 과적합 감소나 성능 향상이 보장되지는 않는다.

## 7. Pooling은 필수인가?

공간 크기는 Stride가 큰 Convolution으로도 줄일 수 있다.

- **Pooling**은 정해진 최댓값·평균 연산으로 요약한다.
- **Strided Convolution**은 학습하는 필터로 특징을 추출하면서 공간 크기를 조절한다.

따라서 CNN에 Pooling Layer를 반드시 넣어야 하는 것은 아니다.

## 핵심 정리

Pooling은 공간 정보를 대표값으로 요약한다. 계산 효율을 높일 수 있지만 정보 손실이 따른다.

**Max와 Average의 차이, Kernel과 Stride의 관계, Global Pooling과 Flatten의 차이**를 구분하는 것이 중요하다.

## 참고자료

- [dsaint31 — Pooling Layer](https://dsaint31.me/mkdocs_site/ML/ch14_cnn/CNN_pooling_layer/)
- [PyTorch — MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) — Max Pooling·Stride·Padding·출력 형태 확인자료
- [PyTorch — AvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) — Average Pooling과 Padding 처리 확인자료
- [PyTorch — AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html) — 출력 크기 지정과 Global Average Pooling 설명 확인자료
