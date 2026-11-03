# CNN: 이미지의 공간 구조를 활용하는 신경망

[TOC]

CNN(Convolutional Neural Network)은 이미지처럼 격자 구조를 가진 데이터를 처리하는 신경망이다. 핵심은 **가까운 영역을 연결하고, 같은 필터를 여러 위치에서 공유하는 것**이다.

## 1. 이미지에 CNN을 사용하는 이유

이미지에서는 주변 픽셀의 관계와 반복되는 패턴이 중요하다. CNN은 이러한 특성을 활용해 특징을 추출한다.

| 핵심 개념 | 의미 |
| --- | --- |
| Locality | 가까운 픽셀 사이의 관계를 우선 고려한다 |
| Parameter Sharing | 같은 필터를 여러 위치에 적용한다 |
| Hierarchy | 낮은 단계의 특징을 조합해 복잡한 특징을 만든다 |

이는 이미지의 모든 위치에 별도의 연결을 두는 완전연결 계층과 다른 설계이다.

## 2. Convolutional Layer와 Feature Map

Convolutional Layer는 작은 Kernel을 이동시키며 입력의 국소 영역을 처리한다. 그 결과인 **Feature Map**은 필터가 반응한 위치와 강도를 나타낸다.

필터의 가중치는 학습 과정에서 조정된다. 여러 필터를 사용하면 서로 다른 특징을 표현할 수 있다.

## 3. Locality: 가까운 영역부터 살펴보기

작은 Kernel은 입력 전체가 아닌 주변 영역과 연결된다. 이러한 **희소 연결**은 불필요한 연결을 줄이고 국소 패턴에 집중하게 한다.

다만 멀리 떨어진 정보가 중요하지 않다는 뜻은 아니다. 계층을 쌓으며 더 넓은 영역의 정보를 결합한다.

## 4. 위치가 달라도 같은 필터를 사용하는 이유

눈이나 경계선 같은 패턴은 이미지의 다른 위치에서도 나타난다. CNN은 같은 가중치를 공유해 여러 위치에서 해당 패턴을 찾는다.

이는 위치마다 별도의 필터를 학습하는 것보다 Parameter를 절약하는 데 도움이 된다.

## 5. Receptive Field와 계층적 특징

**Receptive Field(수용영역)**는 특정 출력에 영향을 주는 입력의 공간적 범위이다.

계층이 깊어지면 더 넓은 영역을 고려할 수 있다. 낮은 계층에서 얻은 경계·질감 등의 특징을 조합해 복잡한 형태를 표현한다.

## 6. Pooling의 역할과 주의점

Pooling은 주변 값을 대표값으로 요약한다.

- **Max Pooling**: 영역에서 가장 큰 값을 선택한다.
- **Average Pooling**: 영역의 평균값을 계산한다.

공간 크기를 줄일 수 있지만 세밀한 위치 정보도 손실된다. 지나치게 큰 영역을 요약하거나 반복해서 축소하면 중요한 정보가 사라질 수 있다.

## 7. 이동 등변성과 이동 불변성

두 개념은 구분해야 한다.

| 구분 | 입력의 패턴이 이동했을 때 |
| --- | --- |
| 이동 등변성(Equivariance) | 출력의 특징 위치도 함께 이동한다 |
| 이동 불변성(Invariance) | 출력 표현이 이동의 영향을 받지 않는다 |

합성곱은 이동 등변성을 갖는 연산이다. Pooling은 작은 위치 변화에 덜 민감한 표현을 만드는 데 도움이 된다.

그러나 실제 CNN에는 경계 처리·Stride 등이 포함되므로, **모든 이동에서 출력이 완전히 같다고 단정할 수 없다.** Pooling도 CNN의 필수 구성요소는 아니다.

## 핵심 정리

CNN은 국소 연결과 가중치 공유를 통해 공간적 패턴을 효율적으로 학습한다. 깊은 계층은 이러한 특징을 조합하며, Pooling은 공간 정보를 요약하는 대신 일부 위치 정보를 잃는다.

## 관련 문서

- [Convolutional Layer: Kernel·Stride·Padding과 출력 크기](convolutional-layer.md)
- [Pooling Layer: 특징을 요약하고 공간 크기를 줄이는 방법](pooling-layer.md)

- [ANN·CNN·RNN 비교](neural-network-architectures.md)
- [CNN 구현 실습 · 준비 중](../../practice/stages/cnn.md)

## 참고자료

- [dsaint31 — Convolutional Neural Network, CNN](https://dsaint31.me/mkdocs_site/ML/ch14_cnn/CNN_intro/) — 정리의 바탕이 된 글이다.
- [Goodfellow·Bengio·Courville — Deep Learning, Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html) — 이동 등변성과 Pooling 설명을 보충한 자료이다.
