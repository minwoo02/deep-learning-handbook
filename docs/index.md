# Deep Learning Handbook

딥러닝의 원리를 이해하고 PyTorch로 직접 구현하는 학습 교재이다.

**Theory**에서는 개념과 수학적 원리를, **Practice**에서는 모델 구현과 학습 과정을 다룬다.

[학습 로드맵](roadmap.md)

## 두 가지 탐색 기준

- **Task**: Regression / Binary Classification / Multiclass Classification
- **기술**: Model Architecture / Weight & Initialization / Loss / Gradient·Autograd / Optimizer / Normalization / Regularization / Evaluation

MLP·CNN·RNN은 모델 구조이며 Regression·Classification은 해결할 문제이다. AdamW는 Optimizer, 초기화는 Weight, clipping은 Gradient에서 다룬다.

## 학습 순서

PyTorch Basics → Regression(수동 gradient → autograd → SGD → nn.Module → Dataset/DataLoader → ANN) → Classification(XOR → Iris) → CNN → RNN

CNN·RNN 구현 실습은 준비 중이다.
