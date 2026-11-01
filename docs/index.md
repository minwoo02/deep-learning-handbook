# Deep Learning Handbook

DL_a = **Theory**, DL_c = **Practice**. Velog의 설명과 코드를 보존하고 개념 분류와 학습 순서를 연결한 개인 학습 교재입니다.

[학습 로드맵](roadmap.md) · [전체 원문·검증 기록](sources.md)

## 두 가지 탐색 기준

- **Task**: Regression / Binary Classification / Multiclass Classification
- **기술**: Model Architecture / Weight & Initialization / Loss / Gradient·Autograd / Optimizer / Normalization / Regularization / Evaluation

MLP·CNN·RNN은 모델 구조이며 Regression·Classification은 해결할 문제입니다. AdamW는 Optimizer에, 초기화는 Weight에, clipping은 Gradient에 둡니다.

## 학습 순서

PyTorch Basics → Regression(수동 gradient → autograd → SGD → nn.Module → Dataset/DataLoader → ANN) → Classification(XOR → Iris) → CNN → RNN

CNN·RNN은 기존의 독립 실습 글이 확인되지 않아 **학습 예정**으로 표시합니다. 원문에 없는 구현이나 성능 수치는 추가하지 않았습니다.
