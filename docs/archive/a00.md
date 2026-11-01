# [DL_a] 딥러닝 기초 개념 정리 — 전체 목차 및 학습 가이드

원문: [[DL_a] 딥러닝 기초 개념 정리 — 전체 목차 및 학습 가이드](https://velog.io/@haru02/DLa-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EC%B4%88-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC-%EC%A0%84%EC%B2%B4-%EB%AA%A9%EC%B0%A8-%EB%B0%8F-%ED%95%99%EC%8A%B5-%EA%B0%80%EC%9D%B4%EB%93%9C) · 수집: 2026-11-01 · 원문 보존 이관본

[원본 목록·검증 범위](../sources.md)

---

Machine Learning과 Deep Learning을 공부하면서 이해한 개념들을 체계적으로 정리한 시리즈이다.

단순히 알고리즘의 정의와 수식을 암기하기보다, **왜 이러한 개념이 필요한지, 서로 어떻게 연결되는지, 실제 모델 학습 과정에서 어떤 역할을 하는지** 이해하는 것을 목표로 한다.

이 시리즈에서는 머신러닝과 딥러닝의 기본 개념부터 인공신경망의 동작 원리, 최적화, 확률·통계, 모델 성능평가, Cross Validation과 Hyperparameter Tuning까지 순서대로 다룬다.

실제 모델 구현 및 코드 중심의 실습은 별도의 **[👾DL_c] 딥러닝 구현 및 실습** 시리즈에서 정리한다.

---

## Part 1. 머신러닝과 딥러닝의 개요

머신러닝과 딥러닝의 기본적인 개념을 이해하고, 대표적인 Learning Paradigm과 Task를 살펴본다.

1. [딥러닝의 역사: AI, ML, DL은 어떻게 발전했는가?](../theory/foundations/a01.md)
2. [Supervised Learning(지도학습) 완전 정리: Classification·Regression부터 Object Detection·Segmentation까지](../theory/foundations/a02.md)
3. [Classification과 Regression](../theory/foundations/a03.md)
4. [ANN • DNN • CNN • RNN 개념 및 핵심 차이 총정리](../theory/architectures/a04.md)

**핵심 학습 내용**

- AI, Machine Learning, Deep Learning의 관계
- Supervised Learning과 대표적인 Task
- Classification과 Regression의 차이
- ANN, DNN, CNN, RNN의 구조적 차이

---

## Part 2. 인공신경망의 구조와 동작 원리

Perceptron에서 시작하여 Multi-Layer Perceptron(MLP)으로 확장되는 과정을 살펴보고, 신경망이 비선형 관계를 학습하는 원리를 이해한다.

5. [Perceptron에서 MLP와 Backpropagation까지: 신경망은 어떻게 학습하는가?](../theory/architectures/a05.md)
6. [TLU와 Perceptron의 기하학적 이해](../theory/architectures/a06.md)
7. [XOR로 이해하는 Perceptron의 한계와 MLP](../theory/architectures/a07.md)
8. [딥러닝을 공간 변환 관점에서 이해하기](../theory/architectures/a08.md)
9. [Activation Function](../theory/architectures/a09.md)

**핵심 학습 내용**

- Perceptron의 구조와 Decision Boundary
- Linear Separability와 XOR Problem
- MLP와 Non-linear Transformation
- Forward Propagation과 Backpropagation
- Activation Function의 필요성과 역할

---

## Part 3. 모델 학습과 최적화의 기초

신경망에서 학습되는 Parameter와 학습 과정을 제어하는 Hyperparameter를 구분하고, Loss Function을 최소화하는 최적화의 기본 원리를 살펴본다.

10. [Parameter vs Hyperparameter: 모델이 학습하는 값과 사람이 정하는 값](../theory/weight/a10.md)
11. [Gradient Descent와 Learning Rate: Linear Regression과 Logistic Regression까지](../theory/gradients/a11.md)

**핵심 학습 내용**

- Parameter와 Hyperparameter의 차이
- Weight와 Bias의 역할
- Gradient와 Gradient Descent
- Learning Rate와 Parameter Update
- Linear Regression과 Logistic Regression의 최적화

---

## Part 4. 확률·통계와 손실함수의 수학적 이해

확률분포의 기초부터 Likelihood, Bayes' Theorem, MLE·MAP까지 살펴보고, 딥러닝에서 사용하는 Loss Function이 어떤 확률적 가정으로부터 유도될 수 있는지 이해한다.

12. [확률분포 완전 정리: PMF·PDF·CDF의 개념과 차이](../theory/foundations/a12.md)
13. [베르누이 분포 완전 정리: PMF·기댓값·분산과 Binary Classification](../theory/foundations/a13.md)
14. [범주형 분포 완전 정리: One-Hot Encoding부터 PMF·Multinomial·Cross-Entropy까지](../theory/foundations/a14.md)
15. [라플라스 분포 완전 정리: PDF·통계적 특성과 MAE·L1 정규화](../theory/foundations/a15.md)
16. [Likelihood(우도) 완전 정리: Probability와의 차이부터 MLE·MAP·Log-Likelihood까지](../theory/foundations/a16.md)
17. [베이즈 정리(Bayes' Theorem) 완전 정리: 사전확률·우도·사후확률부터 MLE와 MAP 추정까지](../theory/foundations/a17.md)
18. [딥러닝 손실함수의 확률적 이해: MLE·NLL부터 Cross-Entropy, MSE, MAE, L1·L2 정규화까지](../theory/loss/a18.md)

**핵심 학습 내용**

- Probability Distribution과 PMF·PDF·CDF
- Bernoulli·Categorical·Laplace Distribution
- Probability와 Likelihood의 차이
- Prior, Likelihood, Posterior
- MLE와 MAP Estimation
- Negative Log-Likelihood와 Cross-Entropy
- MSE, MAE, L1·L2 Regularization

---

## Part 5. 모델 성능평가·Cross Validation·Hyperparameter Tuning

학습된 모델의 예측 성능을 평가하는 지표를 이해하고, 데이터 분할 및 Cross Validation을 통해 Generalization Performance를 추정하는 방법을 살펴본다.

이후 Validation Performance를 이용하여 모델과 Hyperparameter를 선택하는 과정을 이해한다.

19. [머신러닝 분류 성능평가 완전 정리: Confusion Matrix부터 Accuracy·Precision·Recall·F1-score까지](../theory/evaluation/a20.md)
20. [Cross Validation 완전 정리: Hold-out부터 K-Fold·Stratified K-Fold와 모델 일반화 성능 평가까지](../theory/evaluation/a22.md)
21. [Hyperparameter Tuning](../theory/evaluation/a23.md)

**핵심 학습 내용**

- Confusion Matrix와 TP·TN·FP·FN
- Accuracy, Precision, Recall, F1-score
- Class Imbalance와 모델 성능 해석
- Training·Validation·Test Set의 역할
- Hold-out Validation과 K-Fold Cross Validation
- Stratified K-Fold와 Class Distribution
- Generalization Performance와 Data Leakage
- Cross Validation을 이용한 Model Selection
- Hyperparameter Search와 Tuning

---

## 전체 학습 흐름

```text
Machine Learning & Deep Learning
              │
              ▼
     Supervised Learning
              │
              ▼
   Classification / Regression
              │
              ▼
       Neural Network
              │
              ▼
   Perceptron → MLP → DNN
              │
              ▼
   Forward / Backpropagation
              │
              ▼
   Gradient-based Optimization
              │
              ▼
   Probability & Statistics
              │
              ▼
     Likelihood / MLE / MAP
              │
              ▼
         Loss Function
              │
              ▼
      Model Evaluation
              │
              ▼
       Cross Validation
              │
              ▼
     Hyperparameter Tuning
              │
              ▼
    Generalization Evaluation
```

이 시리즈는 개별 개념을 독립적으로 학습할 수 있도록 구성되어 있으며, 전체 순서대로 읽으면 머신러닝의 기본 원리에서 딥러닝의 학습 및 평가 과정까지 연결하여 이해할 수 있다.

특히 모델 학습 이후에는 **어떤 지표로 평가할 것인가(Model Evaluation), 어떤 데이터 분할로 검증할 것인가(Cross Validation), 어떤 설정을 선택할 것인가(Hyperparameter Tuning)** 라는 세 가지 문제를 구분하여 학습할 수 있도록 구성하였다.

---

## 다음 단계: 딥러닝 구현 및 실습

기초 개념을 이해한 후에는 실제 코드를 통해 모델을 구현하고 학습 과정을 확인하는 것이 중요하다.

별도의 **[👾DL_c] 딥러닝 구현 및 실습** 시리즈에서는 다음 내용을 다룬다.

- scikit-learn 기반 Perceptron과 MLP 구현
- PyTorch의 `nn.Module`, `Dataset`, `DataLoader`
- Linear Regression과 ANN 구현
- XOR Problem과 Iris Classification
- Weight Initialization
- Learning Rate Scheduling과 AdamW Optimizer
- Batch Normalization
- Gradient Clipping

**[→ 딥러닝 구현 및 실습 시리즈 바로가기](https://velog.io/@haru02/series/DL-PyTorch%EB%A1%9C-%EB%B0%B0%EC%9A%B0%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%8B%A4%EC%8A%B5)**

---

> 이 시리즈는 머신러닝과 딥러닝을 학습하면서 정리한 개인 학습 노트이다. 개념에 대한 이해가 깊어지거나 새로운 내용을 학습하면 기존 글도 지속적으로 보완할 예정이다.
