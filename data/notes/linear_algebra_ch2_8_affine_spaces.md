# 📐 2.8 Affine Spaces (아핀 공간과 아핀 사상)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.8 전수 분석 & 4단계 정밀 해설 노트

## 🌐 0. 지난 노트(2.7절)와의 연결 및 빌드업: 왜 "아핀 공간"을 배우는가?

우리는 지난 2.4~2.7절까지 무조건 원점 $\mathbf{0}$ 을 지나야만 하는 벡터 공간(Vector Space)의 규칙 안에서 사고해 왔습니다.

하지만 현실의 데이터나 연립방정식계 $A\mathbf{x} = \mathbf{b}$ ($\mathbf{b} \neq \mathbf{0}$) 의 해집합은 원점을 지나지 않고 공중에 붕 떠서 이동된 직선, 평면, 초평면의 형태를 띱니다.

원점을 지나는 벡터 공간의 엄밀한 수학적 한계를 극복하고, 원점을 지나지 않는 평행이동 데이터와 신경망의 편향(Bias) 연산을 자유롭게 다루기 위해 등장하는 개념이 바로 아핀 공간(Affine Subspace)과 아핀 사상(Affine Mapping)입니다!

## 1. ⚔️ Section 2.8.1: Affine Subspaces (아핀 부분공간)

### 📌 1. 아핀 부분공간(Affine Subspace)의 정의 (Definition 2.25 & Eq 2.130)
벡터 공간 $V$ 와 $V$ 의 선형 부분공간 $U \subseteq V$, 그리고 고정된 지지점 $\mathbf{x}_0 \in V$ 에 대해 다음과 같이 정의되는 부분집합 $L \subseteq V$ 를 아핀 부분공간(Affine Subspace) 또는 선형 다양체(Linear Manifold)라 부릅니다:

$$L = \mathbf{x}_0 + U := \{ \mathbf{x}_0 + \mathbf{u} \mid \mathbf{u} \in U \} \subseteq V \quad (\text{Eq 2.130})$$

- 지지점 (Support Point / Support Vector): $\mathbf{x}_0$ 은 원점에서 아핀 공간으로 건너가는 기준 위치 벡터입니다.
- 방향 공간 (Direction Space): $U$ 는 원점을 지나는 본래의 $k$차원 선형 부분공간입니다.

### 📌 2. 매개변수 방정식 (Parametric Equation: Eq 2.131)
$k$차원 아핀 공간 $L = \mathbf{x}_0 + U$ 에서 방향 공간 $U$ 의 순서기저가 $(\mathbf{b}_1, \dots, \mathbf{b}_k)$ 일 때, $L$ 안의 모든 벡터 $\mathbf{x} \in L$ 은 매개변수 $\lambda_1, \dots, \lambda_k \in \mathbb{R}$ 로 오직 유일하게 표현됩니다:

$$\mathbf{x} = \mathbf{x}_0 + \lambda_1 \mathbf{b}_1 + \dots + \lambda_k \mathbf{b}_k \quad (\text{Eq 2.131})$$

### 📌 3. 기하학적 차원에 따른 아핀 공간의 분류 (Example 2.26)
- 직선 (Line): $y = \mathbf{x}_0 + \lambda \mathbf{b}_1$ (1차원 아핀 부분공간).
- 평면 (Plane): $y = \mathbf{x}_0 + \lambda_1 \mathbf{b}_1 + \lambda_2 \mathbf{b}_2$ (2차원 아핀 부분공간).
- 초평면 (Hyperplane): $\mathbb{R}^n$ 공간에서 $(n-1)$차원의 아핀 부분공간.

### 📌 4. 비동차 선형방정식 $A\mathbf{x} = \mathbf{b}$ 의 해집합의 본질 (Remark p.54)
행렬 $A \in \mathbb{R}^{m \times n}$ 과 $\mathbf{b} \in \mathbb{R}^m$ ($\mathbf{b} \neq \mathbf{0}$) 에 대한 비동차 방정식계 $A\mathbf{x} = \mathbf{b}$ 의 해집합은 다음과 같이 주어집니다:

$$\text{Solution Set} = \mathbf{x}_p + \text{ker}(A) = \{ \mathbf{x}_p + \mathbf{x}_h \mid A\mathbf{x}_h = \mathbf{0} \}$$

- 특수해 $\mathbf{x}_p$ (Particular Solution): 아핀 공간의 지지점 $\mathbf{x}_0$ 역 할을 수행.
- 영공간 $\text{ker}(A)$ (Null Space): 아핀 공간의 방향 공간 $U$ 역할을 수행.
- 따라서 해집합의 차원은 $\text{dim}(L) = n - \text{rk}(A)$ 인 아핀 공간이 됩니다!

## 2. ⚔️ Section 2.8.2: Affine Mappings (아핀 사상)

### 📌 1. 아핀 사상(Affine Mapping)의 정의 (Definition 2.26 & Eq 2.133)
두 벡터 공간 $V, W$ 와 선형사상 $\Phi : V \to W$, 그리고 이동 벡터 $\mathbf{a} \in W$ 에 대해 다음과 같이 정의되는 사상을 아핀 사상(Affine Mapping)이라 부릅니다:

$$\phi : V \to W, \quad \mathbf{x} \mapsto \mathbf{a} + \Phi(\mathbf{x}) \quad (\text{Eq 2.133})$$

- 행렬 표현: $\mathbf{y} = A\mathbf{x} + \mathbf{a}$ ($A$ 는 선형변환 행렬, $\mathbf{a}$ 는 평행이동 벡터 / Translation Vector).

## 🚀 3. 4단계 실전 AI / 머신러닝 연결고리
- 인공신경망 Linear Layer ($Y = W X + b$) & SVM Hyperplane:
  - 퍼셉트론과 딥러닝 레이어의 입력 $X$ 에 가중치 $W$ 를 곱하고 편향(Bias) $b$ 를 더하는 행위는 수학적으로 완벽한 아핀 사상(Affine Mapping)입니다.
  - Support Vector Machine(SVM)의 클래스 분류 경계면 또한 지지점 $\mathbf{x}_0$ 과 차원 $(n-1)$ 의 아핀 초평면(Affine Hyperplane)을 찾는 최적화 알고리즘입니다.
