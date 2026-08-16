# 📐 2.8 Affine Spaces (아핀 공간과 아핀 사상 & Ch2 총정리)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.8 & 2.9 전수 분석 & 4단계 정밀 해설 노트

## 🌐 0. 지난 노트(2.7절)와의 연결 및 빌드업: 왜 "아핀 공간"을 배우는가?

우리는 지난 2.4~2.7절까지 무조건 원점 $\mathbf{0}$ 을 지나야만 하는 벡터 공간(Vector Space)의 엄밀한 규칙 안에서 사고해 왔습니다.

하지만 현실 세계의 데이터나 머신러닝 최적화 문제, 그리고 연립방정식계 $A\mathbf{\lambda} = \mathbf{x}$ ($\mathbf{x} \neq \mathbf{0}$) 의 해집합은 원점을 지나지 않고 공중에 붕 떠서 이동된 직선, 평면, 초평면의 형태를 띱니다.

머신러닝 문헌에서는 '선형(Linear)'과 '아핀(Affine)'의 구분이 혼용되기도 하지만, 원점을 지나는 선형 부분공간의 한계를 극복하고 원점을 지나지 않는 평행이동 데이터와 신경망의 편향(Bias) 연산을 수학적으로 엄밀히 다루기 위해 등장하는 개념이 바로 아핀 공간(Affine Subspace)과 아핀 사상(Affine Mapping)입니다!

## 1. ⚔️ Section 2.8.1: Affine Subspaces (아핀 부분공간)

### 📌 1. 아핀 부분공간(Affine Subspace)의 정의 (Definition 2.25 & Eq 2.130a, 2.130b)
실수 벡터 공간 $V$ 와 선형 부분공간 $U \subseteq V$, 그리고 지지점 $\mathbf{x}_0 \in V$ 에 대해 다음과 같이 정의되는 부분집합 $L \subseteq V$ 를 아핀 부분공간(Affine Subspace) 또는 선형 다양체(Linear Manifold)라 부릅니다:

$$L = \mathbf{x}_0 + U := \{ \mathbf{x}_0 + \mathbf{u} \mid \mathbf{u} \in U \} = \{ \mathbf{v} \in V \mid \exists \mathbf{u} \in U : \mathbf{v} = \mathbf{x}_0 + \mathbf{u} \} \subseteq V \quad (\text{Eq 2.130a,b})$$

- 지지점 (Support Point / Support Vector): $\mathbf{x}_0$ 은 원점에서 아핀 공간으로 건너가는 기준 위치 벡터입니다.
- 방향 공간 (Direction Space): $U$ 는 원점을 지나는 본래의 $k$차원 선형 부분공간입니다.
- 핵심 구분: $\mathbf{x}_0 \notin U$ 이면 영벡터 $\mathbf{0} \notin L$ 이 되므로, 아핀 부분공간은 그 자체로는 원점을 포함하지 않아 선형 벡터 부분공간이 아닙니다!
- 아핀 공간의 포섭 관계 (Remark p.61): 두 아핀 공간 $L = \mathbf{x}_0 + U$ 와 $\tilde{L} = \tilde{\mathbf{x}}_0 + \tilde{U}$ 에 대해 $L \subseteq \tilde{L} \iff U \subseteq \tilde{U}$ 이고 $\mathbf{x}_0 - \tilde{\mathbf{x}}_0 \in \tilde{U}$.

### 📌 2. 매개변수 방정식 (Parametric Equation: Eq 2.131)
$k$차원 아핀 공간 $L = \mathbf{x}_0 + U$ 에서 방향 공간 $U$ 의 순서기저가 $(\mathbf{b}_1, \dots, \mathbf{b}_k)$ 일 때, $L$ 안의 모든 벡터 $\mathbf{x} \in L$ 은 매개변수 $\lambda_1, \dots, \lambda_k \in \mathbb{R}$ 로 오직 유일하게 표현됩니다:

$$\mathbf{x} = \mathbf{x}_0 + \lambda_1 \mathbf{b}_1 + \dots + \lambda_k \mathbf{b}_k \quad (\text{Eq 2.131})$$

- $\mathbf{b}_1, \dots, \mathbf{b}_k$: 방향 벡터 (Directional Vectors)
- $\lambda_1, \dots, \lambda_k$: 매개변수 (Parameters)

### 📌 3. 기하학적 차원에 따른 아핀 공간의 분류 (Example 2.26 & Figure 2.13)
- 직선 (Line: Figure 2.13): $y = \mathbf{x}_0 + \lambda \mathbf{b}_1$ ($\lambda \in \mathbb{R}$, $U = \text{span}[\mathbf{b}_1] \subseteq \mathbb{R}^n$ 인 1차원 아핀 부분공간).
- 평면 (Plane): $y = \mathbf{x}_0 + \lambda_1 \mathbf{b}_1 + \lambda_2 \mathbf{b}_2$ ($\lambda_1, \lambda_2 \in \mathbb{R}$, $U = \text{span}[\mathbf{b}_1, \mathbf{b}_2] \subseteq \mathbb{R}^n$ 인 2차원 아핀 부분공간).
- 초평면 (Hyperplane): $\mathbb{R}^n$ 공간에서 $(n-1)$차원의 아핀 부분공간 ($y = \mathbf{x}_0 + \sum_{i=1}^{n-1} \lambda_i \mathbf{b}_i$).
  - $\mathbb{R}^2$ 에서 직선은 초평면이고, $\mathbb{R}^3$ 에서 평면은 초평면입니다. (Chapter 12 Support Vector Machine 분류기 핵심 개념).

### 📌 4. 비동차 선형방정식 $A\mathbf{\lambda} = \mathbf{x}$ 와 아핀 공간의 관계 (Remark p.62)
행렬 $A \in \mathbb{R}^{m \times n}$ 과 $\mathbf{x} \in \mathbb{R}^m$ 에 대한 비동차 방정식계 $A\mathbf{\lambda} = \mathbf{x}$ 의 해집합은 공집합이거나 차원이 $n - \text{rk}(A)$ 인 $\mathbb{R}^n$ 상의 아핀 부분공간이 됩니다!

- $\mathbb{R}^n$ 상의 모든 $k$차원 아핀 부분공간은 비동차 선형방정식계 $A\mathbf{x} = \mathbf{b}$ ($A \in \mathbb{R}^{m \times n}, \mathbf{b} \in \mathbb{R}^m, \text{rk}(A) = n-k$) 의 해집합입니다.
- 동차 방정식계 $A\mathbf{x} = \mathbf{0}$ 의 해집합(선형 부분공간)은 지지점이 $\mathbf{x}_0 = \mathbf{0}$ 인 특수한 아핀 공간입니다.

## 2. ⚔️ Section 2.8.2: Affine Mappings (아핀 사상)

### 📌 1. 아핀 사상(Affine Mapping)의 정의 (Definition 2.26 & Eq 2.132, 2.133)
두 벡터 공간 $V, W$ 와 선형사상 $\Phi : V \to W$, 그리고 이동 벡터 $\mathbf{a} \in W$ 에 대해 다음과 같이 정의되는 사상을 아핀 사상(Affine Mapping)이라 부릅니다:

$$\phi : V \to W, \quad \mathbf{x} \mapsto \mathbf{a} + \Phi(\mathbf{x}) \quad (\text{Eq 2.132, 2.133})$$

- $\mathbf{a} \in W$: 평행이동 벡터 (Translation Vector).
- 합성 분해 (Composition): 모든 아핀 사상 $\phi : V \to W$ 는 선형사상 $\Phi : V \to W$ 와 $W$ 상의 평행이동 사상 $\tau : W \to W$ ($\tau(\mathbf{w}) = \mathbf{a} + \mathbf{w}$) 의 합성으로 유일하게 분해됩니다:

$$\phi = \tau \circ \Phi$$

- 아핀 사상의 핵심 성질:
  1. 두 아핀 사상의 합성 $\phi' \circ \phi$ 도 무조건 아핀 사상입니다.
  2. $\phi$ 가 전단사(Bijective)이면, 기하학적 구조(차원 및 평행 관계 Parallelism)를 불변 보존합니다.

## 📚 3. Section 2.9: Further Reading & Chapter 2 로드맵 총정리

### 📌 1. 선형대수학 참고문헌 및 추천 도서 (Strang, Axler, Golub)
- 기초 선형대수학 교재: Strang (2003), Golan (2007), Axler (2015), Liesen and Mehrmann (2015).
- 수치 선형대수학 (Numerical Linear Algebra): Stoer and Bulirsch (2002), Golub and Van Loan (2012), Horn and Johnson (2013).

### 📌 2. Chapter 2 (Linear Algebra) ➡️ Chapter 3 (Analytic Geometry) 연결 로드맵
Chapter 2에서는 내적이나 길이 개념 없이 벡터, 행렬, 선형독립, 기저, 사상 등의 대수적 구조를 정립했습니다. 
이어지는 Chapter 3 (내적 공간과 분석 기하학) 에서는 내적(Inner Product)과 노름(Norm)을 도입하여:
- 벡터의 길이(Length), 각도(Angle), 거리(Distance)를 수학적으로 정의합니다.
- 직교 정사영(Orthogonal Projection)을 유도하고, 이는 Chapter 9 선형 회귀(Linear Regression) 및 Chapter 10 주성분 분석(PCA) 의 핵심 수학적 토대가 됩니다!

## 🚀 4. 4단계 실전 AI / 머신러닝 연결고리
- 인공신경망 Linear Layer ($Y = W X + b$) & SVM Hyperplane:
  - 퍼셉트론과 딥러닝 레이어의 입력 $X$ 에 가중치 $W$ 를 곱하고 편향(Bias) $b$ 를 더하는 행위는 수학적으로 완벽한 아핀 사상(Affine Mapping)입니다.
  - Support Vector Machine(SVM)의 클래스 분류 경계면 또한 지지점 $\mathbf{x}_0$ 과 차원 $(n-1)$ 의 아핀 초평면(Affine Hyperplane)을 찾는 최적화 알고리즘입니다.
