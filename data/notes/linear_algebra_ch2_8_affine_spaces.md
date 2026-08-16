# 📐 2.8 Affine Spaces (아핀 공간과 아핀 사상)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.8 원문 완전 대조 스토리텔링 노트

---

## 1. 🌐 서론: 왜 "아핀 공간(Affine Space)"을 다루는가?

2.4절에서 부분공간(Subspace)은 반드시 원점 $\mathbf{0}$ 을 지나야 한다고 배웠습니다.
하지만 현실 세상의 데이터나 초평면(Hyperplane)은 원점에서 벗어나 공중에 떠 있는 경우가 대부분입니다!

원점을 지나지 않는 이동된 공간(Offset Spaces) 및 평행이동(Translation)이 포함된 사상을 수학적으로 엄밀히 다루는 개념이 바로 아핀 공간(Affine Space)과 아핀 사상(Affine Mapping)입니다.

---

## 2. ⚔️ Section 2.8.1: Affine Subspaces (아핀 부분공간)

### 📌 1. 아핀 부분공간의 정의 (Definition 2.25 & Eq 2.130)
벡터 공간 $V$, 지지점(Support Point) $\mathbf{x}_0 \in V$, 그리고 부분공간(Direction Space) $U \subseteq V$ 에 대해:

$$L = \mathbf{x}_0 + U := \{\mathbf{x}_0 + \mathbf{u} \mid \mathbf{u} \in U\} \subseteq V \quad (2.130)$$

를 $V$ 의 아핀 부분공간(Affine Subspace) 또는 선형 다형체(Linear Manifold)라 부릅니다.

- 원점 배제 성질: $\mathbf{x}_0 \notin U$ 이면 $L$ 은 영벡터 $\mathbf{0}$ 을 포함하지 않으므로 더 이상 (선형) 벡터 부분공간이 아닙니다.
- 매개변수 방정식 (Parametric Equation: Eq 2.131):
  $$\mathbf{x} = \mathbf{x}_0 + \lambda_1 \mathbf{b}_1 + \dots + \lambda_k \mathbf{b}_k \quad (\mathbf{b}_1, \dots, \mathbf{b}_k \text{ 는 방향 공간 } U \text{ 의 기저})$$

---

### 📌 2. 아핀 공간의 대표적 종류 (Example 2.26 & Figure 2.13)
- 1차원 아핀 공간 (Line 직선): $\mathbf{y} = \mathbf{x}_0 + \lambda \mathbf{b}_1$ (지지점 $\mathbf{x}_0$ 와 방향 벡터 $\mathbf{b}_1$)
- 2차원 아핀 공간 (Plane 평면): $\mathbf{y} = \mathbf{x}_0 + \lambda_1 \mathbf{b}_1 + \lambda_2 \mathbf{b}_2$
- $(n-1)$차원 아핀 공간 (Hyperplane 초평면): $\mathbf{y} = \mathbf{x}_0 + \sum_{i=1}^{n-1} \lambda_i \mathbf{b}_i$
  - $\mathbb{R}^2$ 에서 직선은 초평면이고, $\mathbb{R}^3$ 에서 평면은 초평면입니다.
  - SVM(Support Vector Machine)의 분류 경계면이 대표적인 아핀 초평면입니다!

---

### 📌 3. 비동차 선형계 $A\mathbf{x} = \mathbf{b}$ 와 아핀 공간의 본질적 관계 (Remark p.62)
- 동차 방정식계 $A\mathbf{x} = \mathbf{0}$ 의 해집합: 원점을 지나는 벡터 부분공간(Kernel).
- 비동차 방정식계 $A\mathbf{x} = \mathbf{b} \ (\mathbf{b} \neq \mathbf{0})$ 의 해집합:
  - 특수해 $\mathbf{x}_p$ 만큼 평행이동된 차원 $(n - \text{rk}(A))$ 의 아핀 부분공간 $L = \mathbf{x}_p + \text{ker}(A)$!

---

## 3. ⚔️ Section 2.8.2: Affine Mappings (아핀 사상)

### 📌 1. 아핀 사상(Affine Mapping)의 정의 (Definition 2.26)
두 벡터 공간 $V, W$ 간의 선형사상 $\Phi : V \to W$ 와 이동 벡터 $\mathbf{a} \in W$ 에 대해:

$$\psi : V \to W, \quad \mathbf{x} \mapsto \Phi(\mathbf{x}) + \mathbf{a} \quad (\text{행렬 표기: } \psi(\mathbf{x}) = A\mathbf{x} + \mathbf{a})$$

를 아핀 사상(Affine Mapping)이라 부릅니다.

- 본질적 구조: [선형 사상(Linear Transformation $A\mathbf{x}$)] + [평행 이동(Translation $\mathbf{a}$)]
- 인공지능 퍼셉트론 / 인공신경망 레이어:
  $$\mathbf{y} = W \mathbf{x} + \mathbf{b}$$
  신경망의 가중치 곱 $W\mathbf{x}$ 와 편향(Bias) 더하기 $+\mathbf{b}$ 구조가 바로 수학적으로 완전한 아핀 사상(Affine Mapping)입니다!
