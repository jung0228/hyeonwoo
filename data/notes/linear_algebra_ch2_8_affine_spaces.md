# 📐 2.8 Affine Spaces (아핀 공간과 아핀 사상)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.8 원문 완전 복사 & 심층 가공 노트**

---

## 📖 Part 1. MML 교재 원문 (Textbook Original Text)

### 1. Definition 2.25 (Affine Subspace) & Parametric Equation
```text
Definition 2.25 (Affine Subspace).
Let V be a vector space, x0 in V and U \subseteq V a subspace. Then the subset
L = x0 + U := {x0 + u : u in U} = {v in V | \exists u in U : v = x0 + u} \subseteq V  (Eq 2.130)
is called affine subspace or linear manifold of V. U is called direction space, and x0 is called support point.

Parametric Equation:
Consider a k-dimensional affine space L = x0 + U of V. If (b1, ..., bk) is an ordered basis of U, then every element x in L can be uniquely described as
x = x0 + \lambda_1 b1 + ... + \lambda_k bk  (Eq 2.131)
where \lambda_1, ..., \lambda_k in R.
```

### 2. Example 2.26 (Lines, Planes, Hyperplanes) & Remark (Inhomogeneous Linear Systems)
```text
Example 2.26 (Affine Subspaces):
- Line: y = x0 + \lambda b1 (1-dimensional affine subspace).
- Plane: y = x0 + \lambda_1 b1 + \lambda_2 b2 (2-dimensional affine subspace).
- Hyperplane: (n-1)-dimensional affine subspace in R^n.

Remark (Inhomogeneous systems and affine subspaces):
For A in R^{m \times n} and x in R^m, the solution of Ax = b is either empty or an affine subspace of R^n of dimension (n - rk(A)).
In particular, the solution of inhomogeneous system Ax = b (b != 0) is a special affine space with support point x0 = x_p.
```

### 3. Definition 2.26 (Affine Mapping)
```text
Definition 2.26 (Affine Mapping).
For two vector spaces V, W, a linear mapping \Phi : V -> W, and a in W, the mapping
\phi : V -> W,  x |-> a + \Phi(x)  (Eq 2.133)
is an affine mapping from V to W. The vector a is called the translation vector of \phi.
```

---

## 🧠 Part 2. 한국어 정밀 가공 & 개념 설명 (Deep Interpretation)

### 📌 1. [개념 정의] 아핀 공간(Affine Subspace)과 아핀 사상(Affine Mapping)이란 무엇인가?
- **아핀 공간 (Affine Subspace)**: 원점 $\mathbf{0}$ 을 지나야만 하는 부분공간(Subspace)의 속박에서 벗어나, 지지점 $\mathbf{x}_0$ 만큼 공중에 붕 떠서 이동된 공간($L = \mathbf{x}_0 + U$)입니다.
- **아핀 사상 (Affine Mapping)**: 단순 회경/축소 등의 선형사상 $A\mathbf{x}$ 에 평행이동(Translation) $+\mathbf{a}$ 가 결합된 사상입니다.

### 📌 2. [존재 이유 & 직관] 비동차 선형계 $A\mathbf{x} = \mathbf{b}$ 의 해집합의 본질
- **$A\mathbf{x} = \mathbf{0}$ 동차 방정식의 해**: 원점을 무조건 포함하므로 **선형 부분공간(Kernel)**이 됩니다.
- **$A\mathbf{x} = \mathbf{b}$ 비동차 방정식의 해**: 특수해 $\mathbf{x}_p$ 만큼 평행이동되어 원점을 안 지나므로 **아핀 부분공간($\mathbf{x}_p + \text{ker}(A)$)**이 됩니다!

### 📌 3. [상황별 Trade-off & 맹점]
- **원점이 없는 공간의 파탄**: 아핀 공간은 원점을 지나지 않는 한 원소끼리 더하거나 스칼라를 곱했을 때 원점을 벗어나므로 **단독으로는 벡터 공간 공리가 파탄** 납니다. 반드시 "방향 부분공간 $U$"와 "평행이동 벡터 $\mathbf{x}_0$" 의 결합으로 표현해야 합니다.

### 📌 4. [실전 AI 연결고리]
- **인공신경망 Linear Layer ($Y = W X + b$) & SVM Hyperplane**:
  - 퍼셉트론과 딥러닝 레이어의 입력 $X$ 에 가중치 $W$ 를 곱하고 편향(Bias) $b$ 를 더하는 행위는 수학적으로 완벽한 **아핀 사상(Affine Mapping)**입니다.
  - Support Vector Machine(SVM)의 클래스 분류 경계면 또한 지지점 $\mathbf{x}_0$ 과 차원 $(n-1)$ 의 **아핀 초평면(Affine Hyperplane)**을 찾는 최적화 알고리즘입니다.
