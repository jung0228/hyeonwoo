# 📐 2.1 Systems of Linear Equations (선형방정식계)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.1 완전 해부**

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **선형방정식계 (Systems of Linear Equations)**: $m$개의 방정식과 $n$개의 미지수 $x_1, \dots, x_n$으로 구성된 연립 1차 방정식계.
  $$\sum_{j=1}^n a_{ij} x_j = b_i \quad (i = 1, \dots, m) \iff A x = b$$

---

### 2️⃣ [2단계 기하학적 2가지 시각 (Row vs Column Picture)]
- **Row Picture (행 시각)**:
  - 각 행(Row)을 $n$차원 공간상의 **초평면(Hyperplane)** 방정식으로 해석함.
  - $Ax = b$를 푼다는 것은 **"모든 초평면들이 동시에 만나는 단 하나의 교점(Intersection)을 찾는 일"**.
- **Column Picture (열 시각 - AI/ML 핵심!)**:
  - 행렬 $A = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix}$ 의 각 열벡터 $a_j$들에 가중치 $x_j$를 곱해 더하는 **선형 결합(Linear Combination)**.
  - $$a_1 x_1 + a_2 x_2 + \dots + a_n x_n = b$$
  - **★ 핵심 인사이트**: $Ax = b$에 해가 존재한다는 것은 **"결과 벡터 $b$가 행렬 $A$의 열벡터들이 펼치는 공간(Column Space)에 정확히 속한다($b \in \text{Col}(A)$)"**는 뜻임!

---

### 3️⃣ [3단계 상황별 직관 & 해의 3가지 가능성]
1. **유일해 (Unique Solution)**: 열공간 내에 $b$가 정확히 1개의 계수 조합으로 표현됨.
2. **무수히 많은 해 (Infinite Solutions)**: 열벡터 간 중복(Redundancy)이 존재하여 계수 자유 변수(Free Variable)가 발생함.
3. **해 없음 (Inconsistent System)**: 결과 벡터 $b$가 열공간 밖으로 튕겨 나감 ($b \notin \text{Col}(A)$).

---

### 4️⃣ [4단계 실전 AI 연결고리]
- **선형 회귀 (Linear Regression)**: $y = Xw$에서 노이즈로 해가 없을 때($y \notin \text{Col}(X)$), 정사영을 내려 최적 웨이트 **$w = (X^T X)^{-1} X^T y$** 추정.

---

## 🔍 2. ★ MML 교재 원문 예시 해부: Example 2.1 (자원 배분과 생산 계획 선형계)

MML 교재 2.1절 원문 **Example 2.1**:
> *"A company produces products $N_1, \dots, N_n$ for which resources $R_1, \dots, R_m$ are required. To produce a unit of product $N_j$, $a_{ij}$ units of resource $R_i$ are needed, where $i = 1, \dots, m$ and $j = 1, \dots, n$. The objective is to find an optimal production plan, i.e., a plan of how many units $x_j$ of product $N_j$ should be produced if a total of $b_i$ units of resource $R_i$ are available and (ideally) no resources are left over."*

### 📐 수학적 선형방정식계 $Ax = b$ 수식화
- 자원 $R_i$의 사용 총량 방정식:
  $$a_{i1} x_1 + a_{i2} x_2 + \dots + a_{in} x_n = b_i \quad (i = 1, \dots, m)$$
- **행렬 표기법 (Matrix Notation $Ax = b$)**:
  $$\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\\\ a_{21} & a_{22} & \dots & a_{2n} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\\\ b_2 \\\\ \vdots \\\\ b_m \end{bmatrix}$$
- **의미**: 각 제품 $N_j$의 생산량 $x_j$를 구하는 문제가 곧 **결과 자원 벡터 $b$를 기술 행렬 $A$의 열벡터 선형결합으로 분해하는 선형방정식계 문제**와 완벽히 동일함!

---

### 📌 Example 2.2 (MML 원문: 해의 3가지 가능성 - 해 없음, 유일해, 무수히 많은 해)

MML 교재 2.1절 원문 **Example 2.2**:

#### 1️⃣ [Case 1: No Solution (해 없음)]
$$\begin{aligned}
x_1 + x_2 + x_3 &= 3 \quad (1) \\\\
x_1 - x_2 + 2x_3 &= 2 \quad (2) \\\\
2x_1 + 3x_3 &= 1 \quad (3)
\end{aligned}$$
- **분석**: (1)식과 (2)식을 더하면 $2x_1 + 3x_3 = 5$ 가 되는데, 이는 (3)식의 $2x_1 + 3x_3 = 1$ 과 **모순($5 = 1$)**됨 ➡️ **해 없음(No Solution)**!

#### 2️⃣ [Case 2: Unique Solution (유일해)]
$$\begin{aligned}
x_1 + x_2 + x_3 &= 3 \quad (1) \\\\
x_1 - x_2 + 2x_3 &= 2 \quad (2) \\\\
x_2 + x_3 &= 2 \quad (3)
\end{aligned}$$
- **분석**: (1)식에서 (3)식을 빼면 $x_1 = 1$. (1)+(2)에서 $2x_1 + 3x_3 = 5 \implies x_3 = 1$. (3)식에서 $x_2 = 1$.
- **최종 유일해**: $\mathbf{(x_1, x_2, x_3) = (1, 1, 1)}$ 만 존재!

#### 3️⃣ [Case 3: Infinite Solutions (무수히 많은 해)]
$$\begin{aligned}
x_1 + x_2 + x_3 &= 3 \quad (1) \\\\
x_1 - x_2 + 2x_3 &= 2 \quad (2) \\\\
2x_1 + 3x_3 &= 5 \quad (3)
\end{aligned}$$
- **분석**: (1)+(2)=(3) 이므로 3번 식은 중복(Redundancy)되어 제거 가능. $x_3 = a \in \mathbb{R}$ 를 자유 변수로 두면 해집합:
  $$\mathbf{x = \begin{bmatrix} \frac{5}{2} - \frac{3}{2}a \\\\ \frac{1}{2} + \frac{1}{2}a \\\\ a \end{bmatrix} = \begin{bmatrix} \frac{5}{2} \\\\ \frac{1}{2} \\\\ 0 \end{bmatrix} + a \begin{bmatrix} -\frac{3}{2} \\\\ \frac{1}{2} \\\\ 1 \end{bmatrix}, \quad a \in \mathbb{R}}$$
