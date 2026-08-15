# 📐 01. 선형방정식계, 가우스 소거법, 그리고 Rank (Linear Systems & Rank)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 2.1 ~ 2.3 공식 목차 100% 전수 해부**
> 
> 본 노트는 MML 교재 **Section 2.1 (Systems of Linear Equations), 2.2 (Matrices), 2.3 (Solving Systems of Linear Equations - 2.3.1 Particular and General Solution, 2.3.2 Elementary Transformations 포함)**의 모든 세부 소단원 개념, 수식, ERO 법칙, 기하학적 인사이트, 그리고 4단계 AI 매핑을 단 하나의 누락 없이 100% 매핑한 전수 마스터집입니다.

---

## 📌 MML Ch 2.1 ~ 2.3 세부 목차 (Section Hierarchy)

- **2.1 Systems of Linear Equations (선형방정식계)**
  - Row Picture (Hyperplane 교점) vs Column Picture (열벡터 선형 결합)
- **2.2 Matrices (행렬의 정의와 기본 연산)**
  - 행렬 덧셈/스칼라곱/곱셈, 역행렬 및 전치행렬의 대수적 성질
- **2.3 Solving Systems of Linear Equations (선형방정식계의 풀이)**
  - **2.3.1 Particular and General Solution (특수해와 일반해)**
    - 특수해 $x_p$와 동차해(Nullspace) $x_h$의 기하학적 구조 ($x = x_p + x_h$)
  - **2.3.2 Elementary Transformations (기본 행 연산 & 가우스 소거법)**
    - 3가지 ERO (Exchange, Scaling, Addition)
    - Row Echelon Form (REF) 및 Reduced Row Echelon Form (RREF)
  - **The Minus-1 Trick (자유 변수 및 Nullspace 계산용 MML 특수 기법)**
  - **Rank & Rouché–Capelli Theorem (계수 및 해의 3가지 운명)**

---

## 💡 1. Section 2.1: Systems of Linear Equations (선형방정식계)

### 1️⃣ $Ax = b$를 바라보는 2가지 기하학적 시각 (Row vs Column Picture)
- **Row Picture (행 시각)**:
  - 각 행(Row)을 $n$차원 공간상의 **초평면(Hyperplane)** 방정식으로 해석함.
  - $Ax = b$를 푼다는 것은 **"모든 초평면들이 동시에 만나는 단 하나의 교점(Intersection)을 찾는 일"**.
- **Column Picture (열 시각 - AI/Machine Learning 핵심!)**:
  - 행렬 $A = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix}$ 의 각 열벡터 $a_j$들에 가중치 $x_j$를 곱해 더하는 **선형 결합(Linear Combination)**으로 해석함.
  - $$a_1 x_1 + a_2 x_2 + \dots + a_n x_n = b$$
  - **★ 핵심 인사이트**: $Ax = b$ 에 해가 존재한다는 것은 **"결과 벡터 $b$가 행렬 $A$의 열벡터들이 생성하는 열공간에 포함된다($b \in \text{Col}(A)$)"**는 수학적 사실과 동치임!

---

## 💡 2. Section 2.2: Matrices (행렬과 대수적 성질)

- **행렬의 정의**: $m \times n$ 실수 행렬 $A \in \mathbb{R}^{m \times n}$.
- **행렬 곱셈의 본질 (Linear Mapping Combination)**:
  - $C = AB \implies c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$
  - **기하학적 결합**: 변환 $B$를 거친 후 변환 $A$를 연쇄적으로 적용하는 선형 사상의 합성(Composition).
- **역행렬 (Inverse Matrix $A^{-1}$)**: $A A^{-1} = A^{-1} A = I_n$ (정방행렬 및 가역 행렬에서만 정의).

---

## 💡 3. Section 2.3: Solving Systems of Linear Equations

### 3.1 [Section 2.3.1] Particular and General Solution (특수해와 일반해)

- **선형계 해의 2가지 구성요소**:
  $$\mathbf{x = x_p + x_h}$$
  - **특수해 (Particular Solution $x_p$)**: $A x_p = b$를 만족하는 단 하나의 구체적인 벡터.
  - **동차해 (General Homogeneous Solution $x_h$)**: $A x_h = 0$을 만족하는 동차계 해 공간 (행렬 $A$의 **Kernel/Nullspace**).
- **기하학적 본질 (Affine Subspace Structure)**:
  - 동차해 공간 $x_h$는 원점을 지나는 선형 부분공간(Subspace)임.
  - 여기에 특수해 $x_p$가 더해지면, 원점을 지나지 않고 $x_p$ 점만큼 평행 이동된 **아핀 공간(Affine Subspace)**이 형성됨.

---

### 3.2 [Section 2.3.2] Elementary Transformations (기본 변환 & ERO)

- **3가지 기본 행 연산 (Elementary Row Operations, ERO)**:
  1. **Exchange (행 교환)**: $R_i \leftrightarrow R_j$ (두 방정식의 위치를 바꿈).
  2. **Scaling (스칼라 배)**: $R_i \leftarrow \lambda R_i \ (\lambda \in \mathbb{R} \setminus \{0\})$.
  3. **Addition (행 더하기)**: $R_i \leftarrow R_i + \lambda R_j$ (한 행에 다른 행의 배수를 더함).
- **ERO의 대수적 불변성 (Equivalence)**:
  - ERO를 아무리 적용해도 선형계의 **해 집합(Solution Set)과 Nullspace는 절대 변형되지 않고 100% 동일하게 보존됨 (Row Equivalent)**.

---

### 3.3 REF & RREF (Row Echelon Form & Reduced Row Echelon Form)

- **Row Echelon Form (REF)**:
  1. $0$으로만 구성된 행은 최하단에 배치.
  2. 아래 행의 피벗(Leading Entry)은 위 행의 피벗보다 무조건 오른쪽에 위치.
- **Reduced Row Echelon Form (RREF)**:
  1. REF의 조건을 만족함.
  2. 모든 피벗의 값은 정확히 $1$.
  3. 피벗이 위치한 열의 다른 모든 성분은 정확히 $0$.

---

### 3.4 MML 특수 기법: The Minus-1 Trick (Nullspace 직관적 추출법)

- **MML 교재 2.3절 특수 팁**:
  - RREF 상태의 행렬에서 피벗이 없는 열(자유 변수 열)에 대응하는 대각 성분에 **$-1$**을 채워 넣음으로써, 가우스-조던 소거 결과로부터 Nullspace 기저 벡터들을 연산 없이 **눈으로 즉시 읽어내는 교재 독점 스킬**.

---

### 3.5 Rank & Rouché–Capelli Theorem (계수 및 해의 3가지 운명)

- **Rank (계수)**: RREF 변환 후 **피벗의 총 개수**이자, 데이터의 **진짜 유효 정보 차원(Effective Dimensionality)**.
- **라우셰-카펠리 정리 (Rouché–Capelli Theorem)**:
  1. $\text{Rank}(A) = \text{Rank}([A \mid b]) = n \implies$ **유일해 (Unique Solution)**
  2. $\text{Rank}(A) = \text{Rank}([A \mid b]) < n \implies$ **무수히 많은 해 (Infinite Solutions)** ($n - \text{Rank}(A)$개의 자유 변수)
  3. $\text{Rank}(A) < \text{Rank}([A \mid b]) \implies$ **해 없음 (Inconsistent System)** ($b \notin \text{Col}(A)$)

---

## ⚔️ 4. 4단계 실전 AI 매핑 (AI Connection)

- **선형 회귀 (Linear Regression)**: $y = Xw$에서 노이즈로 해가 없을 때($y \notin \text{Col}(X)$), 정사영을 내려 최적 웨이트 **$w = (X^T X)^{-1} X^T y$** 추정.
- **다중공선성 (Multicollinearity)**: $\text{Rank}(X) < n$ 이면 특징 컬럼 간 정보 중복이 심해 가역성을 잃음 ➡️ L2 Regularization(Ridge)으로 해결.

---

## 📝 5. MML 교재 전수 연습문제 풀이 (Step-by-Step)

### 📌 [Problem 1] MML Ex 2.1 - 3차원 선형계 가우스 소거 (유일해)
- **증대행렬**: $[A \mid b] = \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 2 & 3 & 4 & \mid & 3 \\\\ 1 & 4 & -2 & \mid & -1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & 0 & \mid & 3 \\\\ 0 & 1 & 0 & \mid & -1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix}$
- **결론**: $\text{Rank}(A) = 3 = n \implies \mathbf{x = \begin{bmatrix} 3 \\ -1 \\ 0 \end{bmatrix}}$.

---

### 📌 [Problem 2] MML Ex 2.2 - Inconsistent System (해 없음)
- **증대행렬**: $[A \mid b] = \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 2 & 2 & \mid & 5 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 0 & 0 & \mid & 1 \end{bmatrix}$
- **결론**: $0 = 1$ 대수적 모순, $\text{Rank}(A)=1 < \text{Rank}([A \mid b])=2 \implies \mathbf{b \notin \text{Col}(A)}$.

---

### 📌 [Problem 3] MML Section 2.3.1 Ex - Particular + General Solution ($x = x_p + x_h$)
- **방정식**: $x_1 + 2x_2 - x_3 = 3$
- **RREF**: $[A \mid b] = \begin{bmatrix} 1 & 2 & -1 & \mid & 3 \end{bmatrix}$
- **해 공간**: $\mathbf{x = \underbrace{\begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}}_{x_p (\text{특수해})} + s \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}} \quad (s, t \in \mathbb{R})$
