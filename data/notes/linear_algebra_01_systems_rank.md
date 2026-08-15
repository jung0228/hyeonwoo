# 📐 01. 선형방정식계, 가우스 소거법, 그리고 Rank (Linear Systems & Rank)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 2.1 ~ 2.3 공식 원문 완전 해부**
> 
> 본 노트는 MML 교재 2.1절부터 2.3절까지의 **원문 핵심 예시, 기하학적 2가지 시각(Row vs Column Picture), 기본 행 연산(ERO), 피벗(Pivot)과 Rank의 본질적 의미, 라우셰-카펠리 정리, 그리고 실전 AI 연결고리**를 전수 정리한 마스터집입니다.

---

## 💡 1. MML 원문의 2대 핵심 직관 (Core Insights)

### 1️⃣ $Ax = b$를 바라보는 2가지 기하학적 시각 (Row vs Column Picture)
- **Row Picture (행 시각)**:
  - 각 행(Row)을 $n$차원 공간상의 **초평면(Hyperplane)** 방정식으로 해석함.
  - $Ax = b$를 푼다는 것은 **"모든 초평면들이 동시에 만나는 단 하나의 교점(Intersection)을 찾는 일"**.
- **Column Picture (열 시각 - AI에서 훨씬 중요!)**:
  - 행렬 $A = \begin{bmatrix} a_1 & a_2 & \dots & a_n \end{bmatrix}$ 의 각 열벡터 $a_j$들에 가중치 $x_j$를 곱해 더하는 **선형 결합(Linear Combination)**으로 해석함.
  - $$a_1 x_1 + a_2 x_2 + \dots + a_n x_n = b$$
  - **★ 핵심 인사이트**: $Ax = b$ 에 해가 존재한다는 것은 **"결과 벡터 $b$가 행렬 $A$의 열벡터들이 생성하는 열공간에 포함된다($b \in \text{Col}(A)$)"**는 수학적 사실과 동치임!

---

### 2️⃣ 피벗(Pivot)과 Rank의 본질 = "정보의 비중복성 (Non-redundancy)"
- **피벗 (Pivot)**: 행렬을 가우스 소거했을 때 각 행에서 0이 아닌 숫자로 처음 나타나는 대장 성분.
- **Rank (계수)**: 행렬 내에서 **"서로 겹치지 않고 진짜 새로운 정보를 제공하는 독립 축의 개수"**.
- **원문 인사이트**:
  - 기본 행 연산(ERO)을 수행할 때 $0 = 0$ 으로 소실되는 행은 **"이전에 주어졌던 방정식들의 단순 조합일 뿐인 중복(Redundant) 데이터"**이다.
  - 따라서 피벗의 개수 $\text{Rank}(A)$는 데이터가 가진 **진짜 유효 정보 차원(Effective Dimensionality)**을 뜻한다.

---

## ⚔️ 2. 4단계 표준 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **선형방정식계 (Systems of Linear Equations)**: $m$개의 방정식과 $n$개의 미지수 $x_1, \dots, x_n$의 연립 형태 $Ax = b$.
- **증대행렬 (Augmented Matrix)**: 계수 행렬 $A$와 결과 $b$를 합친 $[A \mid b] \in \mathbb{R}^{m \times (n+1)}$.
- **REF (Row Echelon Form)**: 피벗이 계단형으로 내려가고 0행은 최하단에 배치된 형태.
- **RREF (Reduced Row Echelon Form)**: 모든 피벗이 $1$이고, 피벗이 속한 열의 다른 성분이 모두 $0$인 형태.

---

### 2️⃣ [2단계 왜 쓰는가?] (라우셰-카펠리 정리 & 해의 3가지 운명)
- **라우셰-카펠리 정리 (Rouché–Capelli Theorem)**:
  1. $\text{Rank}(A) = \text{Rank}([A \mid b]) = n \implies$ **유일해 (Unique Solution)**: 열공간 내에 $b$가 유일하게 생성됨.
  2. $\text{Rank}(A) = \text{Rank}([A \mid b]) < n \implies$ **무수히 많은 해 (Infinite Solutions)**: 자유 변수(Free Variables)가 $n - \text{Rank}(A)$개 존재하여 해가 직선/평면 아핀 공간을 형성함.
  3. $\text{Rank}(A) < \text{Rank}([A \mid b]) \implies$ **해 없음 (Inconsistent System)**: $b$가 열공간 밖으로 튕겨 나감 ($b \notin \text{Col}(A)$).

---

### 3️⃣ [3단계 상황별 직관 & 수치적 맹점]
- **3가지 기본 행 연산 (Elementary Row Operations, ERO)**:
  1. 두 행 교환 ($R_i \leftrightarrow R_j$)
  2. 한 행에 스칼라 곱 ($R_i \leftarrow c R_i, c \neq 0$)
  3. 한 행에 다른 행의 배수 더하기 ($R_i \leftarrow R_i + c R_j$)
  - **직관**: ERO는 방정식을 변형해도 해 공간(Nullspace)을 보존함.
- **수치적 오차 폭발 (Floating-point Instability)**:
  - 컴퓨터 연산 시 피벗이 $0$ 근처($10^{-16}$)이면 나누기 과정에서 소수점 오차가 폭발 ➡️ **부분 피벗팅(Partial Pivoting)** 필수.

---

### 4️⃣ [4단계 실전 AI 연결고리]
- **선형 회귀 (Linear Regression)**: $y = Xw$에서 노이즈로 해가 없을 때($y \notin \text{Col}(X)$), 정사영을 내려 최적 웨이트 **$w = (X^T X)^{-1} X^T y$** 추정.
- **다중공선성 (Multicollinearity)**: $\text{Rank}(X) < n$ 이면 특징 컬럼 간 정보 중복이 심해 가역성을 잃음 ➡️ L2 Regularization(Ridge)으로 해결.

---

## 📝 3. MML 교재 원문 예시 및 전수 연습문제 풀이

### 📌 [Problem 1] MML Ex 2.1 - 3차원 선형계 가우스 소거 및 유일해 완전 유도

#### 1. 문제 정의
$$\begin{aligned} 
x_1 + 2x_2 + x_3 &= 1 \\\\ 
2x_1 + 3x_2 + 4x_3 &= 3 \\\\ 
x_1 + 4x_2 - 2x_3 &= -1 
\end{aligned}$$

#### 2. 상세 기본 행 연산 (Step-by-Step Row Operations)
- **초기 증대행렬**:
  $$[A \mid b] = \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 2 & 3 & 4 & \mid & 3 \\\\ 1 & 4 & -2 & \mid & -1 \end{bmatrix}$$

- **Step 1**: 1열 피벗($1$) 아래 소거 ($R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - R_1$)
  $$\begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 2 & -3 & \mid & -2 \end{bmatrix}$$

- **Step 2**: 2열 피벗($-1$) 아래 소거 ($R_3 \leftarrow R_3 + 2R_2$)
  $$\begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix} \quad \implies \text{REF 완성!}$$

- **Step 3**: RREF 변환 ($R_2 \leftarrow -R_2$, $R_1 \leftarrow R_1 + 2R_2$, 후방 소거)
  $$\begin{bmatrix} 1 & 0 & 0 & \mid & 3 \\\\ 0 & 1 & 0 & \mid & -1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix} \quad \implies \text{RREF 완료!}$$

- **결론**: $\text{Rank}(A) = \text{Rank}([A \mid b]) = 3 = n$. 유일해 **$\mathbf{x = \begin{bmatrix} 3 \\ -1 \\ 0 \end{bmatrix}}$**.

---

### 📌 [Problem 2] MML Ex 2.2 - Inconsistent System (해 없는 계) 3대 맹점 분석

#### 1. 문제 수식 정의
$$\begin{aligned}
x_1 + x_2 &= 2 \\\\
2x_1 + 2x_2 &= 5
\end{aligned}$$

#### 2. 상세 기본 행 연산
$$[A \mid b] = \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 2 & 2 & \mid & 5 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 0 & 0 & \mid & 1 \end{bmatrix}$$

#### 3. 3단계 비판적 모순 증명
1. **대수적 모순**: $0 \cdot x_1 + 0 \cdot x_2 = 1 \implies \mathbf{0 = 1}$ (해 불가능).
2. **랭크 불일치**: $\text{Rank}(A) = 1 < \text{Rank}([A \mid b]) = 2 \implies \mathbf{b \notin \text{Col}(A)}$.
3. **기하학적 모순**: 평면상에서 $y = -x + 2$ 와 $y = -x + \frac{5}{2}$ 의 교점이 없는 **두 평행선**.

---

### 📌 [Problem 3] MML 원문 예시 - 1차원 직선 해 공간 (자유 변수 $n - \text{Rank}(A) = 1$)

#### 1. 문제 정의
$$\begin{aligned}
x_1 + 2x_2 - x_3 &= 3 \\\\
2x_1 + 4x_2 - 2x_3 &= 6
\end{aligned}$$

#### 2. RREF 변환
$$[A \mid b] = \begin{bmatrix} 1 & 2 & -1 & \mid & 3 \\\\ 2 & 4 & -2 & \mid & 6 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 2 & -1 & \mid & 3 \\\\ 0 & 0 & 0 & \mid & 0 \end{bmatrix}$$

#### 3. 원문 핵심 인사이트: 특수해 + 동차해 벡터 표현
- $\text{Rank}(A) = 1$, 미지수 $n = 3 \implies$ 자유 변수(Free Variables) 개수 = $3 - 1 = 2$개 ($x_2 = s, x_3 = t$).
- $x_1 = 3 - 2s + t$.
- **일반해 벡터 표현 (Particular Solution + Nullspace Span)**:
  $$\mathbf{x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \underbrace{\begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}}_{\text{특수해 } x_p} + s \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}} \quad (s, t \in \mathbb{R})$$
- **기하학적 본질**: 원점을 지나는 동차해 공간(Nullspace)이 특수해 점 $x_p = [3, 0, 0]^T$ 만큼 평행 이동된 **2차원 아핀 평면(Affine Plane)**이 곧 해 공간임.
