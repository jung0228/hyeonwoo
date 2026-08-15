# 📐 01. 선형방정식계, 기본 행 연산, 그리고 계수 (Linear Systems & Rank)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 2.1 ~ 2.3 전수 완전 정리**
> 
> 본 노트는 MML 교재 2.1절부터 2.3절까지의 **모든 이론, 수학적 정의, 기본 행 연산(Elementary Row Operations) 법칙, 피벗(Pivot), 사다리꼴(REF/RREF), 라우셰-카펠리 정리(Rouché-Capelli Theorem), 그리고 4단계 실전 AI 매핑**을 빈틈없이 전수 정리한 수료집입니다.

---

## 1. ⚔️ 4단계 표준 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **선형방정식계 (Systems of Linear Equations)**: $m$개의 선형방정식과 $n$개의 미지수 $x_1, x_2, \dots, x_n$으로 구성된 연립방정식.
  $$\sum_{j=1}^n a_{ij} x_j = b_i \quad (i = 1, 2, \dots, m) \iff A x = b$$
- **증대행렬 (Augmented Matrix)**: 계수 행렬 $A \in \mathbb{R}^{m \times n}$과 결과 벡터 $b \in \mathbb{R}^m$을 하나로 합친 형태 $[A \mid b] \in \mathbb{R}^{m \times (n+1)}$.
- **피벗 (Pivot / Leading Entry)**: 행렬의 각 행에서 $0$이 아닌 숫자로 처음 등장하는 제일 왼쪽 요소.
- **행 사다리꼴 (Row Echelon Form, REF)**:
  1. 모든 성분이 $0$인 행은 최하단에 위치한다.
  2. 아래 행의 피벗은 위 행 피벗보다 무조건 오른쪽에 위치한다.
- **기하된 행 사다리꼴 (Reduced Row Echelon Form, RREF)**:
  1. REF의 조건을 100% 만족한다.
  2. 모든 피벗의 값은 정확히 $1$이다.
  3. 피벗이 속한 열(Column)의 나머지 모든 성분은 정확히 $0$이다.
- **계수 (Rank)**: 행렬 $A$를 REF/RREF로 변환했을 때 생성되는 **피벗의 총 개수**이며, 행렬이 만드는 기하학적 차원 $\dim(\text{Col}(A))$을 뜻함.

---

### 2️⃣ [2단계 왜 쓰는가?] (존재 이유 & 기하학적 직관)
- **공간의 가역성 및 차원 보존 판별**: 선형방정식 $Ax = b$를 풀 때 $A$가 공간을 찌그러뜨려 정보를 소실시키는지, 아니면 유일하게 복원 가능한지 판별하기 위함.
- **라우셰-카펠리 정리 (Rouché–Capelli Theorem)**:
  - $\text{Rank}(A) = \text{Rank}([A \mid b]) = n \implies$ **유일해 (Unique Solution)**: 공간이 축소되지 않고 정확히 교점 1개가 존재함.
  - $\text{Rank}(A) = \text{Rank}([A \mid b]) < n \implies$ **무수히 많은 해 (Infinite Solutions)**: 자유 변수(Free Variables)가 $n - \text{Rank}(A)$개 존재하여 해 공간이 직선/평면을 이룸.
  - $\text{Rank}(A) < \text{Rank}([A \mid b]) \implies$ **해 없음 (Inconsistent System)**: 결과 벡터 $b$가 열공간 $\text{Col}(A)$ 밖으로 튕겨 나감 ($b \notin \text{Col}(A)$).

---

### 3️⃣ [3단계 상황별 직관 & 수치적 맹점 (Trade-off)]
- **3가지 기본 행 연산 (Elementary Row Operations, ERO)**:
  1. $R_i \leftrightarrow R_j$: 두 행의 위치를 바꾼다.
  2. $R_i \leftarrow c R_i \ (c \neq 0)$: 한 행에 $0$이 아닌 스칼라를 곱한다.
  3. $R_i \leftarrow R_i + c R_j$: 한 행에 다른 행의 스칼라배를 더한다.
  - **성질**: ERO를 수행해도 행렬의 해 공간(Null Space)과 랭크(Rank)는 **절대 변하지 않는다 (Row Equivalent)**.
- **수치적 불안정성 (Numerical Instability & Floating-point Error)**:
  - 컴퓨터 부동소수점 연산 시 피벗이 $0$에 매우 가까우면($10^{-16}$) 나누기 과정에서 오차가 기하급수적으로 폭발함.
  - **해결책**: 행 연산 시 절댓값이 가장 큰 요소를 피벗으로 선택하는 **부분 피벗팅(Partial Pivoting)** 공정이 필수적임.

---

### 4️⃣ [4단계 실전 AI 매핑 (AI Connection)]
- **선형 회귀 및 신경망 기저**: 신경망의 기본 레이어 $y = Wx + b$ 및 선형 회귀 최적화는 본질적으로 $Ax=b$ 시스템을 푸는 과정임.
- **특징 선형 독립성 (Feature Independence)**: 입력 특징 행렬 $X$의 $\text{Rank}(X)$가 피벗으로 꽉 차야(Full Rank) 컬럼 간 다중공선성(Multicollinearity) 문제가 없이 모델이 안정적으로 학습됨.
- **최소제곱법 (Least Squares)**: 센서 데이터 노이즈로 해가 없을 때($b \notin \text{Col}(A)$), 정사영을 내려 최적 근사 웨이트 $w = (X^T X)^{-1} X^T y$ 를 추정.

---

## 📝 2. MML Ch 2.1 교재 전수 연습문제 풀이

### 📌 [Problem 1] MML Ex 2.1 - 3차원 선형계 가우스 소거 및 유일해 완전 유도

#### 1. 문제 수식 정의
$$\begin{aligned} 
x_1 + 2x_2 + x_3 &= 1 \\\\ 
2x_1 + 3x_2 + 4x_3 &= 3 \\\\ 
x_1 + 4x_2 - 2x_3 &= -1 
\end{aligned}$$

#### 2. 상세 기본 행 연산 (Step-by-Step Row Operations)
- **초기 증대행렬**:
  $$[A \mid b] = \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 2 & 3 & 4 & \mid & 3 \\\\ 1 & 4 & -2 & \mid & -1 \end{bmatrix}$$

- **Step 1**: 1열 피벗($1$) 아래 요소 소거 ($R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - R_1$)
  $$\begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 2 & -3 & \mid & -2 \end{bmatrix}$$

- **Step 2**: 2열 피벗($-1$) 아래 요소 소거 ($R_3 \leftarrow R_3 + 2R_2$)
  $$\begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix} \quad \implies \text{행 사다리꼴 (REF) 완성!}$$

- **Step 3**: RREF 변환 ($R_2 \leftarrow -R_2$, $R_1 \leftarrow R_1 + 2R_2$, 후방 소거)
  $$\begin{bmatrix} 1 & 0 & 0 & \mid & 3 \\\\ 0 & 1 & 0 & \mid & -1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix} \quad \implies \text{RREF 완료!}$$

- **결론**: $\text{Rank}(A) = \text{Rank}([A \mid b]) = 3 = n$. 유일해 존재하며 **최종 해벡터 $\mathbf{x = \begin{bmatrix} 3 \\ -1 \\ 0 \end{bmatrix}}$**.

---

### 📌 [Problem 2] MML Ex 2.2 - Inconsistent System (해 없는 계) 3대 맹점 분석

#### 1. 문제 수식 정의
$$\begin{aligned}
x_1 + x_2 &= 2 \\\\
2x_1 + 2x_2 &= 5
\end{aligned}$$

#### 2. 상세 기본 행 연산 (Row Operations)
$$[A \mid b] = \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 2 & 2 & \mid & 5 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 0 & 0 & \mid & 1 \end{bmatrix}$$

#### 3. 3단계 비판적 모순 증명
1. **대수적 모순**: $0 \cdot x_1 + 0 \cdot x_2 = 1 \implies \mathbf{0 = 1}$ (해 불가능).
2. **랭크 불일치**: $\text{Rank}(A) = 1 < \text{Rank}([A \mid b]) = 2 \implies \mathbf{b \notin \text{Col}(A)}$.
3. **기하학적 모순**: 평면상에서 $y = -x + 2$ 와 $y = -x + \frac{5}{2}$ 의 교점이 없는 **두 평행선**.

---

### 📌 [Problem 3] MML Ex 2.3 - 무수히 많은 해 (Infinite Solutions & 자유 변수) 유도

#### 1. 문제 수식 정의
$$\begin{aligned}
x_1 + 2x_2 - x_3 &= 3 \\\\
2x_1 + 4x_2 - 2x_3 &= 6
\end{aligned}$$

#### 2. 상세 기본 행 연산 및 RREF
$$[A \mid b] = \begin{bmatrix} 1 & 2 & -1 & \mid & 3 \\\\ 2 & 4 & -2 & \mid & 6 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 2 & -1 & \mid & 3 \\\\ 0 & 0 & 0 & \mid & 0 \end{bmatrix}$$

#### 3. 해 공간 일반식 유도 (Parametric Form)
- 피벗은 1열($x_1$)에만 존재하므로 $\text{Rank}(A) = 1$.
- 자유 변수(Free Variables): $x_2 = s, \ x_3 = t \ (s, t \in \mathbb{R})$.
- $x_1 = 3 - 2s + t$.
- **일반해 (General Solution Vector)**:
  $$\mathbf{x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix} + s \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + t \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}} \quad (s, t \in \mathbb{R})$$
- **기하학적 해석**: 3차원 공간에서 특수해 $[3, 0, 0]^T$를 지나고 두 방향벡터가 만드는 **2차원 아핀 평면 전체가 해 공간**이 됨.
