# 📐 선형대수학 MML 전수 연습문제 풀이집 (Linear Algebra Problem Set)

> POSTECH 대학원 준비 4단계 표준 체계 100% 준수
> 
> 본 문서에는 MML(Mathematics for Machine Learning) 교재 Part I (Chapter 2~4)에 수록된 핵심 연습문제들의 유도 과정, 대수적/기하학적 증명, 수치적 맹점 및 실전 AI 연결고리가 수록되어 있습니다.

---

---

## 📝 1. 선형방정식계 & Rank (Ch 2.1 ~ 2.2)

### [Problem 1] Ex 2.1 - 선형계 가우스 소거법 단계별 전개 및 유일해 완전 유도

#### 1. 문제 정의 (Problem Statement)
다음 3차원 선형방정식계 $Ax = b$ 의 해 공간 구조를 구하고 유일해 존재 여부를 판별하라.
$$\begin{aligned} 
x_1 + 2x_2 + x_3 &= 1 \\\\ 
2x_1 + 3x_2 + 4x_3 &= 3 \\\\ 
x_1 + 4x_2 - 2x_3 &= -1 
\end{aligned}$$

#### 2. 상세 기본 행 연산 단계 (Step-by-Step Row Operations)
- 초기 증대행렬 (Initial Augmented Matrix):
  $$[A \mid b] = \begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 2 & 3 & 4 & \mid & 3 \\\\ 1 & 4 & -2 & \mid & -1 \end{bmatrix}$$

- Step 1: 1열 피벗 아래 요소 소거 ($R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - R_1$)
  $$\begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 2 & -3 & \mid & -2 \end{bmatrix}$$

- Step 2: 2열 피벗 아래 요소 소거 ($R_3 \leftarrow R_3 + 2R_2$)
  $$\begin{bmatrix} 1 & 2 & 1 & \mid & 1 \\\\ 0 & -1 & 2 & \mid & 1 \\\\ 0 & 0 & 1 & \mid & 0 \end{bmatrix} \quad \implies \text{상삼각 행렬 (REF) 완성!}$$

#### 3. 4단계 구조적 분석 및 해 판별 (Rouché–Capelli Analysis)
1. [1단계 명확한 개념 정의]: 피벗(Pivot)은 각 행에서 0이 아닌 첫 번째 요소이며, 계수 $\text{Rank}(A)$는 피벗의 개수와 같다.
2. [2단계 존재 이유]: 라우셰-카펠리 정리에 의해 계수 행렬의 랭크 $\text{Rank}(A) = 3$ 이고 증대 행렬의 랭크 $\text{Rank}([A \mid b]) = 3$ 이며, 미지수의 개수 $n = 3$ 과 동일하므로 공간의 찌그러짐 없이 유일해(Unique Solution)가 존재합니다.
3. [3단계 후방 대입법 (Back-Substitution)]:
   - $3\text{행}: x_3 = 0$
   - $2\text{행}: -x_2 + 2(0) = 1 \implies x_2 = -1$
   - $1\text{행}: x_1 + 2(-1) + (0) = 1 \implies x_1 = 3$
   - 최종 해벡터: $\mathbf{x = \begin{bmatrix} 3 \\\\ -1 \\\\ 0 \end{bmatrix}}$
4. [4단계 실전 AI 연결]: AI 학습 시 $Ax=b$ 형태의 선형계는 최적화 가중치 구하기의 근본이며, 피벗이 3개로 꽉 찬 것은 모델의 특징(Feature) 컬럼 간 선형 독립성이 완벽함을 뜻한다.

---

### [Problem 2] Ex 2.2 - Inconsistent System (해 없는 선형계) 대수적/기하학적 맹점 심층 분석

#### 1. 문제 정의 (Problem Statement)
다음 2차원 연립방정식계의 해 공간을 분석하고, 해가 존재하지 않는 원인을 3단계 맹점으로 증명하라.
$$\begin{aligned}
x_1 + x_2 &= 2 \\\\
2x_1 + 2x_2 &= 5
\end{aligned}$$

#### 2. 상세 기본 행 연산 단계 (Step-by-Step Row Operations)
- 증대행렬 소거 ($R_2 \leftarrow R_2 - 2R_1$):
  $$[A \mid b] = \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 2 & 2 & \mid & 5 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 1 & \mid & 2 \\\\ 0 & 0 & \mid & 1 \end{bmatrix}$$

#### 3. 3단계 비판적 모순 증명 (Critical Insight & Contradiction)
1. 대수적 모순 (Algebraic Contradiction): 
   - 2행을 수식으로 복원하면 $0 \cdot x_1 + 0 \cdot x_2 = 1 \implies \mathbf{0 = 1}$ 이라는 절대 불가능한 모순이 발생합니다.
2. 열공간 랭크 불일치 (Rank Discrepancy & Column Space): 
   - $\text{Rank}(A) = 1$ 이지만 $\text{Rank}([A \mid b]) = 2$ 이다.
   - 이는 결과 벡터 $b = \begin{bmatrix} 2 \\\\ 5 \end{bmatrix}$ 가 행렬 $A$의 열공간 $\text{Col}(A) = \text{span}\left(\begin{bmatrix} 1 \\\\ 2 \end{bmatrix}\right)$ 선상에 존재하지 않고 공간 밖으로 튕겨 나갔음($b \notin \text{Col}(A)$)을 수학적으로 증명한다.
3. 기하학적 모순 (Geometric Interpretation): 
   - 2차원 평면상에서 $x_2 = -x_1 + 2$ 와 $x_2 = -x_1 + \frac{5}{2}$ 로 기울기가 동일하지만 $y$절편이 서로 다른 두 평행 직선이므로 교점이 존재할 수 없다.

#### 4. 실전 AI 연결고리 (Least Squares Approximation)
- 데이터 분류/회귀 문제에서 센서 노이즈나 타겟 데이터 $y$가 독립 특징 공간 $\text{Col}(X)$ 밖으로 나가면 완벽한 해 $Xw = y$ 는 존재하지 않습니다.
- 이때 AI는 정사영(Projection)을 통해 $\text{Col}(X)$ 공간 위로 가장 가까운 점 $\hat{y}$을 내려 최적 근사해 최소제곱법 정규방정식 $w = (X^T X)^{-1} X^T y$ 를 추정한다.

---

## 📝 2. 행렬식 & 역행렬 (Ch 2.3 ~ 2.4)

### [Problem 3] Ex 2.3 - 행렬식 det(A) 부피 팽창율 및 3x3 Sarrus 증명
- 개념: $\det(A)$는 $n$차원 단위 초입방체가 선형변환 $A$를 거쳐 변형된 공간의 부피 팽창 비율.
- 3x3 행렬 Sarrus 공식을 라플라스 전개(Laplace Expansion)로 증명:
  $$A = \begin{bmatrix} a & b & c \\\\ d & e & f \\\\ g & h & i \end{bmatrix}$$
  $$\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg) = aei + bfg + cdh - ceg - bdi - afh$$
- AI 연결고리: 생성형 모델인 Normalizing Flow에서 확률밀도 변환 시 부피 변화율인 자코비안 행렬식 $\det(J_f)$을 보정값으로 곱해준다.

---

### [Problem 4] Ex 2.4 - Inverse & Ill-conditioned 행렬 수치 맹점
- 조건수 (Condition Number): $\kappa(A) = \|A\| \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$
- 수치적 맹점:
  - $\det(A) \approx 0$ 이거나 $\kappa(A) \gg 10^3$ 이면 행렬이 Ill-conditioned(악조건) 상태에 빠짐.
  - 이 상태에서 역행렬 $A^{-1}$을 직접 구하면 부동소수점 오차가 기하급수적으로 폭발하여 계산이 무너진다.
- 해결책: 역행렬을 직접 계산하지 않고 QR 분해나 SVD 핀로즈 유사역행렬(Pseudoinverse $A^+$)을 사용한다.

---

## 📝 3. 부분공간, Kernel/Image & Rank-Nullity (Ch 3.1)

### [Problem 5] Ex 3.1 - Rank-Nullity 정리 및 Kernel / Image 공간 분해 증명
- Rank-Nullity Theorem: 선형변환 $T: V \to W$에서 $\dim(V) = \text{Rank}(T) + \text{Nullity}(T)$
- 증명 요약:
  1. $\ker(T)$의 기저를 $\{v_1, \dots, v_k\}$ 라 두고, 이를 확장하여 $V$의 기저 $\{v_1, \dots, v_k, v_{k+1}, \dots, v_n\}$ 을 구성한다.
  2. $\{T(v_{k+1}), \dots, T(v_n)\}$ 이 $\text{Im}(T)$의 독립적인 기저임을 증명합니다.
  3. 따라서 $\dim(\text{Im}(T)) = n - k \implies n = k + (n-k)$ 성립!
- AI 연결고리: Autoencoder에서 복원 불가능한 소실 정보 차원이 $\ker(T)$, 표현 가능한 차원이 $\text{Im}(T)$에 대응됨.

---

## 📝 4. 고유값/고유벡터 & SVD (Ch 4.1 ~ 4.2)

### [Problem 6] Ex 4.1 - Spectral Theorem 대칭행렬 직교 분해 백지 증명
- 증명: $\lambda_1 x_1^T x_2 = (Ax_1)^T x_2 = x_1^T A^T x_2 = x_1^T A x_2 = \lambda_2 x_1^T x_2$
- $(\lambda_1 - \lambda_2)(x_1^T x_2) = 0 \implies \mathbf{x_1 \perp x_2}$ 증명 완료!
- 정규직교행렬 $Q^T Q = I \implies \mathbf{A = Q \Lambda Q^T}$.

---

### [Problem 7] Ex 4.2 - SVD와 A^T A Eigendecomposition 대입 증명
- 증명: $A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = \mathbf{V (\Sigma^T \Sigma) V^T}$
- $V$는 $A^T A$의 고유벡터, 특이값 $\sigma_i = \sqrt{\lambda_i(A^TA)}$.
- LoRA (Low-Rank Adaptation) AI 연결: SVD의 Truncated SVD 원리로 1750억 개 파라미터 $\Delta W$를 상위 Rank $r$개의 $B \times A$로 쪼개 파라미터 99.9% 절감.

---

## 📝 5. 기저변환, 아핀 공간 & 슈어 보강 (Ch 3.2, 2.5, 4.3)

### [Problem 8] Ex 3.2 - 기저변환 (Change of Basis) 동형사상 P^(-1)AP 유도
- 유도: 구기저 $B$에서 신기저 $B'$로의 기저변환 행렬 $P$.
- 동일한 선형변환 $T$의 표현 행렬 $A_{B'}$는 $A_{B'} = P^{-1} A_B P$.
- AI 연결: Transformer의 Self-Attention Projection ($Q = XW_Q, K = XW_K$)이 입력 공간을 Attention 헤드의 기저 공간으로 재좌표화하는 과정.

---

### [Problem 9] Ex 2.5 - 아핀 부분공간 (Affine Subspaces) & 아핀 변환 유도
- 정의: 원점을 지나지 않는 이동된 부분공간 $L = x_0 + U = \{x_0 + u \mid u \in U\}$.
- 아핀 변환: $f(x) = A x + b$ (선형 변환 + 평행 이동).
- AI 연결: 신경망 레이어의 기본 단위인 $y = \sigma(W x + b)$에서 편향(Bias) $b$가 아핀 공간 평행 이동을 담당.

---

### [Problem 10] Ex 4.3 - Schur Complement (슈어 보강) 및 블록 행렬식 증명
- 블록 행렬: $M = \begin{bmatrix} A & B \\\\ C & D \end{bmatrix}$ (단, $D$는 가역)
- Schur Complement: $S = A - B D^{-1} C$
- 블록 소거 분해:
  $$\begin{bmatrix} I & -BD^{-1} \\\\ 0 & I \end{bmatrix} \begin{bmatrix} A & B \\\\ C & D \end{bmatrix} \begin{bmatrix} I & 0 \\\\ -D^{-1}C & I \end{bmatrix} = \begin{bmatrix} A - BD^{-1}C & 0 \\\\ 0 & D \end{bmatrix}$$
- 행렬식 증명: $\det(M) = \det(D) \cdot \det(A - B D^{-1} C)$.
- AI 연결: 가우시안 프로세스(Gaussian Process) 및 칼만 필터(Kalman Filter)의 마지널/조건부 확률 분포 계산의 핵심 연산.
