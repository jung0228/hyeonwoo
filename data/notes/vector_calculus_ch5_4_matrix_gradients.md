# 📐 5.4 & 5.5 Gradients of Matrices & Useful Identities (행렬 미분, 고차원 텐서 야코비안과 행렬 미분 10대 핵심 항등식)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 5.4, 5.5 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 행렬 미분의 세계: 왜 "행렬에 대한 미분"과 "10대 항등식"을 배우는가?

우리는 앞선 5.2~5.3절에서 벡터 입력과 벡터 출력에 대한 미분을 배웠습니다.
하지만 딥러닝과 머신러닝의 파라미터(트랜스포머의 어텐션 가중치 행렬 $W_Q, W_K, W_V$, 선형 회귀의 가중치 행렬 $W$, 가우시안 모델의 공분산 행렬 $\Sigma$)는 대부분 2차원 행렬(Matrix) 형태입니다.

- 왜 행렬 미분을 해야 하는가?: 신경망의 오차를 줄이기 위해 손실함수 $\text{Loss}$ 를 수억 개의 가중치가 모여있는 거대한 행렬 $W$ 로 직접 미분하여 업데이트 방향($\frac{\partial \text{Loss}}{\partial W}$)을 알아내야 하기 때문입니다.
- 고차원 야코비안 텐서 (Jacobian Tensor): 행렬을 또 다른 행렬로 미분하면 4차원 텐서($\mathbb{R}^{m \times n \times p \times q}$)가 튀어나옵니다.
- 평탄화(Flattening)와 벡터 공간 동형사상 ($\mathbb{R}^{m \times n} \cong \mathbb{R}^{mn}$): 행렬을 $mn$ 길이의 벡터로 펴면 복잡한 텐서 미분을 표준 2차원 행렬 곱셈으로 단순화하여 연쇄 법칙을 쉽게 적용할 수 있습니다.
- 머신러닝의 10대 행렬 미분 항등식: 행렬식($\det$), 대각합($\text{tr}$), 역행렬($X^{-1}$), 이차 형식($\mathbf{x}^\top B \mathbf{x}$)의 미분 공식을 암기 수준으로 숙지해야 가우시안 MLE, 칼만 필터, 선형 회귀 정규방정식을 종이 위에서 1초 만에 유도할 수 있습니다.


## 1. ⚔️ Section 5.4: Gradients of Matrices (행렬 미분과 평탄화 기법)


### 💡 왜 행렬 미분을 하면 차원이 폭발하고, 왜 평탄화(Flattening)를 하는가?

1. 차원의 폭발 문제:
   $m \times n$ 크기의 출력 행렬 $A$ 와 $p \times q$ 크기의 입력 행렬 $B$ 가 있을 때, $A$ 의 모든 원소 $A_{ij}$ 각각을 $B$ 의 모든 원소 $B_{kl}$ 각각으로 미분해야 합니다.
   이로 인해 총 $(m \times n) \times (p \times q)$ 개의 편미분 값들이 쏟아져 나오며 4차원 초입체 상자인 4차원 텐서(4D Tensor $J_{ijkl}$) 가 생성됩니다.
2. 평탄화(Flattening)를 통한 해결:
   4차원 텐서를 손으로 다루는 것은 매우 난해하므로, $m \times n$ 행렬 $A$ 를 1줄짜리 $mn$차원 벡터로 펴고, $p \times q$ 행렬 $B$ 를 1줄짜리 $pq$차원 벡터로 폅니다 ($\mathbb{R}^{m \times n} \cong \mathbb{R}^{mn}$).
   이렇게 평탄화하면 4차원 텐서 미분이 우리가 잘 아는 $mn \times pq$ 2차원 야코비안 행렬 곱셈으로 깔끔하게 변환되어 연쇄 법칙을 초고속으로 계산할 수 있습니다!


### 📌 1. 행렬 미분의 차원과 텐서의 정의 (Eq 5.86~5.87)

$$J_{ijkl} := \frac{\partial A_{ij}}{\partial B_{kl}} \in \mathbb{R}^{m \times n \times p \times q}$$


### 📌 2. 벡터 공간 동형사상과 행렬 평탄화 (Flattening / Vectorization)

1. 행렬 $A \in \mathbb{R}^{m \times n}$ 과 $B \in \mathbb{R}^{p \times q}$ 를 열(Column) 기준으로 길게 쌓아 벡터 $\tilde{\mathbf{a}} \in \mathbb{R}^{mn}, \; \tilde{\mathbf{b}} \in \mathbb{R}^{pq}$ 로 평탄화합니다.
2. 미분을 수행하면 표준 $mn \times pq$ 2차원 야코비안 행렬 $\frac{d\tilde{\mathbf{a}}}{d\tilde{\mathbf{b}}}$ 이 도출됩니다 (Figure 5.7).
3. 실전 이점: 고차원 텐서 수축(Tensor Contraction)을 고민할 필요 없이, 다변수 연쇄 법칙을 단순한 2차원 행렬 곱셈으로 초고속 처리할 수 있습니다!


### 💡 [Example 5.12: 벡터를 행렬로 미분 ($\mathbf{f} = A\mathbf{x}$)]
$\mathbf{f} = A\mathbf{x} \in \mathbb{R}^M$ ($A \in \mathbb{R}^{M \times N}, \; \mathbf{x} \in \mathbb{R}^N$) 에 대해 $\frac{d\mathbf{f}}{dA} \in \mathbb{R}^{M \times (M \times N)}$ 도출:
- $i$번째 출력: $f_i = \sum_{j=1}^N A_{ij} x_j$
- $A$ 의 $i$번째 행($A_{i, :}$)에 대한 편미분: $\frac{\partial f_i}{\partial A_{i, :}} = \mathbf{x}^\top \in \mathbb{R}^{1 \times 1 \times N}$
- 다른 행($A_{k \neq i, :}$)에 대한 편미분: $\mathbf{0}^\top \in \mathbb{R}^{1 \times 1 \times N}$
- $i$번째 출력의 그래디언트 (Eq 5.92):
  $$\frac{\partial f_i}{\partial A} = \begin{bmatrix} \mathbf{0}^\top \\\\ \vdots \\\\ \mathbf{x}^\top \\\\ \vdots \\\\ \mathbf{0}^\top \end{bmatrix} \in \mathbb{R}^{1 \times (M \times N)}$$


### 💡 [Example 5.13: 행렬을 행렬로 미분 (커널 행렬 $K = R^\top R$)]
$R \in \mathbb{R}^{M \times N}$ 에 대해 $K = f(R) = R^\top R \in \mathbb{R}^{N \times N}$ 일 때 $\frac{dK}{dR} \in \mathbb{R}^{(N \times N) \times (M \times N)}$ 도출:
- $(p, q)$ 번째 원소: $K_{pq} = \mathbf{r}_p^\top \mathbf{r}_q = \sum_{m=1}^M R_{mp} R_{mq}$
- 편미분 $\frac{\partial K_{pq}}{\partial R_{ij}} = \partial_{pqij}$ (Eq 5.98):
  $$\partial_{pqij} = \begin{cases} R_{iq} & \text{if } j = p, \; p \neq q \\\\ R_{ip} & \text{if } j = q, \; p \neq q \\\\ 2R_{iq} & \text{if } j = p, \; p = q \\\\ 0 & \text{otherwise} \end{cases}$$


## 2. ⚔️ Section 5.5: Useful Identities for Computing Gradients (머신러닝 행렬 미분 10대 핵심 항등식)


### 💡 왜 10대 행렬 미분 공식을 외워야 하는가? (수학 치트키!)

고등학교 미분에서 $(x^3)' = 3x^2$ 공식을 외워 1초 만에 풀었듯이, 행렬 미분에서도 행렬 원소를 일일이 쪼개지 않고 행렬 덩어리를 단번에 1초 만에 미분하는 공식(치트키)이 필요합니다.

| 연산 형태 | 고등학교 일변수 스칼라 미분 | 머신러닝 행렬/벡터 미분 항등식 |
| :--- | :--- | :--- |
| 일차식 미분 | $(a x)' = a$ | $\frac{\partial (\mathbf{a}^\top \mathbf{x})}{\partial \mathbf{x}} = \mathbf{a}^\top$ (Eq 5.105) |
| 이차식 미분 | $(b x^2)' = 2bx$ | $\frac{\partial (\mathbf{x}^\top B \mathbf{x})}{\partial \mathbf{x}} = 2\mathbf{x}^\top B$ (대칭 $B$, Eq 5.107) |
| 역수(역행렬) 미분 | $(\frac{1}{x})' = -\frac{1}{x^2}$ | $\frac{\partial X^{-1}}{\partial X_{ij}} = -X^{-1} E_{ij} X^{-1}$ (Eq 5.102) |
| 로그/행렬식 미분 | $(\ln x)' = \frac{1}{x}$ | $\frac{\partial \ln \det(X)}{\partial X} = X^{-1}$ (대칭 $X$) |

---

### 📌 1. 전치 행렬의 미분 (Eq 5.99)
$$\frac{\partial}{\partial X} f(X)^\top = \left( \frac{\partial f(X)}{\partial X} \right)^\top$$

---

### 📌 2. 대각합(Trace)의 미분 (Eq 5.100)
$$\frac{\partial}{\partial X} \text{tr}(f(X)) = \text{tr}\left( \frac{\partial f(X)}{\partial X} \right)$$
- 특히 $\frac{\partial}{\partial X} \text{tr}(AXB) = A^\top B^\top = (BA)^\top$

---

### 📌 3. 행렬식(Determinant)의 미분 (Eq 5.101 - ★ 가우시안 MLE 핵심!)
$$\frac{\partial}{\partial X} \det(f(X)) = \det(f(X)) \text{tr}\left( f(X)^{-1} \frac{\partial f(X)}{\partial X} \right)$$
- 단순 행렬 $X$ 에 대해:
  $$\frac{\partial \det(X)}{\partial X} = \det(X) (X^{-1})^\top$$
- 로그 행렬식(Log-Determinant) 미분 (자주 출제):
  $$\frac{\partial \ln \det(X)}{\partial X} = (X^{-1})^\top = X^{-1} \quad (\text{대칭 행렬 } X)$$

---

### 📌 4. 역행렬(Inverse)의 미분 (Eq 5.102)
$$\frac{\partial}{\partial X} f(X)^{-1} = -f(X)^{-1} \frac{\partial f(X)}{\partial X} f(X)^{-1}$$
- $X X^{-1} = I$ 의 양변을 곱의 미분법으로 전개하여 유도:
  $$\frac{\partial X}{\partial X} X^{-1} + X \frac{\partial X^{-1}}{\partial X} = 0 \implies \frac{\partial X^{-1}}{\partial X_{ij}} = -X^{-1} E_{ij} X^{-1}$$

---

### 📌 5. 역행렬 이차형식의 미분 (Eq 5.103)
$$\frac{\partial (\mathbf{a}^\top X^{-1} \mathbf{b})}{\partial X} = -(X^{-1})^\top \mathbf{a} \mathbf{b}^\top (X^{-1})^\top$$

---

### 📌 6. 선형 일차 형식의 미분 (Eq 5.104~5.105)
$$\frac{\partial (\mathbf{x}^\top \mathbf{a})}{\partial \mathbf{x}} = \mathbf{a}^\top, \quad \frac{\partial (\mathbf{a}^\top \mathbf{x})}{\partial \mathbf{x}} = \mathbf{a}^\top$$

---

### 📌 7. 쌍선형 형식(Bilinear Form)의 미분 (Eq 5.106)
$$\frac{\partial (\mathbf{a}^\top X \mathbf{b})}{\partial X} = \mathbf{a} \mathbf{b}^\top \in \mathbb{R}^{m \times n}$$

---

### 📌 8. 이차 형식(Quadratic Form)의 벡터 미분 (Eq 5.107 - ★ 최다 출제!)
$$\frac{\partial (\mathbf{x}^\top B \mathbf{x})}{\partial \mathbf{x}} = \mathbf{x}^\top (B + B^\top)$$
- 만약 행렬 $B$ 가 대칭 행렬($B = B^\top$) 이면:
  $$\frac{\partial (\mathbf{x}^\top B \mathbf{x})}{\partial \mathbf{x}} = 2\mathbf{x}^\top B \quad (\text{열벡터 기준: } 2B\mathbf{x})$$

---

### 📌 9. 가중 최소제곱 잔차 미분 (Eq 5.108)
대칭 가중치 행렬 $W = W^\top$ 에 대해:
$$\frac{\partial}{\partial \mathbf{s}} \left[ (\mathbf{x} - A\mathbf{s})^\top W (\mathbf{x} - A\mathbf{s}) \right] = -2(\mathbf{x} - A\mathbf{s})^\top W A$$


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 행렬 미분: 행렬 매개변수 $X$ 의 각 성분 $X_{ij}$ 에 대한 목적함수의 편미분 행렬/텐서를 도출하는 미분법입니다.
- 평탄화(Flattening): 행렬 공간 $\mathbb{R}^{m \times n}$ 을 벡터 공간 $\mathbb{R}^{mn}$ 으로 변환하여 텐서 미분을 표준 2차원 야코비안 행렬로 변환하는 기법입니다.
- 행렬 미분 항등식: 행렬식, 역행렬, 이차형식 등의 복잡한 미분을 단번에 계산할 수 있도록 공식화한 수학적 도구입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 수십억 개 가중치 행렬의 동시 미분: 성분 하나하나를 따로 미분하지 않고 행렬 단위로 한 번에 손실함수를 미분하여 그래디언트를 구하기 위해 사용합니다.
- 다변량 가우시안 모델 및 칼만 필터 파라미터 추정: 공분산 행렬 $\Sigma$ 와 역공분산 행렬(정밀도 행렬 $\Lambda = \Sigma^{-1}$)에 대해 로그 우도를 최대화하는 최적해를 유도하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 텐서 표기법(Tensor Notation) vs 행렬 표기법(Matrix Identities):
  - 텐서 표기법: 모든 인덱스($i, j, k, l$)를 엄밀히 추적하지만 수식이 장황해집니다.
  - 행렬 항등식: 전치, 대각합, 역행렬 성질을 활용해 1줄로 우아하게 유도되므로 실제 머신러닝 논문 작성에 압도적으로 유리합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 다변량 정규분포 최대우도추정 (Multivariate Gaussian MLE - Ch 6, 11):
  로그 우도 함수 $\ln p(\mathbf{x}) = -\frac{D}{2}\ln(2\pi) - \frac{1}{2}\ln\det(\Sigma) - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$ 를 $\Sigma^{-1}$ 에 대해 미분할 때 항등식 3, 5번을 적용하여 표본공분산 $\Sigma^* = \frac{1}{N}\sum (\mathbf{x}_n-\boldsymbol{\mu})(\mathbf{x}_n-\boldsymbol{\mu})^\top$ 을 단번에 도출합니다.
- 트랜스포머 Multi-Head Attention 가중치 그래디언트:
  $Q = X W_Q, K = X W_K$ 에 대해 손실함수를 $W_Q, W_K$ 로 미분할 때 쌍선형 미분 항등식 $\frac{\partial (A X B)}{\partial X} = A^\top B^\top$ 이 핵심 역전파로 사용됩니다.
- 릿지 회귀 및 일반화 최소제곱 (Ridge & Weighted Least Squares - Ch 9):
  가중 최소제곱 잔차 미분 항등식(9번)을 통해 정규방정식 $\mathbf{s}^* = (A^\top W A)^{-1} A^\top W \mathbf{x}$ 를 1초 만에 유도합니다.
