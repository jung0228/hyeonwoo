# 📐 3.7 & 3.8 Inner Product of Functions and Orthogonal Projections (함수의 내적과 직교 정사영)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 3.7 & 3.8 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 3.5/3.6절과의 연결 및 자연스러운 빌드업: 왜 "함수 내적"과 "직교 정사영"을 배우는가?

우리는 지난 3.5절과 3.6절에서 정규직교기저(ONB)와 직교 여공간($U^\perp$)을 통해 공간을 수직 성분들로 분해하는 방법을 배웠습니다.

이제 3.7절에서는 유한차원 벡터($\mathbb{R}^n$)의 성분별 곱의 합($\sum x_i y_i$)을 연속적인 무한차원 함수 공간으로 확장하여 적분 형태($\int u(x)v(x)dx$)로 정의되는 함수의 내적을 살펴봅니다.

그리고 3.8절에서는 인공지능과 머신러닝의 가장 핵심적인 연산이자 차원 축소의 기하학적 본체인 직교 정사영(Orthogonal Projection)을 1차원 직선, 일반 $m$차원 부분공간, 그리고 붕 떠있는 어파인 공간(Affine Space)까지 완벽하게 정복합니다.


## 1. ⚔️ Section 3.7: Inner Product of Functions (함수의 내적)


### 📌 1. 벡터에서 함수로의 연속적 확장 (Eq 3.37)

$n$차원 벡터 $\mathbf{x} \in \mathbb{R}^n$ 은 $n$개의 함수값을 가지는 이산 함수로 해석할 수 있습니다. 벡터의 성분 수가 가산 무한(Countably infinite) 또는 비가산 무한(Uncountably infinite)인 연속 함수로 확장되면, 성분들의 합(Summation)은 자연스럽게 정적분(Definite Integral)으로 바뀝니다.

두 연속 함수 $u: \mathbb{R} \to \mathbb{R}$ 와 $v: \mathbb{R} \to \mathbb{R}$ 의 내적은 구간 $[a, b]$ 상에서의 정적분으로 정의됩니다:

$$\langle u, v \rangle := \int_a^b u(x) v(x) \, dx \quad (\text{Eq 3.37})$$

- 함수 노름(길이): $\Vertu\Vert = \sqrt{\langle u, u \rangle} = \sqrt{\int_a^b u(x)^2 \, dx}$
- 함수의 직교성: $\langle u, v \rangle = \int_a^b u(x) v(x) \, dx = 0 \iff u \perp v$
- 힐베르트 공간(Hilbert Space): 이러한 함수 내적이 수렴하고 완비성(Completeness)을 갖추도록 측도론(Measure Theory)을 엄밀히 적용한 무한차원 내적 공간을 힐베르트 공간이라 부릅니다.


### 📌 2. 함수 직교성의 대표 예시 (Example 3.9 & Remark: Eq 3.38)

1. 삼각함수 $\sin(x)$ 와 $\cos(x)$ 의 직교성 (Example 3.9)
   구간 $[-\pi, \pi]$ 에서 두 함수의 내적을 계산해 봅시다:
   $$\langle \sin(x), \cos(x) \rangle = \int_{-\pi}^{\pi} \sin(x) \cos(x) \, dx$$
   - 피적분 함수 $f(x) = \sin(x)\cos(x)$ 는 $f(-x) = \sin(-x)\cos(-x) = -\sin(x)\cos(x) = -f(x)$ 로 원점 대칭인 기함수(Odd Function)입니다.
   - 대칭 구간 $[-\pi, \pi]$ 에서 기함수의 정적분 값은 정확히 0이므로, 두 함수는 상호 직교합니다 ($\sin(x) \perp \cos(x)$).

2. 푸리에 급수(Fourier Series)의 직교 함수 집합 (Eq 3.38)
   구간 $[-\pi, \pi]$ 상에서 다음 함수 집합은 모든 원소끼리 서로 완벽히 직교합니다:
   $$\{1, \cos(x), \cos(2x), \cos(3x), \dots\} \quad (\text{Eq 3.38})$$
   - 이 직교 함수 집합은 $[-\pi, \pi)$ 에서 정의된 모든 주기 우함수(Even periodic functions) 공간을 생성(Span)합니다.
   - 임의의 복잡한 신호 함수를 이 직교 부분공간 위로 정사영시키는 것이 바로 푸리에 급수(Fourier Series)의 본질입니다.


## 2. ⚔️ Section 3.8: Orthogonal Projections (직교 정사영)


### 📌 1. 정사영의 정의와 사영 행렬의 멱등성 (Definition 3.10)

벡터 공간 $V$ 와 부분공간 $U \subseteq V$ 에 대해, 선형 사상 $\pi: V \to U$ 가 자기 자신과의 합성을 거듭해도 결과가 변하지 않는 멱등성(Idempotency)을 만족할 때 정사영(Projection)이라 부릅니다:

$$\pi^2 = \pi \circ \pi = \pi \quad (\text{Definition 3.10})$$

- 사영 행렬(Projection Matrix $P_\pi$): 사영 변환을 행렬로 표현한 정방행렬 $P_\pi$ 역시 멱등 행렬 성질을 가집니다:
  $$P_\pi^2 = P_\pi$$
  (이미 부분공간 $U$ 위로 내려앉은 점을 다시 정사영해도 그 자리에 그대로 머무릅니다.)


### 📌 2. 1차원 부분공간(직선)으로의 직교 정사영 (Section 3.8.1 & Eq 3.39~3.46)

원점을 지나는 1차원 직선 부분공간 $U = \text{span}[\mathbf{b}]$ 위로 임의의 벡터 $\mathbf{x} \in \mathbb{R}^n$ 을 정사영하는 과정은 "공중에 뜬 점을 바닥 레일 위로 가장 짧은 거리(수직)로 착륙시키는 3단계 과정"으로 유도됩니다:


#### 💡 [왜 수식을 이렇게 3단계로 유도하는가? (직관적 원리 해설)]

1. 1단계: 왜 배율 $\lambda$(스칼라)를 먼저 구하는가?
   - 목적: 어디에 착지할지 모르는 고차원 벡터 문제를 "기준 벡터 $\mathbf{b}$ 를 몇 배($\lambda$) 늘릴 것인가?"라는 숫자 하나 구하는 문제로 단순화하기 위함입니다.
   - 단서: 최단거리로 착지하려면 착지선 오차 벡터 $\mathbf{x} - \lambda \mathbf{b}$ 가 바닥 레일 방향 $\mathbf{b}$ 와 무조건 직각($90^\circ$)이어야 합니다.
   - 수식 전개 (Eq 3.39~3.41):
     $$\langle \mathbf{x} - \lambda \mathbf{b}, \mathbf{b} \rangle = 0 \iff \langle \mathbf{x}, \mathbf{b} \rangle - \lambda \langle \mathbf{b}, \mathbf{b} \rangle = 0$$
     $$\lambda = \frac{\langle \mathbf{x}, \mathbf{b} \rangle}{\langle \mathbf{b}, \mathbf{b} \rangle} = \frac{\mathbf{b}^\top \mathbf{x}}{\Vert\mathbf{b}\Vert^2} \quad (\text{Eq 3.40~3.41})$$
   - 직관: 분자 $\mathbf{b}^\top \mathbf{x}$ 는 두 벡터가 같은 방향으로 얼마나 겹쳐있는지(그림자 크기)를 나타내며, 분모 $\Vert\mathbf{b}\Vert^2$ 는 기준 벡터 $\mathbf{b}$ 의 길이 효과를 나누어 순수한 배율만 남겨주는 정규화 역할을 합니다.
   - 기저 $\mathbf{b}$ 가 단위 벡터($\Vert\mathbf{b}\Vert=1$)라면 분모가 1이 되어 좌표는 단순히 $\lambda = \mathbf{b}^\top \mathbf{x}$ 가 됩니다.

2. 2단계: 왜 사영 벡터를 $\pi_U(\mathbf{x}) = \lambda \mathbf{b}$ 로 쓰는가?
   - 목적: 1단계에서 구한 배율 $\lambda$ 는 단순한 숫자일 뿐이므로, 실제 공간 상의 위치 좌표(착지점 위치 벡터)로 복원하기 위해 기준 방향 벡터 $\mathbf{b}$ 에 배율 $\lambda$ 를 곱해줍니다.
   - 수식 도출 (Eq 3.42):
     $$\pi_U(\mathbf{x}) = \lambda \mathbf{b} = \left( \frac{\mathbf{b}^\top \mathbf{x}}{\Vert\mathbf{b}\Vert^2} \right) \mathbf{b}$$
   - 사영 벡터의 길이: $\Vert\pi_U(\mathbf{x})\Vert = |\lambda| \Vert\mathbf{b}\Vert = |\cos\omega| \Vert\mathbf{x}\Vert$ (삼각법과 완벽 일치, Eq 3.44).

3. 3단계: 왜 굳이 사영 행렬 $P_\pi = \frac{\mathbf{b}\mathbf{b}^\top}{\mathbf{b}^\top \mathbf{b}}$ 로 변환하는가?
   - 목적: 매번 어떤 데이터 벡터 $\mathbf{x}$ 가 들어올 때마다 일일이 내적하고 나누는 연산을 반복하지 않고, "어떤 벡터든 앞에 곱하기만 하면 레일 위로 즉시 떨어뜨려 주는 만능 기계(선형 변환 행렬)"를 만들기 위함입니다.
   - 수식 결합법칙 변형 (Eq 3.45~3.46):
     $$\pi_U(\mathbf{x}) = \mathbf{b} \lambda = \mathbf{b} \left( \frac{\mathbf{b}^\top \mathbf{x}}{\Vert\mathbf{b}\Vert^2} \right) = \left( \frac{\mathbf{b} \mathbf{b}^\top}{\Vert\mathbf{b}\Vert^2} \right) \mathbf{x}$$
     $$P_\pi = \frac{\mathbf{b} \mathbf{b}^\top}{\Vert\mathbf{b}\Vert^2} = \frac{\mathbf{b} \mathbf{b}^\top}{\mathbf{b}^\top \mathbf{b}} \quad (\text{Rank 1 대칭 행렬, Eq 3.46})$$

- 분모와 분자의 본질적 형태 차이:
  - 분모 $\mathbf{b}^\top \mathbf{b}$ (내적): $(1 \times n) \times (n \times 1) =$ 스칼라 (숫자 하나, 길이의 제곱).
  - 분자 $\mathbf{b} \mathbf{b}^\top$ (외적): $(n \times 1) \times (1 \times n) =$ $n \times n$ 정방 행렬 (공간 전체를 직선으로 눌러버리는 변환 기계).


#### 💡 [Example 3.10: 1차원 직선 정사영 수치 계산 예제]
- 직선 기저 방향 $\mathbf{b} = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}$, 정사영할 벡터 $\mathbf{x} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$
- 분모 계산: $\mathbf{b}^\top \mathbf{b} = 1^2 + 2^2 + 2^2 = 9$
- 사영 행렬 $P_\pi$ 구축:
  $$P_\pi = \frac{1}{9} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix} \begin{bmatrix} 1 & 2 & 2 \end{bmatrix} = \frac{1}{9} \begin{bmatrix} 1 & 2 & 2 \\ 2 & 4 & 4 \\ 2 & 4 & 4 \end{bmatrix}$$
- 정사영 벡터 $\pi_U(\mathbf{x})$ 계산:
  $$\pi_U(\mathbf{x}) = P_\pi \mathbf{x} = \frac{1}{9} \begin{bmatrix} 1 & 2 & 2 \\ 2 & 4 & 4 \\ 2 & 4 & 4 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \frac{1}{9} \begin{bmatrix} 5 \\ 10 \\ 10 \end{bmatrix} = \frac{5}{9} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix} \in \text{span}[\mathbf{b}]$$


### 📌 3. 일반 $m$차원 부분공간으로의 직교 정사영 (Section 3.8.2 & Eq 3.49~3.66)

부분공간 $U \subseteq \mathbb{R}^n$ 의 순서기저가 $(\mathbf{b}_1, \dots, \mathbf{b}_m)$ 일 때, 기저들을 열벡터로 쌓은 행렬을 $B = [\mathbf{b}_1 \mid \dots \mid \mathbf{b}_m] \in \mathbb{R}^{n \times m}$ 이라 합니다.

1. 1단계: 정규 방정식(Normal Equation)과 좌표 $\boldsymbol{\lambda}$ 도출
   - 사영점 $\pi_U(\mathbf{x}) = B \boldsymbol{\lambda}$ 에 대해 오차 벡터 $\mathbf{x} - B \boldsymbol{\lambda}$ 는 $U$ 의 모든 기저 벡터 $\mathbf{b}_i$ 와 직교해야 합니다:
     $$B^\top (\mathbf{x} - B \boldsymbol{\lambda}) = \mathbf{0} \iff B^\top B \boldsymbol{\lambda} = B^\top \mathbf{x} \quad (\text{Normal Equation: Eq 3.55~3.56})$$
   - 기저들이 선형독립이므로 $B^\top B \in \mathbb{R}^{m \times m}$ 은 역행렬이 존재합니다:
     $$\boldsymbol{\lambda} = (B^\top B)^{-1} B^\top \mathbf{x} \quad (\text{Eq 3.57})$$
   - 여기서 $(B^\top B)^{-1} B^\top$ 을 행렬 $B$ 의 좌측 의사역행렬(Left Pseudo-inverse)이라 부릅니다.

2. 2단계: 사영점 $\pi_U(\mathbf{x})$ 및 사영 행렬 $P_\pi$ 도출
   $$\pi_U(\mathbf{x}) = B \boldsymbol{\lambda} = B (B^\top B)^{-1} B^\top \mathbf{x} \quad (\text{Eq 3.58})$$
   $$P_\pi = B (B^\top B)^{-1} B^\top \quad (\text{Eq 3.59})$$

3. 3단계: 정규직교기저(ONB)일 때의 극단적 단순화 (Eq 3.65~3.66)
   - 만약 기저들이 정규직교기저(ONB)라면 $B^\top B = I$ 가 되므로 복잡한 역행렬 연산이 완전히 소거됩니다:
     $$\boldsymbol{\lambda} = B^\top \mathbf{x}, \quad \pi_U(\mathbf{x}) = B B^\top \mathbf{x}, \quad P_\pi = B B^\top$$


#### 💡 [Example 3.11: 2차원 부분공간 정사영 전수 손풀기 예제]
- 부분공간 $U = \text{span}\left( \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} \right) \subseteq \mathbb{R}^3$, 사영할 벡터 $\mathbf{x} = \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix}$
- 기저 행렬 $B = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}$

- 1단계: $B^\top B$ 및 $B^\top \mathbf{x}$ 계산
  $$B^\top B = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}$$
  $$B^\top \mathbf{x} = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 6 \\ 0 \end{bmatrix}$$

- 2단계: 정규방정식 풀어서 좌표 $\boldsymbol{\lambda}$ 구하기
  $$\begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix} \begin{bmatrix} \lambda_1 \\ \lambda_2 \end{bmatrix} = \begin{bmatrix} 6 \\ 0 \end{bmatrix} \implies \boldsymbol{\lambda} = \begin{bmatrix} 5 \\ -3 \end{bmatrix}$$

- 3단계: 사영점 $\pi_U(\mathbf{x})$ 계산
  $$\pi_U(\mathbf{x}) = B \boldsymbol{\lambda} = 5 \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} - 3 \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 5 \\ 2 \\ -1 \end{bmatrix}$$

- 4단계: 사영 오차(재구성 오차, Reconstruction Error) 계산
  $$\Vert\mathbf{x} - \pi_U(\mathbf{x})\Vert = \left\Vert \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix} - \begin{bmatrix} 5 \\ 2 \\ -1 \end{bmatrix} \right\Vert = \left\Vert \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} \right\Vert = \sqrt{1^2 + (-2)^2 + 1^2} = \sqrt{6}$$

- 5단계: 사영 행렬 $P_\pi$ 계산
  $$P_\pi = B (B^\top B)^{-1} B^\top = \frac{1}{6} \begin{bmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{bmatrix}$$


### 📌 4. 정사영 관점의 그람-슈미트 직교화 (Section 3.8.3 & Example 3.12)

그람-슈미트 과정은 본질적으로 직교 정사영을 반복 적용하는 알고리즘입니다:

$$\mathbf{u}_1 := \mathbf{b}_1$$
$$\mathbf{u}_k := \mathbf{b}_k - \pi_{\text{span}[\mathbf{u}_1, \dots, \mathbf{u}_{k-1}]}(\mathbf{b}_k) \quad (k = 2, \dots, n, \text{ Eq 3.68})$$

- Example 3.12: $\mathbf{b}_1 = \begin{bmatrix} 2 \\ 0 \end{bmatrix}, \mathbf{b}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \in \mathbb{R}^2$
  $$\mathbf{u}_1 = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$$
  $$\mathbf{u}_2 = \mathbf{b}_2 - \frac{\mathbf{u}_1 \mathbf{u}_1^\top}{\Vert\mathbf{u}_1\Vert^2} \mathbf{b}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} - \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix} - \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
  두 벡터 $\mathbf{u}_1, \mathbf{u}_2$ 는 완벽히 직교합니다 ($\mathbf{u}_1^\top \mathbf{u}_2 = 0$).


### 📌 5. 어파인 부분공간으로의 직교 정사영 (Section 3.8.4 & Eq 3.72~3.73)

원점을 지나지 않고 붕 떠있는 어파인 공간 $L = \mathbf{x}_0 + U$ (지지점 $\mathbf{x}_0$ 와 방향 부분공간 $U$) 위로의 정사영은 다음 3단계 이동으로 해결합니다 (Figure 3.13):

1. 지지점 빼기: 문제 전체를 원점 통과 벡터공간으로 평행이동 ($L - \mathbf{x}_0 = U, \quad \mathbf{x} - \mathbf{x}_0$)
2. 부분공간 정사영: 이동된 벡터 $\mathbf{x} - \mathbf{x}_0$ 를 방향공간 $U$ 위로 직교 정사영 ($\pi_U(\mathbf{x} - \mathbf{x}_0)$)
3. 지지점 다시 더하기: 원상태의 어파인 공간으로 복귀

$$\pi_L(\mathbf{x}) = \mathbf{x}_0 + \pi_U(\mathbf{x} - \mathbf{x}_0) \quad (\text{Eq 3.72})$$

- 어파인 공간까지의 최단 거리:
  $$d(\mathbf{x}, L) = \Vert\mathbf{x} - \pi_L(\mathbf{x})\Vert = \Vert\mathbf{x} - (\mathbf{x}_0 + \pi_U(\mathbf{x} - \mathbf{x}_0))\Vert = d(\mathbf{x} - \mathbf{x}_0, U) \quad (\text{Eq 3.73})$$


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 직교 정사영 (Orthogonal Projection): 고차원 벡터 $\mathbf{x}$ 를 저차원 부분공간 $U$ 상에서 가장 거리가 가까운 최적의 근사점 $\pi_U(\mathbf{x})$ 로 수직 낙하시키는 선형 변환입니다.
- 사영 행렬 $P_\pi = B(B^\top B)^{-1}B^\top$: 멱등성($P_\pi^2 = P_\pi$)과 대칭성($P_\pi^\top = P_\pi$)을 갖는 변환 행렬입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 고차원 데이터의 최적 차원 압축: 고차원 데이터에서 핵심적인 저차원 정보만 추출하고 압축 손실(재구성 오차 $\Vert\mathbf{x} - \pi_U(\mathbf{x})\Vert$)을 수학적으로 최소화하기 위해 사용합니다.
- 해가 없는 연립방정식의 최적 근사해 도출: $A\mathbf{x} = \mathbf{b}$ 에서 $\mathbf{b}$ 가 열공간에 없어 해가 존재하지 않을 때, 열공간 위로 직교 정사영된 최단거리 근사해(최소제곱해)를 구하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 일반 기저 vs 정규직교기저(ONB)의 정사영 연산 비용:
  - 일반 기저 행렬 $B$: $P_\pi = B(B^\top B)^{-1}B^\top$ 로 $m \times m$ 역행렬 연산 필요 ($O(m^3)$).
  - 정규직교기저 행렬 $Q$: $P_\pi = Q Q^\top$ 로 역행렬 연산 없이 단순 행렬곱만으로 즉시 사영 완료 ($O(nm)$).
- 정규방정식의 수치적 불안정성과 릿지(Ridge) 보정:
  - $B^\top B$ 가 특이행렬에 가까워 역행렬 계산이 불안정할 때, 대각선에 작은 값 $\epsilon I$ 를 더해주는 $(B^\top B + \epsilon I)^{-1}$ 기법이 머신러닝의 릿지 회귀(Ridge Regression / Weight Decay)의 수학적 원형입니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 선형 회귀 (Linear Regression - Ch 9): 타겟 벡터 $\mathbf{y}$ 를 데이터 특성 행렬 $X$ 의 열공간으로 직교 정사영시켜 최적 가중치 $\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$ 를 구하는 원리 그 자체입니다.
- PCA (주성분 분석 - Ch 10): 고차원 데이터를 재구성 오차 $\Vert\mathbf{x} - \pi_U(\mathbf{x})\Vert^2$ 가 최소가 되도록 주성분 부분공간으로 직교 정사영시키는 알고리즘입니다.
- 오토인코더 (Auto-Encoder): 고차원 입력을 저차원 잠재 공간(Latent Space)으로 사영(인코더)하고 다시 복원(디코더)하는 비선형 정사영 확장 모델입니다.
- SVM 분리 초평면 (Support Vector Machine - Ch 12): 데이터 점들을 어파인 초평면 $L = \{\mathbf{x} \mid \mathbf{w}^\top \mathbf{x} + b = 0\}$ 위로 직교 정사영하여 마진(Margin) 거리를 최대화하는 분류기를 학습합니다.
