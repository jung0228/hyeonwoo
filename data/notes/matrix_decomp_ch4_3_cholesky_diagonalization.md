# 📐 4.3 & 4.4 Cholesky Decomposition & Eigendecomposition (숄레스키 분해와 고유값 분해/대각화)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 4.3 & Section 4.4 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 앞선 절들과의 자연스러운 빌드업: 왜 "분해(Factorization)"를 하는가?

우리는 앞선 4.1절에서 행렬의 크기를 요약하는 행렬식과 대각합을 배웠고, 4.2절에서 행렬 변환의 불변 주축인 고유값과 고유벡터, 그리고 대칭 행렬의 스펙트럴 정리를 배웠습니다.

이제 우리는 이 도구들을 바탕으로, 거대하고 분석하기 어려운 행렬을 해석하기 쉬운 기본 행렬들의 곱(인수분해)으로 쪼개는 실전 행렬 분해 기법으로 나아갑니다.

- 4.3절 숄레스키 분해 (Cholesky Decomposition):
  마치 양의 실수 $9$ 를 $3 \times 3$ 으로 제곱근 분해하듯, 머신러닝의 공분산 행렬과 같은 대칭 양의 정정 행렬(SPD)을 하삼각행렬의 곱 $A = L L^\top$ 으로 쪼개어 가우시안 샘플링과 VAE 역전파의 핵심 도구로 사용합니다.
- 4.4절 고유값 분해 및 대각화 (Eigendecomposition & Diagonalization):
  공간의 좌표축을 고유벡터 기저로 회전 변환하여, 복잡한 행렬 변환을 서로 간섭 없는 독립된 대각 행렬 $A = P D P^{-1}$ 로 변환합니다.


## 1. ⚔️ Section 4.3: Cholesky Decomposition (숄레스키 분해)


### 📌 1. 숄레스키 분해의 정의와 유일성 (Theorem 4.18 & Eq 4.44)

대칭 양의 정정 행렬(Symmetric Positive Definite, SPD) $A \in \mathbb{R}^{n \times n}$ 은 주대각 성분이 모두 양수인 하삼각행렬(Lower-triangular matrix) $L$ 의 곱으로 유일하게(Unique) 분해됩니다:

$$A = L L^\top \quad (\text{Eq 4.44})$$

$$\begin{bmatrix} a_{11} & \dots & a_{1n} \\\\ \vdots & \ddots & \vdots \\\\ a_{n1} & \dots & a_{nn} \end{bmatrix} = \begin{bmatrix} l_{11} & \dots & 0 \\\\ \vdots & \ddots & \vdots \\\\ l_{n1} & \dots & l_{nn} \end{bmatrix} \begin{bmatrix} l_{11} & \dots & l_{n1} \\\\ \vdots & \ddots & \vdots \\\\ 0 & \dots & l_{nn} \end{bmatrix}$$

여기서 하삼각행렬 $L$ 을 행렬 $A$ 의 숄레스키 인자(Cholesky Factor)라고 부르며, 행렬의 제곱근(Square root of matrix) 역할을 수행합니다.


### 📌 2. 3x3 숄레스키 분해 역방향 점화식 유도 (Example 4.10 & Eq 4.45~4.48)

$A = L L^\top$ 의 우변을 실제로 곱하여 계수를 비교하면, 대각 성분과 비대각 성분에 대한 명쾌한 점화식이 도출됩니다:

1. 대각 성분 $l_{ii}$ 계산 (제곱근 연산: Eq 4.47):
   $$l_{11} = \sqrt{a_{11}}$$
   $$l_{22} = \sqrt{a_{22} - l_{21}^2}$$
   $$l_{33} = \sqrt{a_{33} - (l_{31}^2 + l_{32}^2)}$$
   (일반식: $l_{ii} = \sqrt{a_{ii} - \sum_{k=1}^{i-1} l_{ik}^2}$)

2. 비대각 성분 $l_{ij}$ ($i > j$) 계산 (전진 대입: Eq 4.48):
   $$l_{21} = \frac{a_{21}}{l_{11}}$$
   $$l_{31} = \frac{a_{31}}{l_{11}}$$
   $$l_{32} = \frac{a_{32} - l_{31}l_{21}}{l_{22}}$$
   (일반식: $l_{ij} = \frac{1}{l_{jj}} \left( a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk} \right)$)

- 계산 복잡도 및 효율성:
  일반적인 LU 분해($\frac{2}{3}n^3$)와 달리, 대칭성을 활용하므로 절반의 연산량($\frac{1}{3}n^3$)만으로 초고속 분해가 완료되며 수치적으로 극도로 안정적(Numerically Stable)입니다.


### 📌 3. 숄레스키 분해의 초고속 행렬식 계산

숄레스키 분해 $A = L L^\top$ 가 주어지면, 하삼각행렬 $L$ 의 행렬식은 주대각선의 단순 곱이므로 행렬식 계산이 순식간에 끝납니다:

$$\det(A) = \det(L)\det(L^\top) = \det(L)^2 = \left( \prod_{i=1}^n l_{ii} \right)^2 = \prod_{i=1}^n l_{ii}^2$$


## 2. ⚔️ Section 4.4: Eigendecomposition and Diagonalization (고유값 분해와 대각화)


### 📌 1. 대각행렬(Diagonal Matrix)의 엄청난 계산적 장점 (Eq 4.49)

주대각선 성분만 존재하고 비대각 성분이 모두 0인 대각행렬 $D = \text{diag}(c_1, \dots, c_n)$ 은 다변수 간섭이 완벽히 차단된 행렬입니다:
1. 행렬식: $\det(D) = \prod_{i=1}^n c_i$ ($O(n)$ 계산)
2. 거듭제곱: $D^k = \text{diag}(c_1^k, \dots, c_n^k)$ ($O(n)$ 계산)
3. 역행렬: $D^{-1} = \text{diag}(1/c_1, \dots, 1/c_n)$ ($O(n)$ 계산)


### 📌 2. 대각화 가능(Diagonalizable)과 고유값 분해 정리 (Theorem 4.20)

- 대각화 가능의 정의 (Definition 4.19):
  정방행렬 $A \in \mathbb{R}^{n \times n}$ 이 대각행렬 $D$ 와 유사(Similar)할 때, 즉 가역행렬 $P$ 가 존재하여 $D = P^{-1}AP$ (또는 $A = PDP^{-1}$) 로 표현될 때 행렬 $A$ 를 대각화 가능(Diagonalizable)하다고 부릅니다.

- 고유값/고유벡터와의 필연적 연결 (Eq 4.50~4.54):
  $$A P = P D \iff A [\mathbf{p}_1, \dots, \mathbf{p}_n] = [\lambda_1 \mathbf{p}_1, \dots, \lambda_n \mathbf{p}_n] \iff A \mathbf{p}_i = \lambda_i \mathbf{p}_i$$
  따라서 $P$ 의 각 열벡터 $\mathbf{p}_i$ 는 반드시 $A$ 의 고유벡터여야 하고, 대각행렬 $D$ 의 원소는 고유값 $\lambda_i$ 여야 합니다!

- 고유값 분해 정리 (Theorem 4.20: Eigendecomposition):
  행렬 $A \in \mathbb{R}^{n \times n}$ 이 $A = PDP^{-1}$ 로 분해될 필요충분조건은 $A$ 의 고유벡터들이 $\mathbb{R}^{n}$ 의 기저를 형성할 때(즉, 결함 행렬이 아닐 때) 입니다.


### 📌 3. 대칭 행렬의 직교 대각화 (Theorem 4.21 & Spectral Theorem)

대칭 행렬 $S = S^\top$ 은 스펙트럴 정리(Theorem 4.15)에 의해 언제나 대각화가 100% 보장되며, 고유벡터들을 정규직교기저(ONB)로 구성할 수 있으므로 $P$ 는 직교 행렬($P^{-1} = P^\top$) 이 됩니다:

$$S = P D P^\top \quad (\text{단, } P^\top P = I)$$


### 📌 4. 고유값 분해의 기하학적 3단계 직관 (Figure 4.7)

임의의 벡터 $\mathbf{x}$ 에 행렬 $A = PDP^{-1}$ 를 적용하는 과정은 공간의 좌표축을 갈아 끼우는 3단계 변환으로 완벽히 시각화됩니다:

1. 1단계: $P^{-1}$ (기저 변환 / 회전):
   표준 기저 $\mathbf{e}_i$ 기준의 좌표계를 고유벡터 축 $\mathbf{p}_i$ 기준의 고유좌표계(Eigenbasis)로 회전 정렬합니다.
2. 2단계: $D$ (독립된 축 스케일링):
   정렬된 직교 축들을 따라 각각의 고유값 $\lambda_i$ 배만큼 순수하게 늘리거나 줄입니다 (단위 원이 타원으로 팽창).
3. 3단계: $P$ (역회전 / 원래 좌표계 복귀):
   스케일링된 타원을 다시 원래의 표준 기저 좌표계 방향으로 되돌려놓습니다.


### 📌 5. 2x2 행렬 직교 대각화 수치 계산 전수 분석 (Example 4.11)

행렬 $A = \frac{1}{2}\begin{bmatrix} 5 & -2 \\\\ -2 & 5 \end{bmatrix}$ 에 대한 고유값 분해:

1. 특성 다항식 및 고유값 도출 (Eq 4.56):
   $$p_A(\lambda) = \det\begin{bmatrix} 5/2-\lambda & -1 \\\\ -1 & 5/2-\lambda \end{bmatrix} = \lambda^2 - 5\lambda + \frac{21}{4} = \left(\lambda - \frac{7}{2}\right)\left(\lambda - \frac{3}{2}\right) = 0$$
   - 고유값: $\lambda_1 = \frac{7}{2}, \quad \lambda_2 = \frac{3}{2}$

2. 정규직교 고유벡터 구축 (Eq 4.57~4.58):
   $$\mathbf{p}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\\\ -1 \end{bmatrix}, \quad \mathbf{p}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\\\ 1 \end{bmatrix} \quad (\mathbf{p}_1^\top \mathbf{p}_2 = 0, \|\mathbf{p}_i\| = 1)$$

3. 직교 행렬 $P$ 및 대각 행렬 $D$ 완성 (Eq 4.59~4.61):
   $$P = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\\\ -1 & 1 \end{bmatrix}, \quad D = \begin{bmatrix} 7/2 & 0 \\\\ 0 & 3/2 \end{bmatrix}$$
   $$A = P D P^\top = \left(\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\\\ -1 & 1 \end{bmatrix}\right) \begin{bmatrix} 7/2 & 0 \\\\ 0 & 3/2 \end{bmatrix} \left(\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1 \\\\ 1 & 1 \end{bmatrix}\right)$$


### 📌 6. 고유값 분해를 통한 초고속 거듭제곱과 행렬식 계산 (Eq 4.62~4.63)

- $k$차 거듭제곱 ($A^k$):
  $$A^k = (PDP^{-1})(PDP^{-1})\dots(PDP^{-1}) = P D^k P^{-1} = P \begin{bmatrix} \lambda_1^k & \dots & 0 \\\\ \vdots & \ddots & \vdots \\\\ 0 & \dots & \lambda_n^k \end{bmatrix} P^{-1}$$
- 행렬식 계산:
  $$\det(A) = \det(PDP^{-1}) = \det(P)\det(D)\det(P^{-1}) = \det(D) = \prod_{i=1}^n \lambda_i$$


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 숄레스키 분해 ($A = LL^\top$): 대칭 양의 정정 행렬을 하삼각행렬과 그 전치행렬의 곱으로 쪼개는 행렬의 제곱근 분해입니다.
- 고유값 분해/대각화 ($A = PDP^{-1}$): 정방행렬을 고유벡터 기저 행렬 $P$ 와 고유값 대각행렬 $D$ 의 곱으로 분해하여 다변수 얽힘을 해체하는 대각화입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 계산량의 기하급수적 절감: $O(n^3)$ 의 행렬 거듭제곱, 역행렬, 행렬식 연산을 $O(n)$ 대각 연산으로 단축시킵니다.
- 확률 변수의 선형 변환 및 샘플링: 공분산 구조를 갖는 다변수 난수를 표준 정규분포로부터 손쉽게 생성하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 숄레스키($A=LL^\top$) vs 고유값 분해($A=PDP^\top$):
  - 숄레스키 분해: $O(n^3/3)$ 으로 고유값 분해보다 훨씬 빠르고 간단하지만, $A$ 가 반드시 대칭 양의 정정(SPD)이어야만 작동합니다.
  - 고유값 분해: $O(n^3)$ 으로 연산량이 더 들지만, 고유축과 고유스펙트럼 전체를 분리해 공간의 변환 특성을 완벽히 분석할 수 있습니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 변분 오토인코더(VAE)의 Reparameterization Trick (Kingma & Welling 2014):
  잠재 공간 $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ 에서 샘플링할 때 역전파가 불가능한 문제를 해결하기 위해, 숄레스키 인자 $L$ 을 이용해 $\mathbf{z} = \boldsymbol{\mu} + L \boldsymbol{\epsilon}$ ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$) 로 변환하여 엔드투엔드 미분을 가능하게 만듭니다.
- 가우시안 프로세스 회귀(Gaussian Process Regression - GPR):
  커널 행렬 $K$ 의 역행렬과 로그 행렬식을 계산할 때 수치적 안정성을 위해 반드시 숄레스키 분해 $K = L L^\top$ 를 거쳐 계산합니다.
- 마르코프 연쇄 및 PageRank 거듭제곱 고속화:
  전이 행렬 $T$ 의 $k$단계 상태 확률 $T^k \mathbf{x}_0$ 을 고유값 분해 $P D^k P^{-1} \mathbf{x}_0$ 로 계산하여 $k=10000$ 단계 이후의 정상 상태를 단번에 도출합니다.
