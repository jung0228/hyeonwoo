# 📐 6.5 Gaussian Distribution (가우시안 정규분포, 주변/조건부 닫힘성, 아핀변환과 숄레스키 샘플링)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 6.5 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 확률론의 왕: 왜 "가우시안 분포(Gaussian Distribution)"인가?

가우시안 분포(Normal Distribution)는 머신러닝과 통계학에서 가장 널리 연구되고 쓰이는 연속 확률분포의 절대적 제왕입니다.

- 중심극한정리 (Central Limit Theorem, CLT): 독립인 무수한 확률변수들의 합은 원래 어떤 분포였든 상관없이 결국 가우시안 분포로 수렴합니다.
- 연산의 닫힘성 (Closed-Form Closure): 가우시안 분포끼리 결합하면 주변분포, 조건부분포, 아핀 변환, 밀도의 곱까지 모두 100% 예외 없이 가우시안 분포로 귀환합니다.
- 최대 엔트로피 성질: 평균과 분산이 고정되어 있을 때 불확실성(Entropy)을 가장 크게 만드는 가장 자연스럽고 무작위적인 분포입니다.


## 1. ⚔️ Section 6.5: 가우시안 분포의 수학적 정의 (Eq 6.62~6.63)


### 📌 1. 단변량 및 다변량 가우시안 PDF

1. 단변량 가우시안 (Univariate Gaussian: Eq 6.62):
   $$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

2. 다변량 가우시안 (Multivariate Gaussian: Eq 6.63):
   $$\mathbf{x} \in \mathbb{R}^D \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$
   $$p(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma) = (2\pi)^{-\frac{D}{2}} |\Sigma|^{-\frac{1}{2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$

- 파라미터: 평균 벡터 $\boldsymbol{\mu} \in \mathbb{R}^D$, 공분산 행렬 $\Sigma \in \mathbb{R}^{D \times D}$ (대칭 양의 정정 행렬 $\Sigma \succ 0$).
- 표준 정규분포 (Standard Normal Distribution): $\boldsymbol{\mu} = \mathbf{0}, \Sigma = I$ 인 특수 경우.
- 마하노비스 거리 (Mahalanobis Distance): 지수부의 $(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$ 는 공분산의 곡률을 고려한 평균으로부터의 거리를 나타냅니다.


## 2. ⚔️ Section 6.5.1: 주변분포와 조건부분포의 닫힘 성질 (★ 핵심 수식 유도)


### 📌 1. 결합 가우시안의 블록 표현 (Eq 6.64)

두 다변량 확률변수 $\mathbf{x}, \mathbf{y}$ 의 결합 가우시안 분포 $p(\mathbf{x}, \mathbf{y})$ 는 다음과 같이 블록 행렬로 표현됩니다:

$$p(\mathbf{x}, \mathbf{y}) = \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_x \\\\ \boldsymbol{\mu}_y \end{bmatrix}, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\\\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix} \right) \quad (\text{Eq 6.64})$$


### 📌 2. 조건부 가우시안 분포 $p(\mathbf{x} \mid \mathbf{y})$ (Eq 6.65~6.67 - ★ 칼만 필터 & GP의 심장!)

$\mathbf{y}$ 가 관측되었을 때 조건부 분포 $p(\mathbf{x} \mid \mathbf{y})$ 역시 완벽한 가우시안 분포가 됩니다:

$$p(\mathbf{x} \mid \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_{x \mid y}, \Sigma_{x \mid y}) \quad (\text{Eq 6.65})$$

$$\boldsymbol{\mu}_{x \mid y} = \boldsymbol{\mu}_x + \Sigma_{xy} \Sigma_{yy}^{-1} (\mathbf{y} - \boldsymbol{\mu}_y) \quad (\text{Eq 6.66})$$

$$\Sigma_{x \mid y} = \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx} \quad (\text{Eq 6.67})$$

- 직관적 해석: 관측값 $\mathbf{y}$ 가 기댓값 $\boldsymbol{\mu}_y$ 와 차이가 나면, 교차 공분산 $\Sigma_{xy} \Sigma_{yy}^{-1}$ 만큼 $\mathbf{x}$ 의 평균을 갱신합니다. 또한 관측을 얻었으므로 분산은 $\Sigma_{xx}$ 보다 불확실성이 줄어듭니다!
- 연관 알고리즘: 칼만 필터(Kalman Filter) 및 가우시안 프로세스(Gaussian Process) 의 사후분포 갱신 공식과 100% 동일합니다.


### 📌 3. 주변 가우시안 분포 $p(\mathbf{x})$ (Eq 6.68)

결합 가우시안에서 $\mathbf{y}$ 를 적분 소거한 주변 분포 $p(\mathbf{x})$ 도 가우시안입니다:

$$p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{y}) d\mathbf{y} = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_x, \Sigma_{xx}) \quad (\text{Eq 6.68})$$


### 💡 [Example 6.6: 2변수 가우시안 수치 계산]
$p(x_1, x_2) = \mathcal{N}\left( \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \begin{bmatrix} 0.3 & -1 \\ -1 & 5 \end{bmatrix} \right)$ 에서 $x_2 = -1$ 이 관측된 경우:
- 조건부 평균: $\mu_{x_1 \mid x_2 = -1} = 0 + (-1) \cdot 5^{-1} \cdot (-1 - 2) = 0 + (-0.2)(-3) = 0.6$
- 조건부 분산: $\sigma^2_{x_1 \mid x_2 = -1} = 0.3 - (-1) \cdot 5^{-1} \cdot (-1) = 0.3 - 0.2 = 0.1$
- 최종 조건부 분포: $p(x_1 \mid x_2 = -1) = \mathcal{N}(0.6, 0.1)$


## 3. ⚔️ Section 6.5.2 ~ 6.5.4: 밀도의 곱, 아핀변환, 혼합 분포와 숄레스키 샘플링


### 📌 1. 가우시안 밀도의 곱 (Product of Densities: Eq 6.74~6.76)

두 가우시안 밀도 $\mathcal{N}(\mathbf{x} \mid \mathbf{a}, A)$ 와 $\mathcal{N}(\mathbf{x} \mid \mathbf{b}, B)$ 의 곱은 스케일링 상수 $c$ 가 곱해진 가우시안 밀도가 됩니다:

$$\mathcal{N}(\mathbf{x} \mid \mathbf{a}, A) \mathcal{N}(\mathbf{x} \mid \mathbf{b}, B) = c \cdot \mathcal{N}(\mathbf{x} \mid \mathbf{c}, C)$$

$$C = (A^{-1} + B^{-1})^{-1}, \quad \mathbf{c} = C(A^{-1}\mathbf{a} + B^{-1}\mathbf{b}) \quad (\text{Eq 6.74~6.75})$$

- 선형 회귀(Linear Regression - Ch 9)에서 Likelihood $\times$ Prior 의 베이즈 정리 사후분포 계산의 핵심 기초입니다.


### 📌 2. 아핀 변환의 닫힘성 (Eq 6.88)

$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ 에 대해 아핀 변환 $\mathbf{y} = A\mathbf{x} + \mathbf{b}$ 가 적용되면 $\mathbf{y}$ 도 가우시안입니다:

$$p(\mathbf{y}) = \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\Sigma A^\top) \quad (\text{Eq 6.88})$$


### 📌 3. 가우시안 혼합 분포 (GMM: Theorem 6.12 & Eq 6.80~6.85)

$p(x) = \alpha p_1(x) + (1-\alpha) p_2(x)$ 에 대해:
- 평균: $\mathbb{E}[x] = \alpha \mu_1 + (1-\alpha) \mu_2 \quad (\text{Eq 6.81})$
- 분산 (전분산 법칙 Law of Total Variance: Eq 6.82):
  $$V[x] = [\alpha \sigma_1^2 + (1-\alpha)\sigma_2^2] + [\alpha \mu_1^2 + (1-\alpha)\mu_2^2] - [\alpha \mu_1 + (1-\alpha)\mu_2]^2$$


### 📌 4. 숄레스키 분해를 이용한 다변량 가우시안 샘플링 (Section 6.5.4 ★ VAE의 조상!)

임의의 다변량 가우시안 $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ 로부터 컴퓨터로 샘플을 뽑는 방법:

1. 표준정규분포 $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, I)$ 샘플을 생성합니다.
2. 공분산 행렬 $\Sigma$ 의 숄레스키 분해(Cholesky Decomposition: $\Sigma = L L^\top$)를 수행하여 하삼각행렬 $L$ 을 구합니다.
3. 선형 변환 $\mathbf{y} = L\mathbf{x} + \boldsymbol{\mu}$ 를 적용하면 완벽하게 $\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ 가 유도됩니다!

- ★ 딥러닝 VAE의 재파라미터화 트릭 (Reparameterization Trick: $\mathbf{z} = \boldsymbol{\mu} + \mathbf{L} \odot \boldsymbol{\epsilon}$) 의 직계 수학적 기반입니다!


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 다변량 가우시안 분포 ($\mathcal{N}(\boldsymbol{\mu}, \Sigma)$): 평균 벡터 $\boldsymbol{\mu}$ 와 공분산 행렬 $\Sigma$ 에 의해 완벽히 규정되는 연속 확률분포의 대표 모델입니다.
- 가우시안 연산의 닫힘성: 주변분포, 조건부분포, 아핀 변환, 밀도의 곱이 모두 100% 가우시안으로 유지되는 닫힌 형태(Closed-form) 성질입니다.
- 숄레스키 변환 샘플링: $\Sigma = LL^\top$ 분해를 이용해 표준정규분포 샘플을 임의의 공분산 분포 샘플로 선형 변환하는 기법입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 해석적(Closed-Form) 사후분포 계산: 베이즈 정리 및 선형 회귀에서 사후분포를 복잡한 numerical 적분 없이 수식으로 완벽히 닫힌 형태로 구하기 위해 사용합니다.
- 데이터 생성 모델의 샘플링: VAE나 가우시안 프로세스에서 역전파가 가능한 선형 아핀 변환 형태($\mathbf{y} = L\mathbf{x} + \boldsymbol{\mu}$)로 난수를 생성하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 단순 가우시안 vs 가우시안 혼합 모델 (GMM):
  - 단일 가우시안: 단봉(Unimodal) 형태만 표현 가능하므로 복잡한 복수 클러스터 데이터를 표현하지 못함.
  - 가우시안 혼합 모델 (GMM): K개의 가우시안을 조합하여 임의의 복잡한 다봉(Multimodal) 손실 곡면과 데이터 분포를 근사할 수 있음.


### 4️⃣ [4단계 실전 AI 연결고리]
- 선형 회귀 (Linear Regression - Ch 9):
  가우시안 노이즈 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 를 가정한 Likelihood 해석과 베이지안 릿지 회귀 사후분포 유도.
- 변분 자가부호화기 (VAE - Reparameterization Trick):
  잠재 공간 $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2(\mathbf{x})))$ 에서 그래디언트 역전파를 위해 $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ 선형 아핀 샘플링 적용.
- 가우시안 프로세스 (Gaussian Process - GP):
  무한 차원 함수의 결합 분포를 다변량 가우시안으로 정의하고, 관측 데이터가 들어왔을 때 조건부 가우시안 공식(Eq 6.66~6.67)으로 미지의 함수값 사후분포 예측.
