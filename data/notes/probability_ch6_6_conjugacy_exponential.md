# 📐 6.6 Conjugacy and the Exponential Family (공액 사전분포, 충분통계량과 지수 족)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 6.6 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 확률 모델링의 통일 이론: 왜 "공액성과 지수 족"인가?

통계학 교재에는 수많은 유서 깊은 확률분포들(가우시안, 베르누이, 이항, 베타, 디리클레 등)이 등장하지만, 머신러닝 연산 환경에서는 다음 3가지 조건이 필수적입니다:

1. 연산의 닫힘성 (Closure): 사후분포(Posterior)를 계산했을 때 사전분포(Prior)와 동일한 분포 형태를 유지할 것.
2. 파라미터 개수의 고정성 (Finite Parameters): 데이터가 무한히 늘어나도 이를 대표하는 파라미터 개수가 고정될 것.
3. 볼록 최적화 (Concave Optimization): 우도(Likelihood)의 로그값이 오목(Concave)하여 경사하강법으로 전역 최적해(Global Optimum)를 찾을 수 있을 것.

지수 족(Exponential Family) 은 수많은 분포들을 하나로 통일하는 거대한 수학적 족(Family)이며, 공액 사전분포(Conjugate Prior) 와 충분통계량(Sufficient Statistics) 을 통해 위 3가지 조건을 100% 완벽히 충족시킵니다.


## 1. ⚔️ Section 6.6: 베르누이, 이항, 베타 분포 (Examples 6.8 ~ 6.10)


### 📌 1. 베르누이 분포 (Bernoulli Distribution: Example 6.8 & Eq 6.92~6.94)

단일 이진 확률변수 $X \in \{0, 1\}$ (성공 확률 $\mu \in [0, 1]$) 에 대한 분포:

$$p(x \mid \mu) = \mu^x (1 - \mu)^{1-x}, \quad x \in \{0, 1\} \quad (\text{Eq 6.92})$$

$$\mathbb{E}[x] = \mu, \quad V[x] = \mu(1 - \mu) \quad (\text{Eq 6.93~6.94})$$


### 📌 2. 이항 분포 (Binomial Distribution: Example 6.9 & Eq 6.95~6.97)

베르누이 시도를 $N$ 번 독립 반복했을 때 성공 횟수 $m$ 이 나타날 이산 확률분포:

$$p(m \mid N, \mu) = \begin{pmatrix} N \\\\ m \end{pmatrix} \mu^m (1 - \mu)^{N-m} \quad (\text{Eq 6.95})$$

$$\mathbb{E}[m] = N\mu, \quad V[m] = N\mu(1 - \mu) \quad (\text{Eq 6.96~6.97})$$


### 📌 3. 베타 분포 (Beta Distribution: Example 6.10 & Eq 6.98~6.101)

연속적인 확률 파라미터 $\mu \in [0, 1]$ 자체의 불확실성을 모델링하는 분포 (형상 파라미터 $\alpha > 0, \beta > 0$):

$$p(\mu \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1} (1 - \mu)^{\beta-1} \quad (\text{Eq 6.98})$$

$$\mathbb{E}[\mu] = \frac{\alpha}{\alpha + \beta}, \quad V[\mu] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)} \quad (\text{Eq 6.99})$$

- $\Gamma(t) = \int_0^\infty x^{t-1} e^{-x} dx$ 는 감마 함수(Gamma Function).
- $\alpha$ 는 확률 질량을 1 쪽으로, $\beta$ 는 0 쪽으로 이동시킵니다 ($\alpha = \beta = 1$ 일 때 균등분포 $\mathcal{U}[0, 1]$).


## 2. ⚔️ Section 6.6.1: Conjugacy (공액성과 공액 사전분포)


### 📌 1. 공액 사전분포의 정의 (Definition 6.13)

사전분포 $p(\theta)$ 가 우도 함수 $p(y \mid \theta)$ 에 대해 공액 사전분포(Conjugate Prior) 가 된다는 것은, 베이즈 정리로 계산된 사후분포 $p(\theta \mid y)$ 가 사전분포 $p(\theta)$ 와 동일한 확률분포 족(Family)에 속함을 의미합니다!

$$\text{Prior } p(\theta) \in \text{Family } \mathcal{F} \implies \text{Posterior } p(\theta \mid y) \propto p(y \mid \theta) p(\theta) \in \text{Family } \mathcal{F}$$


### 📌 2. Beta-Binomial / Beta-Bernoulli 공액성 증명 (Examples 6.11 & 6.12)

1. Beta-Binomial 공액성 (Eq 6.104):
   $N$ 번 중 $h$ 번 성공 관측 시 사후분포:
   $$p(\mu \mid x = h, N, \alpha, \beta) \propto \mu^h (1 - \mu)^{N-h} \cdot \mu^{\alpha-1} (1 - \mu)^{\beta-1} = \mu^{(h+\alpha)-1} (1 - \mu)^{(N-h+\beta)-1}$$
   $$\implies \mathbf{\text{Beta}(h + \alpha, \; N - h + \beta)}$$
   (사후분포 파라미터가 단순히 사전 파라미터에 관측 횟수를 더하는 덧셈으로 갱신됨!).

2. Beta-Bernoulli 공액성 (Eq 6.105):
   $$p(\theta \mid x, \alpha, \beta) \propto \mathbf{\text{Beta}(\alpha + x, \; \beta + 1 - x)}$$


### 👑 Table 6.2 핵심 공액 사전분포 대조표 (★ 면접 필수!)

| 우도 함수 (Likelihood) | 공액 사전분포 (Conjugate Prior) | 사후분포 (Posterior) |
| :--- | :--- | :--- |
| Bernoulli / Binomial | Beta | Beta |
| Multinomial | Dirichlet | Dirichlet |
| Gaussian (평균 $\mu$) | Gaussian | Gaussian |
| Gaussian (분산 $\sigma^2$) | Inverse Gamma | Inverse Gamma |
| Multivariate Gaussian (공분산 $\Sigma$) | Inverse Wishart | Inverse Wishart |


## 3. ⚔️ Section 6.6.2 & 6.6.3: Sufficient Statistics and Exponential Family (충분통계량과 지수 족)


### 📌 1. 충분통계량과 피셔-네이만 정리 (Theorem 6.14 & Eq 6.106)

충분통계량(Sufficient Statistics $\boldsymbol{\phi}(\mathbf{x})$) 은 데이터 $\mathbf{x}$ 로부터 파라미터 $\theta$ 를 추론하는 데 필요한 모든 정보를 담고 있는 결정론적 통계량 벡터입니다.

#### 💡 피셔-네이만 인수분해 정리 (Fisher-Neyman Factorization Theorem 6.14)
$$\text{PDF } p(\mathbf{x} \mid \theta) = h(\mathbf{x}) g_\theta(\boldsymbol{\phi}(\mathbf{x})) \quad (\text{Eq 6.106})$$
(데이터 $\mathbf{x}$ 의 크기 $N$ 이 아무리 커져도, 파라미터 $\theta$ 에 의존하는 부분은 오직 고정된 유한 차원의 통계량 벡터 $\boldsymbol{\phi}(\mathbf{x})$ 를 통해서만 전달됩니다.)


### 📌 2. 지수 족 (Exponential Family: Eq 6.107~6.108)

파라미터 $\boldsymbol{\theta} \in \mathbb{R}^D$ 에 대한 지수 족 분포의 일반적 수식 구조:

$$p(\mathbf{x} \mid \boldsymbol{\theta}) = h(\mathbf{x}) \exp\left( \langle \boldsymbol{\theta}, \boldsymbol{\phi}(\mathbf{x}) \rangle - A(\boldsymbol{\theta}) \right) \propto \exp\left( \boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x}) \right) \quad (\text{Eq 6.107~6.108})$$

- $\boldsymbol{\theta}$: 자연 파라미터 (Natural Parameters)
- $\boldsymbol{\phi}(\mathbf{x})$: 충분통계량 벡터 (Sufficient Statistics)
- $A(\boldsymbol{\theta})$: 로그 분할 함수 (Log-Partition Function) (적분합을 1로 맞추는 정규화 상수)


### 💡 [Example 6.14: 베르누이 지수 족 변환과 로지스틱 시그모이드 유도]
베르누이 $p(x \mid \mu) = \mu^x (1-\mu)^{1-x}$ 를 지수 족 형태로 전개:
$$p(x \mid \mu) = \exp\left[ x \ln\mu + (1-x) \ln(1-\mu) \right] = \exp\left[ x \ln\frac{\mu}{1-\mu} + \ln(1-\mu) \right] \quad (\text{Eq 6.113d})$$

- 자연 파라미터: $\theta = \ln \frac{\mu}{1-\mu}$ (로짓 / Log-Odds)
- 역관계 유도:
  $$\mu = \frac{1}{1 + e^{-\theta}} = \mathbf{\sigma(\theta)} \quad (\text{Eq 6.118})$$
  (★ 딥러닝 분류의 핵심인 로지스틱 시그모이드(Sigmoid) 함수가 수학적으로 100% 정밀 유도됨!)


### 📌 3. 지수 족의 3대 핵심 장점

1. 유한 차원 충분통계량 보유 (Pitman-Darmois-Koopman 정리): 데이터 수 $N$ 이 무한히 커져도 파라미터 차원이 증가하지 않음.
2. 공액 사전분포의 자동 유도 (Example 6.15): 지수 족 형태(Eq 6.120)로부터 공액 사전분포 수식을 100% 공식으로 도출 가능 (베르누이 ➡️ 베타 유도).
3. 음의 로그 우도의 완벽한 볼록성 (Concavity): $\ln p(\mathbf{x} \mid \boldsymbol{\theta})$ 가 오목 함수(Concave)이므로, 경사하강법 시 국소 최솟값(Local Minima)에 빠지지 않고 100% 전역 최적해(Global Optimum) 보장!


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 공액 사전분포 (Conjugate Prior): 사후분포가 사전분포와 동일한 확률분포 족에 속하게 만드는 사전분포로, 파라미터의 단순 덧셈 갱신을 가능하게 합니다.
- 지수 족 (Exponential Family $p(x|\theta) \propto \exp(\theta^\top \phi(x))$): 자연 파라미터와 충분통계량의 내적 지수 형태로 표현되는 확률분포들의 통합 클래스입니다.
- 충분통계량 ($\phi(x)$): 데이터의 크기 $N$ 과 상관없이 모집단 파라미터 추론에 필요한 모든 정보를 담고 있는 고정 차원의 통계량입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 온라인/스트리밍 데이터 실시간 베이즈 갱신: 새로운 데이터가 하나씩 들어올 때마다 대용량 역행렬이나 적분 없이 공액 사전분포의 파라미터($\alpha + x, \beta + 1 - x$)만 더해서 사후분포를 0초 만에 갱신하기 위해 사용합니다.
- 전역 최적해(Global Optimum) 보장: 로지스틱 회귀 및 일반화 선형 모델(GLM)에서 음의 로그 우도 손실함수가 볼록(Convex)하여 경사하강법으로 최적 파라미터를 확정하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 수식 계산의 편리성 vs 표현력의 한계:
  - 공액 사전분포 / 지수 족: 계산이 완벽한 닫힌 형태(Closed-form)로 떨어지고 매우 빠르지만, 실제 복잡한 복합 데이터 분포를 100% 표현하기에는 제약이 존재함.
  - 비공액 분포: 현실의 정교한 분포를 표현할 수 있지만, 사후분포 적분이 불가능(Intractable)하여 MCMC나 VI 같은 근사 추론이 필수적임.


### 4️⃣ [4단계 실전 AI 연결고리]
- 로지스틱 회귀 (Logistic Regression):
  베르누이 지수 족의 자연 파라미터 역관계로부터 유도된 시그모이드 $\sigma(\mathbf{w}^\top \mathbf{x})$ 를 이진 분류 활성화 함수로 사용.
- 소프트맥스 함수 (Softmax Function & Cross-Entropy):
  다항 분포(Multinomial Distribution)의 지수 족 자연 파라미터로부터 다중 클래스 분류의 Softmax 함수 $P(Y=k|\mathbf{x}) = \frac{\exp(z_k)}{\sum \exp(z_j)}$ 정밀 유도.
- 토픽 모델링 (LDA - Latent Dirichlet Allocation):
  문서 내 단어의 다항 분포(Multinomial)에 대한 공액 사전분포로 디리클레(Dirichlet) 분포를 사용하여 텍스트 잠재 주제 추출.
