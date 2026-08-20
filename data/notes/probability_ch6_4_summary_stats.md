# 📐 6.4 Summary Statistics and Independence (요약 통계량, 공분산 행렬, 아핀 변환과 통계적 독립성)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 6.4 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 통계량의 세계: 왜 "요약 통계량과 독립성"을 배우는가?

우리는 확률변수의 전체 분포($p(x)$)를 직접 가지고 있어도, 그 분포의 특성을 대표하는 몇 개의 숫자(통계량, Statistics)로 요약하고 변수 간의 관계를 분석해야 합니다.

- 기댓값과 평균 ($\mathbb{E}[X], \boldsymbol{\mu}$): 분포의 중심 위치를 나타내는 1차 모멘트.
- 분산과 공분산 행렬 ($V[X], \Sigma$): 데이터가 중심으로부터 얼마나 넓게 퍼져 있는지(퍼짐성, Spread)와 변수 간의 선형적 상호관계를 나타내는 대칭 양의 준정정 행렬.
- 아핀 변환 ($\mathbf{y} = A\mathbf{x} + \mathbf{b}$): 선형 레이어를 통과한 변환된 데이터의 평균($A\boldsymbol{\mu} + \mathbf{b}$)과 공분산($A\Sigma A^\top$)을 유도하는 딥러닝의 기초 수학.
- 통계적 독립성 ($X \perp\!\!\!\perp Y$)과 조건부 독립성: $Y$ 를 알아도 $X$ 에 대한 추가 정보가 없는 상태를 수식화하여 그래프 모델(Graphical Models)과 나이브 베이즈의 근간을 형성.


## 1. ⚔️ Section 6.4.1 & 6.4.2: Means, Covariances, and Empirical Statistics (기댓값, 평균, 공분산과 표본 통계량)


### 📌 1. 기댓값과 평균의 정의 (Definitions 6.3 ~ 6.4 & Eq 6.28~6.32)

확률변수 $X \sim p(x)$ 의 함수 $g(x)$ 에 대한 기댓값 $\mathbb{E}_X[g(x)]$ 는 다음과 같이 정의됩니다:
- 연속: $\mathbb{E}_X[g(x)] = \int_{\mathcal{X}} g(x) p(x) dx \quad (\text{Eq 6.28})$
- 이산: $\mathbb{E}_X[g(x)] = \sum_{x \in \mathcal{X}} g(x) p(x) \quad (\text{Eq 6.29})$

#### 💡 기댓값의 선형성 (Linearity of Expectation: Eq 6.34)
상수 $a, b$ 와 함수 $g(x), h(x)$ 에 대해 기댓값 연산자는 완벽한 선형 연산자입니다:

$$\mathbb{E}[a g(x) + b h(x)] = a \mathbb{E}[g(x)] + b \mathbb{E}[h(x)] \quad (\text{Eq 6.34d})$$

#### 💡 평균(Mean), 중앙값(Median), 최빈값(Mode) 비교 (Figure 6.4 & Example 6.4)
- 평균 (Mean $\boldsymbol{\mu} = \mathbb{E}[\mathbf{x}]$): 모든 데이터의 산술 평균. 이상치(Outlier)에 민감함.
- 중앙값 (Median): CDF가 0.5인 위치. 이상치에 매우 강건(Robust)하나 고차원 공간에서 순서 정렬이 불가능함.
- 최빈값 (Mode): PDF의 최고 피크점. 다봉 분포(Bimodal)에서 여러 개 존재 가능.


### 📌 2. 공분산과 공분산 행렬 (Covariance Matrix: Definitions 6.5 ~ 6.7 & Eq 6.35~6.38)

두 단변량 확률변수 $X, Y$ 의 공분산은 평균으로부터의 편차의 곱의 기댓값입니다:

$$\text{Cov}[x, y] = \mathbb{E}[(x - \mathbb{E}[x])(y - \mathbb{E}[y])] = \mathbb{E}[xy] - \mathbb{E}[x]\mathbb{E}[y] \quad (\text{Eq 6.36})$$

#### 👑 다변수 공분산 행렬 (Covariance Matrix: Definition 6.7 & Eq 6.38)
다변량 확률변수 $\mathbf{x} \in \mathbb{R}^D$ (평균 $\boldsymbol{\mu}$) 에 대한 분산은 $D \times D$ 대칭 행렬인 공분산 행렬(Covariance Matrix $\Sigma$) 이 됩니다:

$$\Sigma = V_X[\mathbf{x}] = \text{Cov}[\mathbf{x}, \mathbf{x}] := \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top] = \mathbb{E}[\mathbf{x}\mathbf{x}^\top] - \boldsymbol{\mu}\boldsymbol{\mu}^\top \quad (\text{Eq 6.38b})$$

$$\Sigma = \begin{bmatrix} 
\text{Cov}[x_1, x_1] & \text{Cov}[x_1, x_2] & \dots & \text{Cov}[x_1, x_D] \\\\
\text{Cov}[x_2, x_1] & \text{Cov}[x_2, x_2] & \dots & \text{Cov}[x_2, x_D] \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\text{Cov}[x_D, x_1] & \text{Cov}[x_D, x_2] & \dots & \text{Cov}[x_D, x_D]
\end{bmatrix} \in \mathbb{R}^{D \times D} \quad (\text{Eq 6.38c})$$

- 핵심 성질: 주대각선 성분은 각 변수의 분산($V[x_i]$), 비대각선 성분은 교차 공분산($\text{Cov}[x_i, x_j]$)이며, 항상 대칭 행렬($\Sigma = \Sigma^\top$)이자 양의 준정정 행렬($\Sigma \succeq 0$) 입니다.


### 📌 3. 상관계수 (Correlation: Definition 6.8 & Eq 6.40)

공분산을 각 변수의 표준편차로 나눈 표준화된 지표입니다:

$$\text{corr}[x, y] = \frac{\text{Cov}[x, y]}{\sqrt{V[x]V[y]}} = \frac{\text{Cov}[x, y]}{\sigma[x]\sigma[y]} \in [-1, 1] \quad (\text{Eq 6.40})$$


### 📌 4. 표본 평균과 표본 공분산 (Empirical Statistics: Definition 6.9 & Eq 6.41~6.42)

$N$ 개의 관측 데이터 $\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^D$ 로부터 계산하는 표본 통계량:
- 표본 평균: $\overline{\mathbf{x}} = \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \quad (\text{Eq 6.41})$
- 표본 공분산 행렬: $\Sigma = \frac{1}{N} \sum_{n=1}^N (\mathbf{x}_n - \overline{\mathbf{x}})(\mathbf{x}_n - \overline{\mathbf{x}})^\top \quad (\text{Eq 6.42})$


## 2. 3가지 분산 표현식과 아핀 변환 (Section 6.4.3 & 6.4.4)


### 📌 1. 분산의 3가지 수학적 표현식 (Eq 6.43~6.45)

1. 표준 정의 (Squared Deviation: Eq 6.43): $V[x] = \mathbb{E}[(x - \mu)^2]$ (평균 계산 후 재순회 필요 ➡️ 2-pass).
2. 계산용 공식 (Raw-Score Formula: Eq 6.44):
   $$V[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$$
   ("제곱의 평균 마이너스 평균의 제곱". 단 1번의 데이터 순회(1-pass)로 계산 가능하나 부동소수점 정밀도 손실 주의).
3. 쌍간 차이 합 공식 (Pairwise Differences: Eq 6.45):
   $$\frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N (x_i - x_j)^2 = 2 \left[ \frac{1}{N} \sum_{i=1}^N x_i^2 - \left( \frac{1}{N} \sum_{i=1}^N x_i \right)^2 \right]$$
   ($N^2$ 개 데이터 쌍간 거리의 합이 중심 평균으로부터의 거리 합의 2배와 완벽히 일치함을 증명하는 기하학적 공식).


### 📌 2. 확률변수의 아핀 변환 통계량 (Affine Transformations: Eq 6.50~6.52 - ★ 딥러닝 필수!)

확률변수 $\mathbf{x}$ (평균 $\boldsymbol{\mu}$, 공분산 $\Sigma$) 에 대해 아핀 변환 $\mathbf{y} = A\mathbf{x} + \mathbf{b}$ 를 적용했을 때 변환된 확률변수 $\mathbf{y}$ 의 통계량:

1. 변환된 평균:
   $$\mathbb{E}[\mathbf{y}] = \mathbb{E}[A\mathbf{x} + \mathbf{b}] = A \mathbb{E}[\mathbf{x}] + \mathbf{b} = A\boldsymbol{\mu} + \mathbf{b} \quad (\text{Eq 6.50})$$
2. 변환된 공분산 행렬:
   $$V[\mathbf{y}] = V[A\mathbf{x} + \mathbf{b}] = V[A\mathbf{x}] = A V[\mathbf{x}] A^\top = A \Sigma A^\top \quad (\text{Eq 6.51})$$
3. 교차 공분산:
   $$\text{Cov}[\mathbf{x}, \mathbf{y}] = \Sigma A^\top \quad (\text{Eq 6.52d})$$


## 3. ⚔️ Section 6.4.5 & 6.4.6: Independence and Inner Products (독립성과 확률변수의 기하학)


### 📌 1. 통계적 독립성 (Statistical Independence: Definition 6.10 & Eq 6.53)

두 확률변수 $X, Y$ 가 통계적으로 독립일 필요충분조건은 결합 확률분포가 각 확률분포의 곱으로 분해되는 것입니다:

$$p(x, y) = p(x) p(y) \iff p(y \mid x) = p(y) \iff p(x \mid y) = p(x) \quad (\text{Eq 6.53})$$

#### 💡 독립성의 성질과 ★ 함정 (Example 6.5)
- $X, Y$ 가 독립이면 $\text{Cov}[x, y] = 0$ 이고 $V[x + y] = V[x] + V[y]$ 가 성립합니다.
- ★ 치명적 주의점 (역은 성립하지 않음!): 공분산이 0이라고 해서 두 변수가 통계적으로 독립인 것은 아닙니다 ($\text{Cov}[x, y] = 0 \centernot\implies X \perp\!\!\!\perp Y$).
  - 이유: 공분산은 오직 "선형 관계"만 측정하기 때문입니다!
  - *Example 6.5*: $X$ 가 평균 0이고 $\mathbb{E}[X^3] = 0$ 인 분포일 때, $Y = X^2$ (100% 종속 관계) 로 두면 $\text{Cov}[X, Y] = \mathbb{E}[X^3] - \mathbb{E}[X]\mathbb{E}[X^2] = 0$ 이 됩니다.


### 📌 2. 조건부 독립성 (Conditional Independence: Definition 6.11 & Eq 6.55~6.57)

관측 변수 $Z$ 가 주어졌을 때 $X$ 와 $Y$ 가 조건부 독립일 필요충분조건 ($X \perp\!\!\!\perp Y \mid Z$):

$$p(x, y \mid z) = p(x \mid z) p(y \mid z) \iff p(x \mid y, z) = p(x \mid z) \quad (\text{Eq 6.55, 6.57})$$

- 직관적 의미: "$Z$ 의 정보를 알고 나면, $Y$ 에 대한 추가 지식은 $X$ 의 확률을 업데이트하는 데 아무런 도움이 되지 않는다!" (그래피컬 모델의 핵심).


### 📌 3. 확률변수의 내적과 기하학 (Inner Products: Figure 6.6 & Eq 6.59~6.61)

평균이 0인 확률변수 공간에서 공분산을 내적(Inner Product) 으로 정의합니다:

$$\langle X, Y \rangle := \text{Cov}[x, y] \quad (\text{Eq 6.59})$$

- 확률변수의 노름 (길이): $\Vert X \Vert = \sqrt{\langle X, X \rangle} = \sqrt{V[x]} = \sigma[x]$ (표준편차 = 변수의 길이! Eq 6.60).
- 사이각과 상관계수: $\cos \theta = \frac{\langle X, Y \rangle}{\Vert X \Vert \Vert Y \Vert} = \frac{\text{Cov}[x, y]}{\sigma[x]\sigma[y]} = \text{corr}[x, y]$ (Eq 6.61).
- 직교성 ($X \perp Y \iff \text{Cov}[x, y] = 0$): 상관관계가 0(무상관)이면 두 확률변수 벡터는 기하학적으로 직교(Orthogonal)하며, 피타고라스 정리 $V[x + y] = V[x] + V[y]$ 가 정확히 성립합니다 (Figure 6.6).


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 공분산 행렬 ($\Sigma = \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^\top]$): 다변량 확률변수의 각 차원별 퍼짐성과 교차 선형 상관관계를 나타내는 대칭 양의 준정정 행렬입니다.
- 아핀 변환 통계량 ($\mathbb{E}[A\mathbf{x}+\mathbf{b}] = A\boldsymbol{\mu}+\mathbf{b}, V[A\mathbf{x}+\mathbf{b}] = A\Sigma A^\top$): 선형 변환 후 데이터의 평균과 공분산 변화를 나타내는 기본 정리입니다.
- 확률변수의 내적 ($\langle X, Y \rangle = \text{Cov}[x, y]$): 공분산을 내적으로, 표준편차를 길이로, 상관계수를 코사인 사이각으로 해석하는 정교한 기하학적 체계입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 선형 레이어 패스 후 데이터 정규화: 신경망의 선형 레이어($\mathbf{y} = W\mathbf{x} + \mathbf{b}$)를 통과할 때 데이터의 평균과 공분산이 어떻게 변하는지 추적하고 배치 정규화(Batch Normalization)를 수행하기 위해 사용합니다.
- 차원 축소와 정보 손실 최소화: 공분산 행렬 $\Sigma$ 의 고유값 분해를 통해 가장 분산이 큰 축(주성분)을 찾아 다차원 데이터를 압축(PCA)하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 무상관(Uncorrelated) vs 통계적 독립(Independent):
  - 무상관 ($\text{Cov}[x, y] = 0$): 오직 1차 선형 관계만 없음. 비선형 종속 관계($y = x^2$)는 포착하지 못함.
  - 통계적 독립 ($p(x, y) = p(x)p(y)$): 선형/비선형을 포함한 모든 관계가 완전히 부재함 (무상관보다 훨씬 강한 조건).


### 4️⃣ [4단계 실전 AI 연결고리]
- 배치 정규화 (Batch Normalization - BN):
  미니배치 표본 평균 $\overline{\mathbf{x}}$ 과 표본 분산 $\sigma^2$ 을 이용해 $\hat{\mathbf{x}} = \frac{\mathbf{x} - \overline{\mathbf{x}}}{\sqrt{\sigma^2 + \epsilon}}$ 로 정규화하여 Internal Covariate Shift 현상을 방지.
- 주성분 분석 (PCA - Principal Component Analysis - Ch 10):
  표본 공분산 행렬 $\Sigma = \frac{1}{N} X^\top X$ 의 고유벡터를 찾아 분산 $V[y] = \mathbf{w}^\top \Sigma \mathbf{w}$ 를 최대화하는 직교 주축을 추출.
- 가우시안 프로세스 (Gaussian Process - GP):
  입력 간의 커널 행렬 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ 를 공분산 행렬로 사용하여 무한 차원 함수 공간의 사후분포를 유도.
