# 📐 6.3 Sum Rule, Product Rule, and Bayes’ Theorem (확률 4대 요소, 합/곱의 법칙과 베이즈 정리)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 6.3 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 확률 추론의 2대 기둥: 왜 "합의 법칙, 곱의 법칙과 베이즈 정리"인가?

머신러닝과 머신 러닝 추론(Inference)의 모든 복잡한 확률 모델링(Probabilistic Modeling)은 놀랍게도 단 2가지 근본 법칙(합의 법칙 & 곱의 법칙)에서 출발합니다.

- 합의 법칙 (Sum Rule / Marginalization): 관심 없는 수많은 변수를 더하거나 적분하여 제거함(주변화)으로써 원하는 변수만의 주변 확률분포를 구합니다.
- 곱의 법칙 (Product Rule / Factorization): 복잡한 결합 확률분포를 "조건부 확률 $\times$ 사전 확률"의 차곡차곡 쌓이는 곱의 형태로 분해(인수분해)합니다.
- 베이즈 정리 (Bayes' Theorem): 곱의 법칙으로부터 유도되며, 관측 데이터 $y$ 가 들어왔을 때 우리가 가졌던 사전 신념(Prior)을 어떻게 사후 신념(Posterior)으로 업데이트할지 명확히 제시하는 "확률적 역연산(Probabilistic Inverse)" 기법입니다.


## 1. ⚔️ Section 6.3: 확률의 2대 근본 법칙 (Sum Rule & Product Rule)


### 📌 1. 제1법칙: 합의 법칙 (Sum Rule / Marginalization: Eq 6.20~6.21)

결합 확률분포 $p(x, y)$ 에서 변수 $y$ 를 소거(적분/합산)하여 주변 확률분포 $p(x)$ 를 얻는 법칙입니다:

$$p(x) = \begin{cases} \sum_{y \in \mathcal{Y}} p(x, y) & (\text{이산 확률변수 } y) \\\\ \int_{\mathcal{Y}} p(x, y) dy & (\text{연속 확률변수 } y) \end{cases} \quad (\text{Eq 6.20})$$

- 다변량 확장 (Eq 6.21): $\mathbf{x} = [x_1, \dots, x_D]^\top$ 에서 $x_i$ 하나만 남기고 나머지 모든 변수 $\mathbf{x}_{\setminus i}$ 를 소거:
  $$p(x_i) = \int_{\mathcal{X}_{\setminus i}} p(x_1, \dots, x_D) d\mathbf{x}_{\setminus i} \quad (\text{Eq 6.21})$$

#### ⚠️ probabilistic modeling의 치명적 계산 장벽
합의 법칙을 적용할 때 변수가 많거나 연속 공간일 경우, 고차원 적분/합산 $\int \dots \int p(\mathbf{x}) d\mathbf{x}$ 은 다항 시간(Polynomial Time) 내에 계산하는 알고리즘이 존재하지 않는 극악의 연산 폭발(NP-Hard급)을 일으킵니다! (이를 해결하기 위해 MCMC, 변분 추론 VI 등이 등장함).


### 📌 2. 제2법칙: 곱의 법칙 (Product Rule: Eq 6.22)

두 확률변수의 결합 확률분포 $p(x, y)$ 는 조건부 확률과 주변 확률의 곱으로 완벽히 인수분해(Factorization)됩니다:

$$p(x, y) = p(y \mid x) p(x) = p(x \mid y) p(y) \quad (\text{Eq 6.22})$$


## 2. ⚔️ Bayes' Theorem (베이즈 정리와 4대 구성요소: Eq 6.23~6.27)


### 📌 1. 베이즈 정리 유도 (Eq 6.24~6.26)

곱의 법칙의 두 표현 $p(x, y) = p(x \mid y)p(y)$ 와 $p(x, y) = p(y \mid x)p(x)$ 가 같다고 놓으면 단 1초 만에 베이즈 정리가 유도됩니다:

$$p(x \mid y) p(y) = p(y \mid x) p(x) \iff p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)} \quad (\text{Eq 6.26})$$


### 📌 2. 베이즈 정리 4대 요체 완전 해체 (Eq 6.23)

$$\underbrace{p(x \mid y)}_{\text{사후분포 (Posterior)}} = \frac{\overbrace{p(y \mid x)}^{\text{우도 (Likelihood)}} \cdot \overbrace{p(x)}^{\text{사전분포 (Prior)}}}{\underbrace{p(y)}_{\text{증거 / 주변 우도 (Evidence)}}}$$

1. 사후분포 (Posterior $p(x \mid y)$):
   - 데이터 $y$ 를 관측한 후, 우리가 알고 싶어 하는 가설/파라미터 $x$ 에 대해 최종적으로 업데이트된 확률분포입니다 (Bayesian의 최종 목표!).
2. 사전분포 (Prior $p(x)$):
   - 데이터를 보기 전에 우리가 $x$ 에 대해 가지고 있던 주관적/수학적 사전 지식입니다. (아무리 희귀한 값이라도 $p(x) = 0$ 이 되지 않도록 주의해야 합니다).
3. 우도 (Likelihood $p(y \mid x)$ / Measurement Model):
   - 파라미터 $x$ 가 정해졌을 때, 관측 데이터 $y$ 가 나타날 확률/밀도입니다 ($x$ 에 대한 함수로 해석).
4. 증거 / 주변 우도 (Evidence / Marginal Likelihood $p(y)$: Eq 6.27):
   $$p(y) = \int_{\mathcal{X}} p(y \mid x) p(x) dx = \mathbb{E}_X[p(y \mid x)] \quad (\text{Eq 6.27})$$
   - 모든 가능한 $x$ 에 대해 분자값(우도 $\times$ 사전분포)을 적분한 값으로, 사후분포의 전체 적분 합을 1로 맞춰주는 정규화 상수(Normalization Constant)이자 베이지안 모델 선택(Model Selection)의 기준입니다.


### 📌 3. 확률적 역행렬 (Probabilistic Inverse)로서의 의미

원인 $x$ 로부터 결과 데이터 $y$ 가 유도되는 순방향 우도 $p(y \mid x)$ 가 주어졌을 때, 반대로 관측된 결과 $y$ 로부터 원인 $x$ 의 확률 $p(x \mid y)$ 를 거꾸로 뒤집어 추정해주므로 베이즈 정리를 "확률적 역연산(Probabilistic Inverse)" 이라 부릅니다.


### 💡 [Remark: 전체 사후분포(Full Posterior) vs 점추정(MAP/MLE)의 치명적 성능 차이]
- 점추정(MAP/MLE): 계산을 단순화하기 위해 사후분포의 피크점 하나만 선택 ➡️ 불확실성을 무시하여 오버피팅 발생.
- 전체 사후분포 (Full Posterior) 유지: 모델 기반 강화학습(RL, Deisenroth et al., 2015)에서 모델의 모든 불확실성(Full Posterior)을 고려하면 데이터 효율성이 극대화되어 초고속 학습에 성공하지만, 피크점(MAP)만 추정하면 불확실성을 무시하여 일관되게 학습에 실패(Consistent Failures)함이 증명되었습니다!


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 합의 법칙 (Sum Rule): $p(x) = \int p(x, y) dy$. 결합분포에서 원하지 않는 변수를 적분/합산 소거하여 주변분포를 얻는 연산입니다.
- 곱의 법칙 (Product Rule): $p(x, y) = p(y \mid x)p(x)$. 결합분포를 조건부 분포와 사전분포의 곱으로 인수분해하는 법칙입니다.
- 베이즈 정리: 사후분포 $p(x \mid y) = \frac{p(y \mid x)p(x)}{p(y)}$ 로 관측 데이터를 통해 사전 신념을 사후 신념으로 업데이트하는 확률적 역연산 체계입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 관측 데이터로부터 역방향 추론: 입력 데이터 $y$ 만 관측할 수 있는 실제 머신러닝 환경에서, 가려진 숨은 파라미터/클래스 $x$ 를 확률적으로 정확히 추정하기 위해 베이즈 정리를 사용합니다.
- 데이터 노이즈와 사전 지식의 결합: 적은 데이터 환경에서 과적합을 방지하기 위해 정규화(L2 Regularization) 역할을 하는 사전분포(Prior)를 수식적으로 결합하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 점추정 (MLE / MAP) vs 베이지안 전체 사후분포 (Full Posterior):
  - MLE / MAP: $x$ 의 대표값 하나만 뽑으므로 연산 속도가 압도적으로 빠르지만, 모델의 불확실성을 표현하지 못해 데이터가 적을 때 심각하게 왜곡됩니다.
  - Full Posterior: $p(x \mid y)$ 전체 분포를 보존하므로 불확실성에 극도로 로버스트하지만, 분모 $p(y) = \int p(y \mid x)p(x)dx$ 고차원 적분 계산이 극도로 난해합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 나이브 베이즈 분류기 (Naive Bayes Classifier):
  특징 벡터 $\mathbf{x} = [x_1, \dots, x_D]$ 의 조건부 독립성($p(\mathbf{x} \mid Y) = \prod p(x_i \mid Y)$)을 가정하여 텍스트 분류 및 스팸 필터링 수행.
- 변분 자가부호화기 (VAE - Variational Autoencoder - Ch 11):
  잠재 변수 $\mathbf{z}$ 의 사후분포 $p(\mathbf{z} \mid \mathbf{x})$ 적분이 불가능(Intractable)하므로, 변분 분포 $q_\phi(\mathbf{z} \mid \mathbf{x})$ 로 근사하여 ELBO(Evidence Lower Bound)를 극대화.
- 베이지안 최적화 (Bayesian Optimization):
  하이퍼파라미터 튜닝 시 서로게이트 모델(가우시안 프로세스)의 사후분포 수집으로 획득 함수(Acquisition Function)를 계산하여 최소 시도로 최적 하이퍼파라미터 검색.
