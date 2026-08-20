# 📐 6.0 & 6.1 Construction of a Probability Space (확률과 확률분포의 서막, 확률공간과 확률변수의 수학적 구조)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 6 도입부 & Section 6.1 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. Chapter 6의 서막: 왜 "확률과 확률분포(Probability & Distributions)"인가?

우리는 지난 Chapter 2~5를 통해 선형대수학(공간과 변환)과 벡터 미적분학(최적화 기울기)을 마스터했습니다.
하지만 현실 세계의 데이터, 인공지능 모델의 가중치, 그리고 모델의 예측에는 언제나 불확실성(Uncertainty)이 존재합니다.

- 데이터의 불확실성 (Data Uncertainty / Aleatoric): 센서 노이즈, 측정 오차, 데이터 자체의 고유한 확률적 변동성.
- 모델의 불확실성 (Model Uncertainty / Epistemic): 데이터 부족으로 인해 모델 파라미터에 대해 갖는 불확실성 (학습 데이터가 많아지면 줄어듦).
- 예측의 불확실성 (Prediction Uncertainty): 미래의 새로운 입력에 대해 모델이 자신의 예측을 얼마나 확신하는지에 대한 신뢰도.

확률론(Probability Theory)은 불확실성을 추상적인 감이 아니라 엄밀한 실수 범위([0, 1])의 수학적 구조로 정량화하여 자동화된 추론(Automated Reasoning)을 수행하게 해주는 핵심 도구입니다.


## 1. ⚔️ Section 6.1.1: Philosophical Issues & Interpretations (확률의 철학적 관점과 제인스 정리)


### 📌 1. 불리언 논리의 한계와 콕스-제인스 정리 (Cox-Jaynes Theorem)

고전 불리언 논리(Boolean Logic)는 오직 참(True, 1)과 거짓(False, 0)만 다루므로, 개연성(Plausibility)의 정도를 표현하지 못합니다.
- 예시: 약속 시간에 지각한 친구의 3가지 가설 (H1: 제시간 도착, H2: 교통체증, H3: 외계인 납치).
  친구의 지각이 관측되면 H1은 논리적으로 즉시 배제(False)됩니다. 불리언 논리는 H2와 H3에 대해 아무런 우열을 판단하지 못하지만, 우리는 상식적으로 H2가 H3보다 훨씬 더 가능성이 높다고 판단합니다.

#### 💡 콕스-제인스 정리 (Cox-Jaynes Theorem)
E. T. 제인스(E. T. Jaynes)는 인간의 상식적 추론이 가져야 할 3가지 수학적 기준을 제시했습니다:
1. 개연성의 정도는 실수(Real Numbers)로 표현되어야 한다.
2. 이 실수는 인간의 상식 법칙과 부합해야 한다.
3. 추론 결과는 일관성(Consistency), 정직성(Honesty), 재현성(Reproducibility)을 만족해야 한다.

이 공리들을 수학적으로 정리하면 개연성을 다루는 일관된 규칙이 다름 아닌 "확률의 법칙(Rules of Probability)"과 100% 완벽히 일치함이 증명됩니다.


### 📌 2. 빈도주의(Frequentist) vs 베이지안(Bayesian) 관점 대조 (★ 면접 필수!)

| 비교 항목 | 빈도주의자 관점 (Frequentist Interpretation) | 베이지안 관점 (Bayesian Interpretation) |
| :--- | :--- | :--- |
| 확률의 정의 | 무한히 반복되는 실험에서 사건이 일어나는 상대적 빈도(Relative Frequency)의 극한 | 사건에 대한 사용자의 불확실성 및 신념의 정도 (Degree of Belief) |
| 파라미터 ($\boldsymbol{\theta}$) | 세상에 단 하나 존재하는 고정된 참값 (Fixed Constant) | 고정값이 아닌 확률변수 (Random Variable) 로 다룸 |
| 주요 접근법 | 최대우도추정(MLE), p-value, 신뢰구간 | 사전분포(Prior), 사후분포(Posterior), MAP, 베이즈 정리 |
| 적용 한계 | 대선 결과나 동전 1번 던지기처럼 반복 불가능한 사건에 적용 곤란 | 데이터가 무한히 많아지면 빈도주의 결과로 수렴 |


## 2. ⚔️ Section 6.1.2: Probability Space & Random Variables (확률공간과 확률변수)


### 📌 1. 콜모고로프 확률공간의 3요소 $(\Omega, \mathcal{A}, P)$

현대 확률론은 앤드레이 콜모고로프(Kolmogorov)의 공리계에 기초하여 확률공간(Probability Space)을 3요소의 튜플 $(\Omega, \mathcal{A}, P)$ 로 정의합니다:

1. 표본 공간 ($\Omega$, Sample Space):
   실험에서 발생할 수 있는 모든 가능한 결과(Outcome $\omega$)들의 전체 집합입니다.
   - 예: 동전 2번 연속 던지기 $\Omega = \{hh, tt, ht, th\}$.
2. 사건 공간 ($\mathcal{A}$, Event Space):
   관측 가능한 결과들의 모임인 사건(Event $A \subseteq \Omega$)들의 집합입니다 ($\sigma$-대수 구조).
3. 확률 측도 ($P$, Probability Measure):
   각 사건 $A \in \mathcal{A}$ 에 대해 확률값 $P(A) \in [0, 1]$ 을 할당하는 함수입니다 ($P(\Omega) = 1$).


### 📌 2. 확률변수 (Random Variable)의 엄밀한 정의

확률변수(Random Variable $X$)는 이름과 달리 '랜덤'한 값도 아니고 '변수'도 아닌, 표본공간의 결과 $\omega \in \Omega$ 를 우리가 관심 있는 숫자/상태 공간인 타겟 공간(Target Space $T$)으로 매핑해 주는 "함수(Mapping Function / Lookup Table)"입니다!

$$X : \Omega \to T \quad (\text{Outcome } \omega \in \Omega \mapsto \text{State } x \in T)$$

- 예시: 동전 2번 던지기에서 앞면($h$)이 나온 횟수를 재는 확률변수 $X$:
  $$X(hh) = 2, \quad X(ht) = 1, \quad X(th) = 1, \quad X(tt) = 0 \implies T = \{0, 1, 2\}$$


### 📌 3. 원상(Pre-image)과 확률분포 법($P_X = P \circ X^{-1}$)

타겟 공간의 부분집합 $S \subseteq T$ 에 대해, $X$ 에 의해 $S$ 로 매핑되는 표본공간 $\Omega$ 의 원소들의 모임을 원상(Pre-image) 이라 부릅니다:

$$X^{-1}(S) := \{\omega \in \Omega : X(\omega) \in S\}$$

타겟 공간에서의 확률 $P_X(S)$ 는 표본공간에서의 원상의 확률과 같습니다 (Eq 6.8):

$$P_X(S) = P(X \in S) = P(X^{-1}(S)) = P(\{\omega \in \Omega : X(\omega) \in S\})$$

- 확률분포 (Law or Distribution of $X$): 이 매핑 함수 $P_X = P \circ X^{-1}$ 를 확률변수 $X$ 의 확률분포(Distribution)라 부릅니다.


### 💡 [Example 6.1: 미국 $, 영국 £ 동전 복원 추출 수치 전수 분석]
가방 안에 미국 동전($\$, P(\$) = 0.3$)과 영국 동전($\pounds, P(\pounds) = 0.7$)이 들어있고, 복원 추출로 2번 뽑는 실험:
- 표본 공간: $\Omega = \{(\$, \$), (\$, \pounds), (\pounds, \$), (\pounds, \pounds)\}$
- 확률변수 $X$: $\$$ 가 나온 총 횟수 $\implies T = \{0, 1, 2\}$
- 매핑 테이블: $X((\$, \$)) = 2, \; X((\$, \pounds)) = 1, \; X((\pounds, \$)) = 1, \; X((\pounds, \pounds)) = 0$
- 확률질량함수 계산 (Eq 6.5~6.7):
  $$P(X = 2) = P((\$, \$)) = 0.3 \times 0.3 = 0.09$$
  $$P(X = 1) = P((\$, \pounds) \cup (\pounds, \$)) = 0.3 \times 0.7 + 0.7 \times 0.3 = 0.42$$
  $$P(X = 0) = P((\pounds, \pounds)) = 0.7 \times 0.7 = 0.49$$
  (모든 확률의 합: $0.09 + 0.42 + 0.49 = 1.0$)


## 3. ⚔️ Section 6.1.3: Probability vs Statistics (확률론과 통계학의 대조)

- 확률론 (Probability): 알려진 모델과 규칙(가우시안 분포, 동전 확률 0.3)으로부터 "미래에 어떤 데이터가 나올 것인가?"를 정방향(Forward)으로 추론하는 학문.
- 통계학 및 머신러닝 (Statistics & Machine Learning): 이미 관측된 데이터가 주어졌을 때 "이 데이터를 만들어낸 기저의 시스템과 최적 파라미터가 무엇인가?"를 역방향(Inverse)으로 추정하는 학문.
- 일반화 오차 (Generalization Error): 머신러닝은 단순한 통계적 적합을 넘어, 아직 보지 못한 미래 데이터에 대한 성과(일반화 능력)를 확률통계적으로 보장하는 것을 목표로 합니다.


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 확률공간 $(\Omega, \mathcal{A}, P)$: 표본공간 $\Omega$, 사건공간 $\mathcal{A}$, 확률측도 $P$ 로 무장하여 불확실성을 실수값으로 정량화하는 수학적 체계입니다.
- 확률변수 ($X: \Omega \to T$): 추상적인 표본공간의 결과 $\omega$ 를 관심 있는 수치/상태 공간 $T$ 로 매핑하는 함수입니다.
- 콕스-제인스 정리: 개연성의 일관된 추론 법칙이 확률의 연산 법칙과 100% 일치함을 증명한 정리입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 불확실성의 체계적 다루기: 딥러닝 모델의 예측 신뢰도와 데이터 노이즈를 수학적으로 정량화하여 과적합을 방지하고 최적의 추론을 내리기 위해 사용합니다.
- 데이터 공간으로의 변환: 추상적인 현실 사건을 컴퓨터가 계산 가능한 숫자 공간($T = \mathbb{R}^D$)으로 변환하기 위해 확률변수를 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 빈도주의(Frequentist) vs 베이지안(Bayesian):
  - 빈도주의: 데이터가 풍부할 때 계산이 명확하고 MLE로 손쉽게 최적 파라미터를 찾지만, 데이터가 적으면 과적합(Overfitting)에 취약합니다.
  - 베이지안: 데이터가 부족할 때 사전 지식(Prior)을 결합하여 안정적인 사후분포(Posterior)를 얻지만, 분모의 정규화 상수 적분 $\int p(\mathbf{x}|\boldsymbol{\theta})p(\boldsymbol{\theta})d\boldsymbol{\theta}$ 계산이 난해하여 MCMC나 VI(변분 추론)가 필요합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 최대우도추정 (MLE - Maximum Likelihood Estimation - Ch 9, 11):
  관측 데이터 $D$ 가 주어졌을 때 데이터가 나타날 확률(우도 $P(D|\boldsymbol{\theta})$)을 극대화하는 파라미터 $\boldsymbol{\theta}_{\text{MLE}}$ 를 찾는 통계적 추정법.
- 최대 사후확률 추정 (MAP - Maximum A Posteriori):
  베이지안 관점에서 사전분포 $p(\boldsymbol{\theta})$ (L2 규제/가우시안 프라이어)를 결합하여 $\boldsymbol{\theta}_{\text{MAP}} = \text{argmax} [p(D|\boldsymbol{\theta})p(\boldsymbol{\theta})]$ 를 추정.
- 분류 모델의 Softmax 확률 출력:
  분류 신경망의 최종 레이어 출력을 $P(Y=k|\mathbf{x}) = \frac{\exp(z_k)}{\sum \exp(z_j)}$ 로 변환하여 입력 $\mathbf{x}$ 가 각 클래스에 속할 개연성을 확률로 정량화.
