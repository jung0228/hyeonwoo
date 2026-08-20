# 📐 6.2 Discrete and Continuous Probabilities (이산 및 연속 확률분포, PMF, PDF, CDF와 주변화/조건부 확률)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 6.2 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 확률 분포의 이분법: 왜 "이산(Discrete)과 연속(Continuous)"을 구분하는가?

확률변수 $X$ 가 취할 수 있는 타겟 공간(Target Space $T$)의 성격에 따라 확률을 정의하고 다루는 수학적 연산이 완전히 달라집니다.

- 이산 확률변수 (Discrete Random Variable): 동전 던지기, 동영상 카테고리, 텍스트 단어처럼 상태가 뚝뚝 떨어져 있는 경우입니다. 각 상태에서의 확률질량함수(PMF) $P(X = x)$ 를 다루며, 전체 확률의 합은 시그마 합산($\sum = 1$)으로 계산합니다.
- 연속 확률변수 (Continuous Random Variable): 키, 온도, 신경망의 가중치, 고화질 이미지 픽셀처럼 상태가 연속적인 구간($\mathbb{R}^D$)인 경우입니다. 특정 한 점의 확률은 $0$이 되므로, 확률밀도함수(PDF) $f(x)$ 와 누적분포함수(CDF) $F(x)$ 를 다루며, 전체 확률의 합은 적분($\int = 1$)으로 계산합니다.


## 1. ⚔️ Section 6.2.1: Discrete Probabilities (이산 확률분포와 결합/주변/조건부 확률)


### 📌 1. 확률질량함수 (PMF: Probability Mass Function)

타겟 공간 $T$ 가 유한하거나 셀 수 있는(Countable) 이산 확률변수 $X$ 에 대해, 특정 상태 $x$ 가 발생할 확률을 확률질량함수(PMF) 라 부릅니다:

$$P(X = x) \in [0, 1] \quad \text{and} \quad \sum_{x \in T} P(X = x) = 1$$


### 📌 2. 다변수 이산 확률분포의 3대 핵심 개념 (Figure 6.2 & Eq 6.9~6.14)

두 이산 확률변수 $X$ ($x_1, \dots, x_5$) 와 $Y$ ($y_1, \dots, y_3$) 의 2차원 격자 표에서:

1. 결합 확률 (Joint Probability: Eq 6.9):
   두 사건이 동시에 발생하는 교집합 확률입니다.
   $$P(X = x_i, Y = y_j) = P(X = x_i \cap Y = y_j) = \frac{n_{ij}}{N}$$
   (여기서 $n_{ij}$ 는 해당 셀의 사건 빈도, $N$ 은 전체 사건 수입니다.)

2. 주변 확률 (Marginal Probability: Eq 6.10~6.11 - ★ 소거/주변화):
   다른 변수 $Y$ 의 값에 상관없이 특정 변수 $X$ 만 발생할 확률입니다 (행/열의 합).
   $$P(X = x_i) = \frac{c_i}{N} = \frac{\sum_{j=1}^3 n_{ij}}{N} = \sum_{j} P(X = x_i, Y = y_j)$$
   (다른 변수를 합산하여 제거하는 이 연산을 주변화(Marginalization)라 부릅니다.)

3. 조건부 확률 (Conditional Probability: Eq 6.13~6.14):
   특정 변수 $X = x_i$ 가 이미 관측되었을 때, $Y = y_j$ 가 일어날 비중/확률입니다.
   $$P(Y = y_j \mid X = x_i) = \frac{n_{ij}}{c_i} = \frac{P(X = x_i, Y = y_j)}{P(X = x_i)}$$


## 2. ⚔️ Section 6.2.2: Continuous Probabilities (연속 확률분포, PDF와 CDF)


### 📌 1. 확률밀도함수 (PDF: Probability Density Function: Definition 6.1)

연속 확률변수 $\mathbf{x} \in \mathbb{R}^D$ 에 대한 함수 $f: \mathbb{R}^D \to \mathbb{R}$ 가 다음 두 공리를 만족하면 확률밀도함수(PDF) 라 부릅니다:

1. 모든 $\mathbf{x} \in \mathbb{R}^D$ 에 대해 비음성: $f(\mathbf{x}) \ge 0$
2. 전체 실수 공간에 대한 적분값이 1:
   $$\int_{\mathbb{R}^D} f(\mathbf{x}) d\mathbf{x} = 1 \quad (\text{Eq 6.15})$$

#### 💡 연속 확률변수의 핵심 법칙 (Eq 6.16)
연속 확률변수가 특정 구간 $[a, b]$ 에 존재할 확률은 PDF의 적분으로 정의됩니다:

$$P(a \le X \le b) = \int_a^b f(x) dx \quad (\text{Eq 6.16})$$

- ★ 치명적 차이점 (측도 0의 집합 Set of Measure Zero):
  연속 확률변수에서 단 한 점의 확률 $P(X = x) = 0$ 입니다! ($a = b$ 인 적분 구간의 길이가 0이기 때문).
  따라서 연속 확률에서는 한 점에서의 확률을 따지는 것이 불가능하며, $f(x)$ 자체는 확률이 아니라 "밀도(Density)"를 의미하므로 $f(x) > 1$ 일 수 있습니다!


### 📌 2. 누적분포함수 (CDF: Cumulative Distribution Function: Definition 6.2)

확률변수 $X = [X_1, \dots, X_D]^\top$ 이 특정 값 $\mathbf{x} = [x_1, \dots, x_D]^\top$ 이하일 누적 확률을 나타내는 함수입니다:

$$F_X(\mathbf{x}) = P(X_1 \le x_1, \dots, X_D \le x_D) = \int_{-\infty}^{x_1} \dots \int_{-\infty}^{x_D} f(z_1, \dots, z_D) dz_1 \dots dz_D \quad (\text{Eq 6.17~6.18})$$

- 미적분학의 기본 정리에 의해 $\frac{d F_X(x)}{dx} = f(x)$ 관계가 성립합니다.


## 3. ⚔️ Section 6.2.3: Contrasting Discrete and Continuous (이산 vs 연속 대조 총정리)


### 💡 [Example 6.3: 이산 균등분포 vs 연속 균등분포 대조 (Figure 6.3)]

1. 이산 균등분포 (Discrete Uniform Distribution):
   $Z \in \{-1.1, 0.3, 1.5\}$ 3개 상태에 대해:
   $$P(Z = -1.1) = P(Z = 0.3) = P(Z = 1.5) = \frac{1}{3} \le 1$$
   (모든 점 확률의 합: $\frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1$).

2. 연속 균등분포 (Continuous Uniform Distribution):
   구간 $0.9 \le X \le 1.6$ (구간 길이 $0.7$) 에서 고르게 분포하는 경우:
   $$p(x) = \frac{1}{1.6 - 0.9} = \frac{1}{0.7} \approx \mathbf{1.4285 > 1}$$
   - 해석: PDF의 높이(밀도)는 $1$보다 클 수 있습니다! 그러나 전체 구간을 적분하면 $\int_{0.9}^{1.6} 1.4285 dx = 1.4285 \times 0.7 = 1$ 이 되어 밀도 공리를 정확히 만족합니다.


### 📌 3. 확률분포 용어 명확 대조표 (Table 6.1)

| 구분 | 특정 한 점의 확률 ("Point Probability") | 구간의 누적 확률 ("Interval Probability") | 전체 확률 합산 조건 |
| :--- | :--- | :--- | :--- |
| 이산 확률변수 (Discrete) | 확률질량함수 (PMF) $P(X = x) \in [0, 1]$ | 해당 없음 (점 확률의 시그마 합) | $\sum_{x} P(X = x) = 1$ |
| 연속 확률변수 (Continuous) | $P(X = x) = 0$ (PDF 밀도 값 $p(x) \ge 0$ 는 1 초과 가능) | 누적분포함수 (CDF) $F(x) = P(X \le x)$ | $\int_{-\infty}^{\infty} p(x) dx = 1$ |


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 확률질량함수 (PMF $P(X=x)$): 이산 확률변수의 각 이산적 상태에 직접 할당되는 $0 \sim 1$ 범위의 확률값입니다.
- 확률밀도함수 (PDF $f(x)$): 연속 확률변수의 적분 구간 대비 상대적 빽빽함(밀도)을 나타내는 비음성 함수로, 적분값이 1이 됩니다.
- 주변화 (Marginalization $\sum_y P(x, y)$ or $\int p(x, y) dy$): 결합 확률에서 관심 없는 특정 변수를 합산/적분하여 소거하는 핵심 연산입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 분류(Classification) vs 회귀(Regression) 손실함수 정립: 딥러닝에서 출력이 이산적인 범주형 데이터인지, 연속적인 실숫값 데이터인지에 따라 손실함수의 수학적 정의(Cross-Entropy vs MSE / NLL)를 다르게 수립하기 위해 구분합니다.
- 잠재 변수(Latent Variables) 소거: VAE나 가우시안 혼합 모델(GMM)에서 관측할 수 없는 잠재 변수 $\mathbf{z}$ 를 주변화($\int p(\mathbf{x}, \mathbf{z})d\mathbf{z}$)하여 데이터 자체의 우도 $p(\mathbf{x})$ 를 얻기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- PMF의 점 확률 vs PDF의 밀도 값:
  - PMF: $P(X=x)$ 자체가 확률이므로 절대 1을 넘을 수 없습니다.
  - PDF: $f(x)$ 는 확률이 아니라 '단위 길이당 확률 밀도'이므로 구간이 $1$보다 좁으면 높이가 $1$을 초과할 수 있습니다 (예: 폭이 0.1이면 높이는 10).


### 4단계 실전 AI 연결고리]
- 분류 모델의 Cross-Entropy 손실함수:
  클래스 라벨이 이산적인 카테고리컬 분포(PMF)일 때, $L = -\sum_{k} y_k \ln P(Y=k \mid \mathbf{x})$ 로 손실을 산출.
- 생성 모델(VAE, Diffusion)의 Negative Log-Likelihood (NLL):
  생성 데이터 $\mathbf{x}$ 가 연속 공간(이미지 픽셀, 오디오 파형)에 존재하므로 연속 PDF $p_\theta(\mathbf{x})$ 의 적분 우도를 극대화하는 훈련 수행.
- 오브젝트 디텍션(YOLO, Faster R-CNN)의 Bounding Box & Class 예측:
  클래스 분류(이산 PMF: Person, Car, Dog)와 바운딩 박스 좌표 예측(연속 PDF: Center x, y, width, height)을 동시에 멀티타스크 학습.
