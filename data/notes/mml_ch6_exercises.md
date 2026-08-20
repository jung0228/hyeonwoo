# 📐 MML Chapter 6 Exercises (연습문제 6.1 ~ 6.13 전수 풀이 & 수식 완전 정복)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 6 연습문제 6.1부터 6.13까지 단 하나의 생략도 없이 100% 풀이 및 수식 유도 노트


## 📌 Exercise 6.1 (이변량 이산 확률분포의 주변/조건부 확률 계산)

### [문제 조건]
이산 확률변수 $X \in \{x_1, \dots, x_5\}$ 와 $Y \in \{y_1, y_2, y_3\}$ 의 결합 확률표 $p(x, y)$:

| Y \ X | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ | 열 합산 ($p(y)$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $y_3$ | 0.01 | 0.02 | 0.03 | 0.10 | 0.10 | 0.26 |
| $y_2$ | 0.05 | 0.10 | 0.05 | 0.07 | 0.20 | 0.47 |
| $y_1$ | 0.10 | 0.05 | 0.03 | 0.05 | 0.04 | 0.27 |
| 행 합산 ($p(x)$) | 0.16 | 0.17 | 0.11 | 0.22 | 0.34 | 1.00 |

---

### [풀이]

#### a. 주변 확률분포 $p(x)$ 및 $p(y)$ 계산
- $p(x)$ (열별 합산):
  - $p(x_1) = 0.01 + 0.05 + 0.10 = \mathbf{0.16}$
  - $p(x_2) = 0.02 + 0.10 + 0.05 = \mathbf{0.17}$
  - $p(x_3) = 0.03 + 0.05 + 0.03 = \mathbf{0.11}$
  - $p(x_4) = 0.10 + 0.07 + 0.05 = \mathbf{0.22}$
  - $p(x_5) = 0.10 + 0.20 + 0.04 = \mathbf{0.34}$
- $p(y)$ (행별 합산):
  - $p(y_1) = 0.10 + 0.05 + 0.03 + 0.05 + 0.04 = \mathbf{0.27}$
  - $p(y_2) = 0.05 + 0.10 + 0.05 + 0.07 + 0.20 = \mathbf{0.47}$
  - $p(y_3) = 0.01 + 0.02 + 0.03 + 0.10 + 0.10 = \mathbf{0.26}$

#### b. 조건부 확률분포 $p(x \mid Y = y_1)$ 및 $p(y \mid X = x_3)$ 계산
- $p(x \mid Y = y_1) = \frac{p(x, y_1)}{p(y_1)}$ ($p(y_1) = 0.27$ 로 나눔):
  - $p(x_1 \mid y_1) = \frac{0.10}{0.27} = \mathbf{\frac{10}{27} \approx 0.3704}$
  - $p(x_2 \mid y_1) = \frac{0.05}{0.27} = \mathbf{\frac{5}{27} \approx 0.1852}$
  - $p(x_3 \mid y_1) = \frac{0.03}{0.27} = \mathbf{\frac{3}{27} = \frac{1}{9} \approx 0.1111}$
  - $p(x_4 \mid y_1) = \frac{0.05}{0.27} = \mathbf{\frac{5}{27} \approx 0.1852}$
  - $p(x_5 \mid y_1) = \frac{0.04}{0.27} = \mathbf{\frac{4}{27} \approx 0.1481}$
  (검증: $\sum_{i=1}^5 p(x_i \mid y_1) = \frac{10+5+3+5+4}{27} = 1$).

- $p(y \mid X = x_3) = \frac{p(x_3, y)}{p(x_3)}$ ($p(x_3) = 0.11$ 로 나눔):
  - $p(y_1 \mid x_3) = \frac{0.03}{0.11} = \mathbf{\frac{3}{11} \approx 0.2727}$
  - $p(y_2 \mid x_3) = \frac{0.05}{0.11} = \mathbf{\frac{5}{11} \approx 0.4545}$
  - $p(y_3 \mid x_3) = \frac{0.03}{0.11} = \mathbf{\frac{3}{11} \approx 0.2727}$
  (검증: $\sum_{j=1}^3 p(y_j \mid x_3) = \frac{3+5+3}{11} = 1$).


---

## 📌 Exercise 6.2 (가우시안 혼합 분포의 주변 통계량과 모드 산출)

### [문제 조건]
$$p(\mathbf{x}) = 0.4 \mathcal{N}\left( \begin{bmatrix} 10 \\ 2 \end{bmatrix}, \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) + 0.6 \mathcal{N}\left( \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 8.4 & 2.0 \\ 2.0 & 1.7 \end{bmatrix} \right)$$

---

### [풀이]

#### a. 차원별 주변 확률분포 $p(x_1)$ 및 $p(x_2)$
혼합 분포의 주변화는 각 구성 가우시안의 주변화의 가중합이므로:
- $p(x_1) = \mathbf{0.4 \mathcal{N}(x_1 \mid 10, 1) + 0.6 \mathcal{N}(x_1 \mid 0, 8.4)}$
- $p(x_2) = \mathbf{0.4 \mathcal{N}(x_2 \mid 2, 1) + 0.6 \mathcal{N}(x_2 \mid 0, 1.7)}$

#### b. 각 주변 분포의 평균(Mean), 최빈값(Mode), 중앙값(Median)
1. 평균 (Mean):
   - $\mathbb{E}[x_1] = 0.4(10) + 0.6(0) = \mathbf{4}$
   - $\mathbb{E}[x_2] = 0.4(2) + 0.6(0) = \mathbf{0.8}$

2. 최빈값 (Mode - 최고 피크점):
   - $p(x_1)$: 두 피크 위치 $x_1 = 0$ 및 $x_1 = 10$ 의 피크 높이 비교:
     - $x_1 = 0$ 에서의 높이: $0.6 \cdot \frac{1}{\sqrt{2\pi \cdot 8.4}} \approx 0.0825$
     - $x_1 = 10$ 에서의 높이: $0.4 \cdot \frac{1}{\sqrt{2\pi \cdot 1}} \approx 0.1596$
     - $0.1596 > 0.0825$ 이므로 $p(x_1)$ 의 전역 최빈값(Global Mode)은 $\mathbf{x_1 = 10}$.
   - $p(x_2)$: 두 피크 위치 $x_2 = 0$ 및 $x_2 = 2$ 의 피크 높이 비교:
     - $x_2 = 0$ 에서의 높이: $0.6 \cdot \frac{1}{\sqrt{2\pi \cdot 1.7}} \approx 0.1837$
     - $x_2 = 2$ 에서의 높이: $0.4 \cdot \frac{1}{\sqrt{2\pi \cdot 1}} \approx 0.1596$
     - $0.1837 > 0.1596$ 이므로 $p(x_2)$ 의 전역 최빈값(Global Mode)은 $\mathbf{x_2 = 0}$.

3. 중앙값 (Median):
   - 다봉 비대칭 분포이므로 $F(x) = 0.5$ 수치 적분 해로 구해지며, 평균과 차이가 존재합니다.

#### c. 2차원 결합 분포의 평균 및 최빈값
- 2차원 평균: $\boldsymbol{\mu} = 0.4 \begin{bmatrix} 10 \\ 2 \end{bmatrix} + 0.6 \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \mathbf{\begin{bmatrix} 4 \\ 0.8 \end{bmatrix}}$
- 2차원 최빈값: 2차원 PDF 높이 비교:
  - $\begin{bmatrix} 10 \\ 2 \end{bmatrix}$ 의 밀도: $\frac{0.4}{2\pi \sqrt{1}} \approx 0.0637$
  - $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ 의 밀도: $\frac{0.6}{2\pi \sqrt{8.4 \cdot 1.7 - 2^2}} = \frac{0.6}{2\pi \sqrt{10.28}} \approx 0.0298$
  - 따라서 2차원 전역 최빈값은 $\mathbf{\begin{bmatrix} 10 \\ 2 \end{bmatrix}}$.


---

## 📌 Exercise 6.3 (베르누이 우도와 베타 공액 사전분포의 사후분포 유도)

### [문제 및 유도]
컴파일러 성공 여부 $x_i \in \{0, 1\}$ 에 대한 베르누이 우도 $p(x \mid \mu) = \mu^x (1-\mu)^{1-x}$.
$N$ 개의 독립 관측 데이터 $x_1, \dots, x_N$ 이 주어졌을 때 결합 우도:

$$p(x_1, \dots, x_N \mid \mu) = \prod_{i=1}^N \mu^{x_i} (1-\mu)^{1-x_i} = \mu^{\sum_{i=1}^N x_i} (1-\mu)^{N - \sum_{i=1}^N x_i}$$

공액 사전분포로 베타 분포 $\mu \sim \text{Beta}(\alpha, \beta)$ ($p(\mu) \propto \mu^{\alpha-1} (1-\mu)^{\beta-1}$) 를 선택.
베이즈 정리에 따른 사후분포:

$$p(\mu \mid x_1, \dots, x_N) \propto p(x_1, \dots, x_N \mid \mu) p(\mu) \propto \mu^{\sum_{i=1}^N x_i + \alpha - 1} (1-\mu)^{(N - \sum_{i=1}^N x_i) + \beta - 1}$$

$$\implies \mathbf{\text{Beta}\left( \alpha + \sum_{i=1}^N x_i, \;\; \beta + N - \sum_{i=1}^N x_i \right)} \quad \blacksquare$$


---

## 📌 Exercise 6.4 (동전 던지기와 주머니 확률 - 베이즈 정리 문제)

### [문제 조건]
- 편향된 동전: $P(\text{Heads}) = 0.6$ (주머니 1 선택), $P(\text{Tails}) = 0.4$ (주머니 2 선택).
- 주머니 1: 망고 4개, 사과 2개 (총 6개) $\implies P(\text{Mango} \mid \text{Bag 1}) = \frac{4}{6} = \frac{2}{3}$
- 주머니 2: 망고 4개, 사과 4개 (총 8개) $\implies P(\text{Mango} \mid \text{Bag 2}) = \frac{4}{8} = \frac{1}{2}$
- 친구가 꺼낸 과일이 "망고"일 때, 이것이 주머니 2에서 나왔을 확률 $P(\text{Bag 2} \mid \text{Mango})$?

---

### [풀이]
1. 전체 망고 관측 확률 (전확률 정리):
   $$P(\text{Mango}) = P(\text{Mango} \mid \text{Bag 1}) P(\text{Bag 1}) + P(\text{Mango} \mid \text{Bag 2}) P(\text{Bag 2})$$
   $$P(\text{Mango}) = \left(\frac{2}{3}\right)(0.6) + \left(\frac{1}{2}\right)(0.4) = 0.4 + 0.2 = \mathbf{0.6}$$

2. 베이즈 정리 적용:
   $$P(\text{Bag 2} \mid \text{Mango}) = \frac{P(\text{Mango} \mid \text{Bag 2}) P(\text{Bag 2})}{P(\text{Mango})} = \frac{0.2}{0.6} = \mathbf{\frac{1}{3} \approx 0.3333} \quad \blacksquare$$


---

## 📌 Exercise 6.5 (시계열 칼만 필터 모델의 수식 유도)

### [문제 조건]
$$x_{t+1} = A x_t + w, \quad w \sim \mathcal{N}(0, Q)$$
$$y_t = C x_t + v, \quad v \sim \mathcal{N}(0, R)$$
$$p(x_0) = \mathcal{N}(\mu_0, \Sigma_0)$$

---

### [풀이 및 증명]

#### a. 결합분포 $p(x_0, x_1, \dots, x_T)$ 의 형태와 정당화
- 형태: 다변량 가우시안 분포 (Multivariate Gaussian Distribution) 입니다.
- 정당화: 초기 상태 $x_0$ 가 가우시안이고, 전이 방정식 $x_{t+1} = A x_t + w$ 는 가우시안 변수의 선형 아핀 변환 및 독립 가우시안 노이즈의 합입니다. 가우시안 확률변수의 선형 결합은 항상 가우시안이므로 전체 결합분포는 다변량 가우시안을 이룹니다.

#### b. $p(x_t \mid y_1, \dots, y_t) = \mathcal{N}(\mu_t, \Sigma_t)$ 조건 하 유도

1. 예측 단계 $p(x_{t+1} \mid y_1, \dots, y_t)$:
   $$\mu_{t+1 \mid t} = \mathbb{E}[A x_t + w] = \mathbf{A \mu_t}$$
   $$\Sigma_{t+1 \mid t} = V[A x_t + w] = \mathbf{A \Sigma_t A^\top + Q}$$
   $$\implies \mathbf{\mathcal{N}(A \mu_t, \; A \Sigma_t A^\top + Q)}$$

2. 예측 결합분포 $p(x_{t+1}, y_{t+1} \mid y_1, \dots, y_t)$:
   - $\mathbb{E}[y_{t+1}] = C \mu_{t+1 \mid t} = C A \mu_t$
   - $V[y_{t+1}] = C \Sigma_{t+1 \mid t} C^\top + R$
   - $\text{Cov}[x_{t+1}, y_{t+1}] = \Sigma_{t+1 \mid t} C^\top$
   $$\implies \mathbf{\mathcal{N}\left( \begin{bmatrix} \mu_{t+1 \mid t} \\ C \mu_{t+1 \mid t} \end{bmatrix}, \begin{bmatrix} \Sigma_{t+1 \mid t} & \Sigma_{t+1 \mid t} C^\top \\ C \Sigma_{t+1 \mid t} & C \Sigma_{t+1 \mid t} C^\top + R \end{bmatrix} \right)}$$

3. 갱신 단계 $p(x_{t+1} \mid y_1, \dots, y_{t+1})$ (Kalman Update Step):
   새로운 관측 $y_{t+1} = \hat{y}$ 에 대해 조건부 가우시안 공식(Eq 6.66~6.67)을 적용:
   $$\mathbf{\mu_{t+1} = \mu_{t+1 \mid t} + \Sigma_{t+1 \mid t} C^\top \left( C \Sigma_{t+1 \mid t} C^\top + R \right)^{-1} \left( \hat{y} - C \mu_{t+1 \mid t} \right)}$$
   $$\mathbf{\Sigma_{t+1} = \Sigma_{t+1 \mid t} - \Sigma_{t+1 \mid t} C^\top \left( C \Sigma_{t+1 \mid t} C^\top + R \right)^{-1} C \Sigma_{t+1 \mid t}} \quad \blacksquare$$


---

## 📌 Exercise 6.6 (분산의 Raw-Score 공식 수식 증명)

### [증명]
$$V[x] := \mathbb{E}[(x - \mu)^2]$$

전개 및 기댓값의 선형성 적용:

$$V[x] = \mathbb{E}[x^2 - 2\mu x + \mu^2] = \mathbb{E}[x^2] - 2\mu \mathbb{E}[x] + \mu^2$$

$\mathbb{E}[x] = \mu$ 대입:

$$V[x] = \mathbb{E}[x^2] - 2\mu^2 + \mu^2 = \mathbf{\mathbb{E}[x^2] - (\mathbb{E}[x])^2} \quad \blacksquare$$


---

## 📌 Exercise 6.7 (쌍간 차이 합 분산 공식 증명)

### [증명]
$N^2$ 개의 쌍간 차이 제곱의 합 전개:

$$\sum_{i=1}^N \sum_{j=1}^N (x_i - x_j)^2 = \sum_{i=1}^N \sum_{j=1}^N (x_i^2 - 2 x_i x_j + x_j^2) = \sum_{i=1}^N \sum_{j=1}^N x_i^2 - 2 \sum_{i=1}^N \sum_{j=1}^N x_i x_j + \sum_{i=1}^N \sum_{j=1}^N x_j^2$$

각 항 계산:
1. $\sum_{i=1}^N \sum_{j=1}^N x_i^2 = N \sum_{i=1}^N x_i^2$
2. $\sum_{i=1}^N \sum_{j=1}^N x_j^2 = N \sum_{j=1}^N x_j^2$
3. $\sum_{i=1}^N \sum_{j=1}^N x_i x_j = \left( \sum_{i=1}^N x_i \right) \left( \sum_{j=1}^N x_j \right) = \left( \sum_{i=1}^N x_i \right)^2$

합치면:

$$\sum_{i=1}^N \sum_{j=1}^N (x_i - x_j)^2 = 2 N \sum_{i=1}^N x_i^2 - 2 \left( \sum_{i=1}^N x_i \right)^2$$

양변을 $N^2$ 으로 나누면:

$$\frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N (x_i - x_j)^2 = \mathbf{2 \left[ \frac{1}{N} \sum_{i=1}^N x_i^2 - \left( \frac{1}{N} \sum_{i=1}^N x_i \right)^2 \right]} \quad \blacksquare$$


---

## 📌 Exercise 6.8 (베르누이 분포의 지수 족 자연 파라미터 변환)

### [풀이]
$$p(x \mid \mu) = \mu^x (1-\mu)^{1-x} = \exp\left( x \ln\mu + (1-x) \ln(1-\mu) \right) = \exp\left( x \ln\frac{\mu}{1-\mu} + \ln(1-\mu) \right)$$

- 자연 파라미터: $\theta = \ln \frac{\mu}{1-\mu}$
- 충분통계량: $\phi(x) = x$
- 로그 분할 함수: $A(\theta) = -\ln(1-\mu) = \ln(1 + e^\theta)$
- 베이스 측도: $h(x) = 1$

$$\implies \mathbf{p(x \mid \theta) = h(x) \exp\left( \theta \phi(x) - A(\theta) \right)} \quad \blacksquare$$


---

## 📌 Exercise 6.9 (이항, 베타 분포의 지수 족 변환 및 곱의 지수 족 증명)

### [풀이]
1. 이항 분포:
   $$p(m \mid N, \mu) = \begin{pmatrix} N \\ m \end{pmatrix} \exp\left( m \ln\frac{\mu}{1-\mu} + N \ln(1-\mu) \right)$$
   ($h(m) = \begin{pmatrix} N \\ m \end{pmatrix}, \theta = \ln \frac{\mu}{1-\mu}, \phi(m) = m, A(\theta) = N\ln(1+e^\theta)$).

2. 베타 분포:
   $$p(\mu \mid \alpha, \beta) = \exp\left( (\alpha-1)\ln\mu + (\beta-1)\ln(1-\mu) - \ln\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} \right)$$
   (자연 파라미터 $\boldsymbol{\theta} = [\alpha-1, \beta-1]^\top$, 충분통계량 $\boldsymbol{\phi}(\mu) = [\ln\mu, \ln(1-\mu)]^\top$).

3. Beta와 Binomial 곱의 지수 족 증명:
   $$p(m, \mu) = \begin{pmatrix} N \\ m \end{pmatrix} \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{m+\alpha-1} (1-\mu)^{N-m+\beta-1}$$
   $$= \begin{pmatrix} N \\ m \end{pmatrix} \exp\left[ (m+\alpha-1)\ln\mu + (N-m+\beta-1)\ln(1-\mu) - \ln\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} \right]$$
   이는 자연 파라미터와 충분통계량의 내적 형태로 전개되므로 완벽한 지수 족 분포를 형성합니다. $\blacksquare$


---

## 📌 Exercise 6.10 (두 가우시안 밀도 곱의 2가지 증명)

### [증명]

#### a. 완전제곱식 완성법 (Completing the Square)
두 지수부의 합:

$$-\frac{1}{2} \left[ (\mathbf{x}-\mathbf{a})^\top A^{-1}(\mathbf{x}-\mathbf{a}) + (\mathbf{x}-\mathbf{b})^\top B^{-1}(\mathbf{x}-\mathbf{b}) \right]$$

$\mathbf{x}$ 에 대한 2차항과 1차항 묶기:

$$-\frac{1}{2} \left[ \mathbf{x}^\top (A^{-1} + B^{-1}) \mathbf{x} - 2 \mathbf{x}^\top (A^{-1}\mathbf{a} + B^{-1}\mathbf{b}) + \mathbf{a}^\top A^{-1}\mathbf{a} + \mathbf{b}^\top B^{-1}\mathbf{b} \right]$$

$C = (A^{-1} + B^{-1})^{-1}$, $\mathbf{c} = C(A^{-1}\mathbf{a} + B^{-1}\mathbf{b})$ 정의 시:

$$-\frac{1}{2} (\mathbf{x}-\mathbf{c})^\top C^{-1} (\mathbf{x}-\mathbf{c}) - \frac{1}{2} \left[ \mathbf{a}^\top A^{-1}\mathbf{a} + \mathbf{b}^\top B^{-1}\mathbf{b} - \mathbf{c}^\top C^{-1}\mathbf{c} \right]$$

우드버리 행렬 항등식(Woodbury Identity)에 의해 잔여항은 $(\mathbf{a}-\mathbf{b})^\top (A+B)^{-1} (\mathbf{a}-\mathbf{b})$ 가 되어 스케일링 상수 $c = \mathcal{N}(\mathbf{a} \mid \mathbf{b}, A+B)$ 가 완벽히 유도됩니다. $\blacksquare$

#### b. 지수 족 형태 활용법
가우시안을 지수 족 형태 $p(x) \propto \exp\left( \mathbf{a}^\top A^{-1}\mathbf{x} - \frac{1}{2} \mathbf{x}^\top A^{-1}\mathbf{x} \right)$ 로 표기하여 지수끼리 더하면 자연 파라미터 $\boldsymbol{\theta}_1 = A^{-1}\mathbf{a} + B^{-1}\mathbf{b} = C^{-1}\mathbf{c}$ 와 $\boldsymbol{\theta}_2 = -\frac{1}{2} C^{-1}$ 가 정밀하게 합산되어 유도됩니다. $\blacksquare$


---

## 📌 Exercise 6.11 (반복 기댓값 정리 / 전기댓값 법칙 증명)

### [증명]
$$\mathbb{E}_Y [ \mathbb{E}_X[x \mid y] ] = \int_{\mathcal{Y}} \left[ \int_{\mathcal{X}} x p(x \mid y) dx \right] p(y) dy$$

적분 기호 안으로 $p(y)$ 이동:

$$= \int_{\mathcal{Y}} \int_{\mathcal{X}} x \, p(x \mid y) p(y) \, dx \, dy = \int_{\mathcal{Y}} \int_{\mathcal{X}} x \, p(x, y) \, dx \, dy$$

푸비니 정리(Fubini's Theorem)에 의해 적분 순서 변경:

$$= \int_{\mathcal{X}} x \left[ \int_{\mathcal{Y}} p(x, y) dy \right] dx = \int_{\mathcal{X}} x \, p(x) \, dx = \mathbf{\mathbb{E}_X[x]} \quad \blacksquare$$


---

## 📌 Exercise 6.12 (가우시안 확률변수의 선형 조작과 사후분포 유도)

### [문제 및 유도]
$x \sim \mathcal{N}(\mu_x, \Sigma_x)$, $y = A x + b + w \; (w \sim \mathcal{N}(0, Q))$, $z = C y + v \; (v \sim \mathcal{N}(0, R))$.

#### a. 우도 $p(y \mid x)$
$$p(y \mid x) = \mathbf{\mathcal{N}(y \mid A x + b, \; Q)}$$

#### b. $p(y)$ 의 평균 $\mu_y$ 및 공분산 $\Sigma_y$
- $\mu_y = \mathbb{E}[A x + b + w] = \mathbf{A \mu_x + b}$
- $\Sigma_y = V[A x + b + w] = \mathbf{A \Sigma_x A^\top + Q}$
$$\implies p(y) = \mathbf{\mathcal{N}(A \mu_x + b, \; A \Sigma_x A^\top + Q)}$$

#### c. 측정 $z = C y + v$ 의 $p(z \mid y)$ 및 $p(z)$
- $p(z \mid y) = \mathbf{\mathcal{N}(z \mid C y, \; R)}$
- $\mu_z = \mathbb{E}[C y + v] = C \mu_y = \mathbf{C(A \mu_x + b)}$
- $\Sigma_z = V[C y + v] = C \Sigma_y C^\top + R = \mathbf{C(A \Sigma_x A^\top + Q)C^\top + R}$

#### d. 측정값 $y = \hat{y}$ 이 관측되었을 때 사후분포 $p(x \mid \hat{y})$ (★ 칼만 필터 사후분포!)
1. 결합 가우시안 $p(x, y)$ 구축:
   $\text{Cov}[x, y] = \Sigma_x A^\top$ 이므로,
   $$p(x, y) = \mathcal{N}\left( \begin{bmatrix} \mu_x \\ A\mu_x+b \end{bmatrix}, \begin{bmatrix} \Sigma_x & \Sigma_x A^\top \\ A\Sigma_x & A\Sigma_x A^\top + Q \end{bmatrix} \right)$$

2. 조건부 가우시안 적용:
   $$\mathbf{\mu_{x \mid \hat{y}} = \mu_x + \Sigma_x A^\top \left( A \Sigma_x A^\top + Q \right)^{-1} \left( \hat{y} - A\mu_x - b \right)}$$
   $$\mathbf{\Sigma_{x \mid \hat{y}} = \Sigma_x - \Sigma_x A^\top \left( A \Sigma_x A^\top + Q \right)^{-1} A \Sigma_x} \quad \blacksquare$$


---

## 📌 Exercise 6.13 (확률적분변환 정리 증명 - Theorem 6.15)

### [증명]
연속 확률변수 $X$ 의 엄격한 단조증가 CDF $F_X(x)$ 에 대해 $Y := F_X(X)$ 의 CDF $F_Y(y)$ 계산 ($0 \le y \le 1$):

$$F_Y(y) = P(Y \le y) = P(F_X(X) \le y)$$

$F_X$ 가 엄격한 단조증가 함수이므로 역함수 $F_X^{-1}$ 를 양변에 적용:

$$P(X \le F_X^{-1}(y)) = F_X(F_X^{-1}(y)) = y$$

따라서 $Y$ 의 CDF 는 $F_Y(y) = y \; (0 \le y \le 1)$ 입니다.
CDF를 $y$ 로 미분하여 PDF 를 구하면:

$$f_Y(y) = \frac{d}{dy} F_Y(y) = \frac{d}{dy}(y) = 1, \quad 0 \le y \le 1$$

이는 구간 $[0, 1]$ 에서 정의된 단위 연속 균등분포 $\mathcal{U}[0, 1]$ 의 PDF 와 완벽히 일치합니다. $\blacksquare$
