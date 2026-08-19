# 📐 5.0 & 5.1 Vector Calculus: Univariate Differentiation, Taylor Series & Chain Rule (벡터 미적분학의 서막, 일변수 미분과 테일러 급수)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 5 도입부 & Section 5.1 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. Chapter 5의 서막: 왜 "벡터 미적분학(Vector Calculus)"을 배우는가?

우리는 지난 Chapter 2~4에서 선형대수학의 기저, 사상, 내적, 직교 정사영, 그리고 행렬 분해(고유값 분해, 숄레스키, SVD)를 마스터했습니다.
하지만 머신러닝의 진정한 마법은 "데이터를 가장 잘 설명하도록 모델 파라미터를 끊임없이 조율하고 최적화(Optimization)하는 과정"에서 일어납니다.

- 곡선 적합 및 선형 회귀 (Linear Regression - Ch 9): 관측 데이터를 가장 잘 설명하는 최적 가중치 $w$ 를 찾기 위해 손실함수를 미분합니다.
- 심층 신경망 및 오토인코더 (Deep Neural Networks & VAE - Ch 10): 수억 개의 가중치 매개변수에 대해 재구성 오차를 최소화하기 위해 연쇄 법칙(Chain Rule)을 수없이 반복 적용하는 역전파(Backpropagation)를 수행합니다.
- 가우시안 혼합 모델 (GMM - Ch 11): 데이터 분포의 우도(Likelihood)를 극대화하기 위해 평균과 공분산 파라미터의 기울기(Gradient)를 계산합니다.

기울기(Gradient)는 언제나 "함수값이 가장 가파르게 증가하는 방향(Direction of Steepest Ascent)"을 가리킵니다.
Chapter 5는 이 모든 최신 AI 학습의 심장인 벡터 미적분학을 기초 일변수 미분부터 야코비안(Jacobian), 헤시안(Hessian), 다변수 연쇄법칙까지 빈틈없이 정복하는 여정입니다.


## 1. ⚔️ Section 5.1: Differentiation of Univariate Functions (일변수 함수의 미분)


### 📌 1. 차분 몫과 도함수의 정의 (Definitions 5.1 ~ 5.2 & Eq 5.3~5.4)

1. 차분 몫 (Difference Quotient: Definition 5.1 & Eq 5.3):
   함수 $y = f(x)$ 위의 두 점 $(x_0, f(x_0))$ 과 $(x_0 + \delta x, f(x_0 + \delta x))$ 을 잇는 할선(Secant line)의 평균 기울기를 의미합니다 (Figure 5.3):
   $$\frac{\delta y}{\delta x} := \frac{f(x + \delta x) - f(x)}{\delta x}$$

2. 도함수 (Derivative: Definition 5.2 & Eq 5.4):
   $\delta x = h \to 0$ 의 극한을 취할 때, 할선은 특정 점에서의 접선(Tangent line)으로 수렴하며 이 순간 변화율을 도함수라 부릅니다:
   $$\frac{df}{dx} := \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$
   - 기하학적 의미: 도함수 $\frac{df}{dx}$ 는 점 $x$ 에서 함수값이 가장 가파르게 증가하는 방향과 그 증가율을 나타냅니다.


### 📌 2. 다항함수 도함수의 이항정리 엄밀 유도 (Example 5.2 & Eq 5.5~5.6)

다항식 $f(x) = x^n$ ($n \in \mathbb{N}$) 의 도함수가 왜 $n x^{n-1}$ 이 되는지 도함수의 정의와 이항정리(Binomial Theorem)를 통해 엄밀히 유도합니다:

$$\frac{df}{dx} = \lim_{h \to 0} \frac{(x + h)^n - x^n}{h} = \lim_{h \to 0} \frac{\sum_{i=0}^n \binom{n}{i} x^{n-i} h^i - x^n}{h}$$

여기서 $i=0$ 일 때의 항 $\binom{n}{0} x^n h^0 = x^n$ 은 뒤의 $-x^n$ 과 상쇄되므로, 합의 시작을 $i=1$ 로 바꿉니다:

$$\frac{df}{dx} = \lim_{h \to 0} \frac{\sum_{i=1}^n \binom{n}{i} x^{n-i} h^i}{h} = \lim_{h \to 0} \sum_{i=1}^n \binom{n}{i} x^{n-i} h^{i-1}$$

$i=1$ 항을 분리하고 나머지 $i \ge 2$ 항들을 전개하면:

$$\frac{df}{dx} = \lim_{h \to 0} \left[ \binom{n}{1} x^{n-1} + \sum_{i=2}^n \binom{n}{i} x^{n-i} h^{i-1} \right] = \binom{n}{1} x^{n-1} + 0 = \frac{n!}{1!(n-1)!} x^{n-1} = n x^{n-1}$$


## 2. ⚔️ Section 5.1.1: Taylor Series (테일러 급수와 테일러 다항식의 본질)


### 💡 테일러 급수는 왜 탄생했는가? (직관적 배경 스토리)

삼각함수($\sin x, \cos x$), 지수/로그함수($e^x, \ln x$), 그리고 딥러닝의 복잡한 손실함수 $\text{Loss}(w)$ 는 곡선이 너무 복잡하여 컴퓨터가 최적점을 직접 찾기가 어렵습니다.
반면에 다항함수(1차 직선 $ax+b$, 2차 포물선 $ax^2+bx+c$)는 미분과 계산이 극도로 단순합니다.

수학자 브룩 테일러(Brook Taylor)는 생각했습니다:
"복잡한 곡선 $f(x)$ 가 있을 때, 특정 점 $x_0$ 근방에서 이 곡선과 100% 똑같이 행동하는 가짜 다항함수를 만들어낼 수는 없을까?"


### 📌 1. 테일러의 영혼 복사 원리: "미분값을 단계별로 똑같이 맞추기"

특정 기준점 $x_0$ 에서 원래 함수 $f(x)$ 와 가짜 다항식 $P(x)$ 가 똑같아지려면 다음 조건들을 순서대로 일치시켜야 합니다:

1. 0단계 (높이 맞춤): $x_0$ 에서 함수값이 같아야 합니다 ($f(x_0) = P(x_0)$).
2. 1단계 (기울기/접선 맞춤): $x_0$ 에서 1차 미분(기울기)이 같아야 합니다 ($f'(x_0) = P'(x_0)$).
3. 2단계 (곡률/포물선 맞춤): $x_0$ 에서 2차 미분(굽은 정도)이 같아야 합니다 ($f''(x_0) = P''(x_0)$).
4. 3단계 (비틀림 맞춤): $x_0$ 에서 3차 미분(비틀린 정도)이 같아야 합니다 ($f'''(x_0) = P'''(x_0)$).

이렇게 0차부터 $n$차까지의 모든 도함수 값을 영혼까지 똑같이 복사하여 이어 붙인 다항식이 바로 테일러 다항식입니다!


### 📌 2. 테일러 다항식과 테일러 급수의 정의 (Definitions 5.3 ~ 5.4 & Eq 5.7~5.8)

1. $n$차 테일러 다항식 (Taylor Polynomial: Definition 5.3 & Eq 5.7):
   $$T_n(x) := \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k$$

2. 수식 한 줄 뜯어보기:
   $$T_n(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \frac{f'''(x_0)}{3!}(x - x_0)^3 + \dots + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n$$
   - 첫 번째 항 $f(x_0)$: 점 $x_0$ 에서의 높이를 일치시키는 0차 상수항입니다.
   - 두 번째 항 $f'(x_0)(x - x_0)$: 점 $x_0$ 에서의 접선 기울기를 일치시키는 1차 선형 근사입니다.
   - 세 번째 항 $\frac{f''(x_0)}{2!}(x - x_0)^2$: 점 $x_0$ 에서의 곡률을 일치시키는 2차 포물선 근사입니다.

3. 왜 분모에 $k!$ (팩토리얼)이 들어가는가?:
   $(x - x_0)^k$ 를 $k$번 연속 미분하면 거듭제곱 지수가 앞으로 내려오며 $k \times (k-1) \times \dots \times 1 = k!$ 이 곱해집니다.
   이 튀어나온 계수 $k!$ 을 약분하여 원래의 $k$계 도함수 $f^{(k)}(x_0)$ 만 정확히 남기기 위해 분모에 $k!$ 로 나누어둔 것입니다!

4. 테일러 급수 (Taylor Series: Definition 5.4 & Eq 5.8):
   무한 번 미분 가능한 함수($f \in C^\infty$)에 대해 $n \to \infty$ 로 확장한 무한급수:
   $$T_\infty(x) := \sum_{k=0}^\infty \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k$$
   - 매클로린 급수 (Maclaurin Series): 기준점이 $x_0 = 0$ 인 특수한 테일러 급수.
   - 해석적 함수 (Analytic Function): 자신의 테일러 급수와 함수값이 완벽히 일치하는 함수 ($f(x) = T_\infty(x)$).


### 📌 3. 다항식의 테일러 전개 전수 계산 (Example 5.3)

$f(x) = x^4$ 에 대해 $x_0 = 1$ 에서 6차 테일러 다항식 $T_6(x)$ 전개:
- $f(1) = 1, \; f'(1) = 4, \; f''(1) = 12, \; f^{(3)}(1) = 24, \; f^{(4)}(1) = 24, \; f^{(5)}(1) = 0, \; f^{(6)}(1) = 0$.
- 계수 대입:
  $$T_6(x) = 1 + 4(x - 1) + \frac{12}{2!}(x - 1)^2 + \frac{24}{3!}(x - 1)^3 + \frac{24}{4!}(x - 1)^4 + 0$$
  $$T_6(x) = 1 + 4(x - 1) + 6(x - 1)^2 + 4(x - 1)^3 + (x - 1)^4 = x^4 = f(x)$$
  (원래 차수 이하의 다항식은 테일러 다항식과 100% 완벽히 일치합니다.)


### 📌 4. 삼각함수의 매클로린 급수 전개 (Example 5.4 & Figure 5.4)

$f(x) = \sin(x) + \cos(x)$ 의 $x_0 = 0$ 매클로린 급수 전개:
- $f(0) = 1, \; f'(0) = 1, \; f''(0) = -1, \; f^{(3)}(0) = -1, \; f^{(4)}(0) = 1 \dots$ (4주기로 계수 반복).
- 거듭제곱 급수 분리:
  $$T_\infty(x) = \sum_{k=0}^\infty \frac{(-1)^k}{(2k)!} x^{2k} + \sum_{k=0}^\infty \frac{(-1)^k}{(2k+1)!} x^{2k+1} = \cos(x) + \sin(x)$$
- 차수 $n$ 이 증가할수록($T_0 \to T_1 \to T_5 \to T_{10}$) 근사 영역이 $x \in [-4, 4]$ 이상으로 점차 확장됩니다 (Figure 5.4).


## 3. ⚔️ Section 5.1.2: Differentiation Rules (미분 4대 기본 법칙과 연쇄 법칙)


### 📌 1. 미분 4대 핵심 연산 법칙 (Eq 5.29~5.32)

1. 곱의 법칙 (Product Rule: Eq 5.29):
   $$(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)$$
2. 몫의 법칙 (Quotient Rule: Eq 5.30):
   $$\left( \frac{f(x)}{g(x)} \right)' = \frac{f'(x)g(x) - f(x)g'(x)}{(g(x))^2}$$
3. 합의 법칙 (Sum Rule: Eq 5.31):
   $$(f(x) + g(x))' = f'(x) + g'(x)$$
4. 연쇄 법칙 (Chain Rule: Eq 5.32 - ★ 딥러닝 역전파의 근간!):
   합성함수 $g \circ f$ ($x \mapsto f(x) \mapsto g(f(x))$) 에 대해:
   $$(g(f(x)))' = (g \circ f)'(x) = g'(f(x)) \cdot f'(x)$$


### 💡 [Example 5.5: 연쇄 법칙 수치 계산 예제]
$h(x) = (2x + 1)^4$ 의 미분:
- 내부 함수 $f(x) = 2x + 1 \implies f'(x) = 2$
- 외부 함수 $g(f) = f^4 \implies g'(f) = 4f^3$
- 연쇄 법칙 적용:
  $$h'(x) = g'(f(x)) \cdot f'(x) = 4(2x + 1)^3 \cdot 2 = 8(2x + 1)^3$$


## 🧠 4. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 도함수 ($\frac{df}{dx}$): 독립변수가 미세하게 변할 때 종속변수가 변하는 순간 변화율이자 접선의 기울기입니다.
- 테일러 급수 ($T_\infty(x) = \sum \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$): 임의의 매끄러운 비선형 곡선을 점 $x_0$ 근방에서 높이, 기울기, 곡률을 똑같이 맞추어 다항함수들의 합으로 근사 분해하는 도구입니다.
- 연쇄 법칙 ($(g \circ f)' = g'(f(x))f'(x)$): 합성함수의 미분을 각 단계별 도함수들의 곱으로 연결하는 미분 규칙입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 비선형 손실함수의 국소 다항식 근사: 복잡한 딥러닝 손실 평면을 점 $x_0$ 근방에서 1차(접선 기울기 $f'$) 및 2차(곡률 $f''$) 다항식으로 근사하여 다음 이동 위치를 결정하기 위해 테일러 급수를 사용합니다.
- 다층 신경망(Deep Layer)의 그래디언트 전파: 입력층부터 출력층까지 수십 개의 비선형 레이어가 합성된 함수를 한 단계씩 미분하여 가중치를 갱신하기 위해 연쇄 법칙을 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 1차 테일러 근사(경사하강법) vs 2차 테일러 근사(뉴턴-랩슨법 / Second-order Optimization):
  - 1차 근사 ($f(x) \approx f(x_0) + f'(x_0)(x-x_0)$): 접선 기울기만 사용하므로 계산량이 $O(D)$ 로 매우 저렴하지만, 곡률을 몰라 학습률 설정에 민감합니다 (경사하강법 SGD, Adam).
  - 2차 근사 ($f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2$): 2차 포물선 곡률(헤시안)을 고려해 포물선 꼭짓점으로 단번에 도약하지만, 2차 미분 계산량이 $O(D^2 \sim D^3)$ 로 급증합니다 (뉴턴법, L-BFGS).


### 4️⃣ [4단계 실전 AI 연결고리]
- 딥러닝 역전파 알고리즘 (Backpropagation / Autograd):
  PyTorch, JAX 등의 자동 미분 엔진은 연쇄 법칙 $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$ 을 계산 그래프(Computational Graph)를 따라 역방향으로 전파합니다.
- 최적화 알고리즘의 테일러 전개 (Gradient Descent & Newton's Method - Ch 7):
  경사하강법은 손실함수의 1차 테일러 근사를 최소화하는 방향($-\nabla f$)으로 이동하며, 2차 최적화 알고리즘(AdaHessian, L-BFGS)은 2차 테일러 근사를 활용해 최적 스텝 크기를 자동 조절합니다.
- 활성화 함수 미분 (ReLU, GELU, Sigmoid):
  역전파 계산 시 각 활성화 함수의 1차 도함수($\sigma'(x) = \sigma(x)(1-\sigma(x))$ 등)가 연쇄 법칙의 각 마디에 곱해집니다.
