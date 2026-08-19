# 📐 5.7 & 5.8 Higher-Order Derivatives, Hessian Matrix & Multivariate Taylor Series (고계 도함수, 헤시안 곡률 행렬과 다변수 테일러 급수)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 5.7, 5.8 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. Chapter 5의 대단원: 왜 "헤시안 행렬(Hessian)"과 "다변수 테일러 급수"인가?

우리는 지금까지 1차 미분인 그래디언트($\nabla f$)와 야코비안($J$)을 배웠습니다. 그래디언트는 함수의 기울기(접선 경사도)를 알려주지만, "공간이 얼마나 가파르게 굽어있는지(곡률, Curvature)"는 전혀 알려주지 못합니다.

- 헤시안 행렬 (Hessian Matrix, $H \in \mathbb{R}^{n \times n}$): 다변수 함수의 모든 2계 편도함수($\frac{\partial^2 f}{\partial x_i \partial x_j}$)를 모아놓은 대칭 행렬로서, 고차원 손실 평면의 "국소 곡률과 볼록성(Convexity)"을 완벽하게 측정합니다.
- 다변수 테일러 급수 (Multivariate Taylor Series): 임의의 복잡한 다차원 곡면을 점 $\mathbf{x}_0$ 근방에서 "0차(높이) + 1차(접평면 그래디언트) + 2차(포물선 곡률 헤시안)" 로 근사 분해하는 수학적 도구입니다.
- 2차 최적화(Newton's Method)의 핵심: 1차 경사하강법(SGD)의 학습률 튜닝 한계를 극복하고, 헤시안 역행렬($H^{-1}$)을 이용해 2차 포물선 꼭짓점(최적점)으로 단번에 도약하는 뉴턴법(Newton-Raphson)의 이론적 근간이 완성됩니다.


## 1. ⚔️ Section 5.7: Higher-Order Derivatives & Hessian Matrix (고계 도함수와 헤시안)


### 📌 1. 고계 편도함수 표기법과 슈바르츠 정리 (Eq 5.146)

2변수 함수 $f(x, y)$ 에 대해 2차 편미분은 다음과 같이 정의됩니다:
- $\frac{\partial^2 f}{\partial x^2}$: $x$ 로 2번 연속 편미분
- $\frac{\partial^2 f}{\partial y^2}$: $y$ 로 2번 연속 편미분
- $\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y} (\frac{\partial f}{\partial x})$: $x$ 로 먼저 미분 후 $y$ 로 미분
- $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x} (\frac{\partial f}{\partial y})$: $y$ 로 먼저 미분 후 $x$ 로 미분

#### 💡 슈바르츠 정리 (Schwarz's Theorem / Clairaut's Theorem: Eq 5.146)
함수 $f$ 가 2회 연속 미분 가능한 매끄러운 함수($C^2$)이면, 미분하는 순서에 상관없이 교차 편미분 값이 완벽히 일치합니다:

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} \quad (\text{Eq 5.146})$$


### 📌 2. 헤시안 행렬(Hessian Matrix)의 정의와 대칭성 (Eq 5.147)

다변수 스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$ 의 모든 2계 편도함수를 모아놓은 $n \times n$ 행렬을 헤시안 행렬(Hessian Matrix) 이라 부릅니다:

$$H = \nabla^2_{\mathbf{x}} f(\mathbf{x}) := \begin{bmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\\\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix} \in \mathbb{R}^{n \times n} \quad (\text{Eq 5.147})$$

- 완벽한 대칭 행렬 ($H = H^\top$): 슈바르츠 정리에 의해 주대각선을 기준으로 대칭이므로 스펙트럴 정리(Chapter 4)가 적용되어 100% 실수 고유값과 정규직교 고유기저를 갖습니다!
- 기하학적 본질: 헤시안의 고유값($\lambda_i$)들은 각 주축 방향으로의 "순수한 곡률(얼마나 가파르게 휘어있는가)"을 나타냅니다.


### 📌 3. 헤시안과 손실 곡면 극값 판정 (양의 정정 행렬 연결)
- $H \succ 0$ (양의 정정 / 모든 고유값 $> 0$): 모든 방향으로 아래로 볼록한 U자형 곡면 ➡️ 국소 최솟값 (Local Minimum).
- $H \prec 0$ (음의 정정 / 모든 고유값 $< 0$): 모든 방향으로 위로 볼록한 돔형 곡면 ➡️ 국소 최댓값 (Local Maximum).
- 고유값 부호가 섞임: 한쪽은 오르막이고 다른 쪽은 내리막인 말안장 형태 ➡️ 안장점 (Saddle Point).


## 2. ⚔️ Section 5.8: Linearization and Multivariate Taylor Series (선형화와 다변수 테일러 급수)


### 📌 1. 국소 선형화 (Linearization: Figure 5.12 & Eq 5.148)

함수 $f$ 를 기준점 $\mathbf{x}_0$ 근방에서 1차 접평면으로 근사하는 기법입니다:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + (\nabla_\mathbf{x} f)(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0) \quad (\text{Eq 5.148})$$


### 📌 2. 다변수 테일러 급수의 정의 (Definitions 5.7 ~ 5.8 & Eq 5.149~5.155)

변위 벡터 $\boldsymbol{\delta} := \mathbf{x} - \mathbf{x}_0 \in \mathbb{R}^D$ 에 대해 다변수 테일러 급수는 다음과 같이 전개됩니다:

$$f(\mathbf{x}) = \sum_{k=0}^\infty \frac{D^k_\mathbf{x} f(\mathbf{x}_0)}{k!} \boldsymbol{\delta}^k \quad (\text{Eq 5.151})$$

- 외적 텐서 (Outer Product Tensor: Figure 5.13):
  - $\boldsymbol{\delta}^0 = 1$
  - $\boldsymbol{\delta}^1 = \boldsymbol{\delta} \in \mathbb{R}^D$
  - $\boldsymbol{\delta}^2 = \boldsymbol{\delta} \otimes \boldsymbol{\delta} = \boldsymbol{\delta}\boldsymbol{\delta}^\top \in \mathbb{R}^{D \times D}$ (외적 행렬, Eq 5.153)
  - $\boldsymbol{\delta}^3 = \boldsymbol{\delta} \otimes \boldsymbol{\delta} \otimes \boldsymbol{\delta} \in \mathbb{R}^{D \times D \times D}$ (3차원 텐서, Eq 5.154)


### 📌 3. 다변수 테일러 전개 0차~2차 핵심 분해 공식 (★ 딥러닝 2차 최적화의 심장!)

각 차수 $k$ 별 항을 행렬-벡터 연산으로 명쾌하게 분해합니다 (Eq 5.156~5.160):

1. $k=0$ (0차 상수항 / 높이):
   $$D^0_\mathbf{x} f(\mathbf{x}_0) \boldsymbol{\delta}^0 = f(\mathbf{x}_0) \in \mathbb{R}$$
2. $k=1$ (1차 선형항 / 접평면 경사도):
   $$D^1_\mathbf{x} f(\mathbf{x}_0) \boldsymbol{\delta}^1 = \nabla_\mathbf{x} f(\mathbf{x}_0) \boldsymbol{\delta} = \sum_{i=1}^D \frac{\partial f}{\partial x_i} \delta_i \in \mathbb{R}$$
3. $k=2$ (2차 이차항 / 2차 포물선 곡률):
   $$\frac{D^2_\mathbf{x} f(\mathbf{x}_0)}{2!} \boldsymbol{\delta}^2 = \frac{1}{2} \boldsymbol{\delta}^\top H(\mathbf{x}_0) \boldsymbol{\delta} = \frac{1}{2} \sum_{i=1}^D \sum_{j=1}^D H_{ij} \delta_i \delta_j \in \mathbb{R}$$

#### 👑 다변수 2차 테일러 근사 완결 공식 (Quadratic Approximation)
$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla_\mathbf{x} f(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^\top H(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)$$


### 💡 [Example 5.15: 2변수 다항함수의 테일러 전개 전수 수치 계산]
함수 $f(x, y) = x^2 + 2xy + y^3$ 에 대해 $(x_0, y_0) = (1, 2)$ 에서 테일러 전개:

1. 0차 상수항: $f(1, 2) = 1^2 + 2(1)(2) + 2^3 = 13$
2. 1차 편미분 및 그래디언트:
   $$\frac{\partial f}{\partial x} = 2x + 2y \implies 6, \quad \frac{\partial f}{\partial y} = 2x + 3y^2 \implies 14$$
   $$\nabla f(1, 2) = \begin{bmatrix} 6 & 14 \end{bmatrix} \implies 1\text{차 항} = 6(x - 1) + 14(y - 2)$$
3. 2차 편미분 및 헤시안 행렬:
   $$\frac{\partial^2 f}{\partial x^2} = 2, \quad \frac{\partial^2 f}{\partial y^2} = 6y \implies 12, \quad \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} = 2$$
   $$H(1, 2) = \begin{bmatrix} 2 & 2 \\\\ 2 & 12 \end{bmatrix}$$
   $$2\text{차 항} = \frac{1}{2} \begin{bmatrix} x-1 & y-2 \end{bmatrix} \begin{bmatrix} 2 & 2 \\\\ 2 & 12 \end{bmatrix} \begin{bmatrix} x-1 \\\\ y-2 \end{bmatrix} = (x - 1)^2 + 2(x - 1)(y - 2) + 6(y - 2)^2$$
4. 3차 편미분: 유일한 비영 성분 $\frac{\partial^3 f}{\partial y^3} = 6 \implies 3\text{차 항} = \frac{6}{3!}(y - 2)^3 = (y - 2)^3$
5. 최종 테일러 전개 결합 (Eq 5.180c):
   $$f(x, y) = 13 + 6(x - 1) + 14(y - 2) + (x - 1)^2 + 2(x - 1)(y - 2) + 6(y - 2)^2 + (y - 2)^3$$
   (원래 3차 다항식 $x^2 + 2xy + y^3$ 과 완벽히 100% 일치합니다!)


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 헤시안 행렬 ($H = \nabla^2 f \in \mathbb{R}^{n \times n}$): 다변수 스칼라 함수의 모든 2계 편도함수를 모아놓은 대칭 행렬이자 손실 곡면의 국소 곡률 측정기입니다.
- 다변수 테일러 급수: 임의의 매끄러운 다차원 함수를 점 $\mathbf{x}_0$ 근방에서 상수(0차), 그래디언트(1차 접평면), 헤시안(2차 포물선 곡률)으로 국소 근사하는 다항식 전개 도구입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 곡률 기반 초고속 2차 최적화 (뉴턴법): 1차 기울기만으로는 최적의 보폭(Learning Rate)을 알 수 없으므로, 헤시안 곡률 정보를 통해 손실 곡면의 바닥(꼭짓점)으로 단 1스텝 만에 도약하기 위해 사용합니다.
- 손실 평면의 임계점(Critical Point) 성격 판정: 그래디언트가 $\mathbf{0}$ 인 지점이 극솟값인지, 극댓값인지, 안장점(Saddle Point)인지 엄밀히 판정하기 위해 헤시안의 고유값을 분석합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 1차 최적화 (SGD, Adam) vs 2차 최적화 (Newton, L-BFGS):
  - 1차 최적화: 그래디언트 $\nabla f$ 만 계산하므로 스텝당 계산량 $O(D)$ 로 가볍지만, 곡률이 비대칭인 골짜기(Ravine)에서 진동하며 수렴이 느립니다.
  - 2차 최적화: 헤시안 역행렬 $H^{-1} \nabla f$ 로 최적 스텝을 단번에 찾지만, $D \times D$ 헤시안 구축 및 역행렬 계산량이 $O(D^3)$ 로 거대 모델에서는 메모리와 연산량이 폭발합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 뉴턴-랩슨 최적화 (Newton-Raphson Step - Ch 7):
  2차 테일러 근사의 최솟값 조건을 풀면 최적 이동 벡터가 유도됩니다:
  $$\mathbf{x}_{t+1} = \mathbf{x}_t - H(\mathbf{x}_t)^{-1} \nabla f(\mathbf{x}_t)^\top$$
- 준뉴턴법 (Quasi-Newton / L-BFGS & AdaHessian):
  $O(D^3)$ 의 헤시안 역행렬 계산을 피하기 위해 이전 스텝들의 그래디언트 차분 벡터를 이용해 $H^{-1}$ 을 저계수 행렬로 근사하여 초고속 2차 최적화를 수행합니다.
- 손실 평면 평탄도와 일반화 성능 (Flat vs Sharp Minima):
  헤시안의 최대 고유값 스펙트럼 노름 $\Vert H \Vert_2 = \lambda_{\text{max}}$ 이 작을수록 손실 평면이 완만한 평탄 최소점(Flat Minima)에 위치하며, 모델의 테스트 데이터 일반화(Generalization) 성능이 뛰어납니다.
