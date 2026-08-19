# 📐 5.3 Gradients of Vector-Valued Functions (벡터값 함수의 그래디언트, 야코비안 행렬식과 선형 회귀 손실함수 미분)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 5.3 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 벡터 미적분학의 정점: 왜 "벡터값 함수와 야코비안(Jacobian)"인가?

우리는 앞선 5.2절에서 벡터를 입력받아 스칼라 1개를 출력하는 함수($f: \mathbb{R}^n \to \mathbb{R}$)의 그래디언트를 다루었습니다.
하지만 심층 신경망(Deep Neural Networks)의 각 은닉층(Layer), 공간 좌표 변환, 생성 모델(VAE, Normalizing Flows) 등 실제 AI의 핵심 연산은 대부분 "벡터를 입력받아 또 다른 고차원 벡터를 출력하는 벡터값 함수(Vector-Valued Function, $f: \mathbb{R}^n \to \mathbb{R}^m$)"입니다.

- 야코비안 행렬 (Jacobian Matrix, $J \in \mathbb{R}^{m \times n}$): $n$개의 입력 변수와 $m$개의 출력 변수 사이에 존재하는 모든 편미분 $m \times n$ 개를 빠짐없이 총망라한 행렬입니다.
- 국소 1차 선형 근사기: 비선형 공간에서 입력의 미세 변화 $\Delta \mathbf{x}$ 가 출력의 미세 변화 $\Delta \mathbf{y}$ 로 변환되는 순간 관계식을 $\Delta \mathbf{y} \approx J \Delta \mathbf{x}$ 로 완벽히 모델링합니다.
- 야코비안 행렬식과 확률 변수 변환 ($|\det(J)|$): 공간이 몇 배로 팽창/수축하는지의 부피 배율을 제공하여, 정규화 흐름(Normalizing Flows)과 VAE의 확률밀도함수 변환의 절대적 기초가 됩니다.


## 1. ⚔️ Section 5.3: 벡터값 함수의 편미분과 야코비안 행렬 (Definition 5.6)


### 📌 1. 벡터값 함수의 정의와 성분별 편미분 (Eq 5.54~5.55)

함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ ($\mathbf{x} = [x_1, \dots, x_n]^\top \in \mathbb{R}^n$) 의 출력은 $m$개의 스칼라 함수들의 열벡터로 표현됩니다:

$$\mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_1(\mathbf{x}) \\\\ \vdots \\\\ f_m(\mathbf{x}) \end{bmatrix} \in \mathbb{R}^m \quad (\text{Eq 5.54})$$

입력 변수 $x_i$ 에 대한 벡터 함수 $\mathbf{f}$ 의 편미분은 각 출력 성분 $f_1, \dots, f_m$ 을 각각 편미분한 $m$차원 열벡터가 됩니다:

$$\frac{\partial \mathbf{f}}{\partial x_i} = \begin{bmatrix} \frac{\partial f_1}{\partial x_i} \\\\ \vdots \\\\ \frac{\partial f_m}{\partial x_i} \end{bmatrix} = \begin{bmatrix} \lim_{h \to 0} \frac{f_1(x_1, \dots, x_i + h, \dots, x_n) - f_1(\mathbf{x})}{h} \\\\ \vdots \\\\ \lim_{h \to 0} \frac{f_m(x_1, \dots, x_i + h, \dots, x_n) - f_m(\mathbf{x})}{h} \end{bmatrix} \in \mathbb{R}^m \quad (\text{Eq 5.55})$$


### 📌 2. 야코비안 행렬의 정의 (Jacobian: Definition 5.6 & Eq 5.56~5.59)

모든 $n$개 입력 변수에 대한 편미분 열벡터들을 가로로 나란히 모으면 $m \times n$ 크기의 야코비안 행렬(Jacobian Matrix) 이 완성됩니다:

$$J = \nabla_\mathbf{x} \mathbf{f} = \frac{d\mathbf{f}(\mathbf{x})}{d\mathbf{x}} := \begin{bmatrix} \frac{\partial \mathbf{f}(\mathbf{x})}{\partial x_1} & \dots & \frac{\partial \mathbf{f}(\mathbf{x})}{\partial x_n} \end{bmatrix} = \begin{bmatrix} \frac{\partial f_1(\mathbf{x})}{\partial x_1} & \dots & \frac{\partial f_1(\mathbf{x})}{\partial x_n} \\\\ \vdots & \ddots & \vdots \\\\ \frac{\partial f_m(\mathbf{x})}{\partial x_1} & \dots & \frac{\partial f_m(\mathbf{x})}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n} \quad (\text{Eq 5.58})$$

$$J(i, j) = \frac{\partial f_i}{\partial x_j} \quad (\text{Row } i: i\text{번째 출력 함수, Column } j: j\text{번째 입력 변수})$$


### 📌 3. 분자 배치 표기법 (Numerator Layout vs Denominator Layout)

- 분자 배치 (Numerator Layout - MML 교재 표준):
  미분 $\frac{d\mathbf{f}}{d\mathbf{x}}$ 의 행(Row)을 출력 $\mathbf{f}$ 의 차원($m$), 열(Column)을 입력 $\mathbf{x}$ 의 차원($n$)으로 배치하여 $m \times n$ 행렬로 정의합니다.
- 분모 배치 (Denominator Layout):
  반대로 전치하여 $n \times m$ 행렬로 정의하는 표기법입니다.


### 📌 4. 도함수의 차원 체계 총정리 (Figure 5.6)

| 함수 형태 | 입력 차원 | 출력 차원 | 도함수 / 야코비안 형태 | 차원 크기 |
| :--- | :--- | :--- | :--- | :--- |
| $f: \mathbb{R} \to \mathbb{R}$ | 스칼라 ($1$) | 스칼라 ($1$) | 스칼라 도함수 $\frac{df}{dx}$ | $1 \times 1$ |
| $f: \mathbb{R}^D \to \mathbb{R}$ | 벡터 ($D$) | 스칼라 ($1$) | 그래디언트 행벡터 $\nabla f$ | $1 \times D$ |
| $f: \mathbb{R} \to \mathbb{R}^E$ | 스칼라 ($1$) | 벡터 ($E$) | 열벡터 도함수 $\frac{d\mathbf{f}}{dx}$ | $E \times 1$ |
| $\mathbf{f}: \mathbb{R}^D \to \mathbb{R}^E$ | 벡터 ($D$) | 벡터 ($E$) | 야코비안 행렬 $J$ | $\mathbf{E \times D}$ |


## 2. ⚔️ 야코비안 행렬식과 공간 부피 팽창 배율 (Figure 5.5 & Eq 5.60~5.66)


### 📌 1. 야코비안 행렬식(Jacobian Determinant)의 기하학적 의미

정방 야코비안 행렬($f: \mathbb{R}^n \to \mathbb{R}^n$)에서 야코비안 행렬식의 절댓값 $|\det(J)|$ 은 입력 공간의 단위 부피(Unit Volume)가 함수 변환을 거친 후 몇 배로 팽창/수축하는지를 나타내는 "국소 부피 확대 배율(Magnification Factor)"입니다!

#### 💡 [2차원 평면 단위 격자 변환 사례: Figure 5.5]
- 표준 기저 벡터: $\mathbf{b}_1 = [1, 0]^\top, \; \mathbf{b}_2 = [0, 1]^\top \implies \text{Area} = |\det(I)| = 1$.
- 변환된 기저 벡터: $\mathbf{c}_1 = [-2, 1]^\top, \; \mathbf{c}_2 = [1, 1]^\top \implies \text{Area} = |\det(\begin{bmatrix} -2 & 1 \\\\ 1 & 1 \end{bmatrix})| = |-3| = 3$.
- 야코비안 계산:
  $$y_1 = -2x_1 + x_2, \quad y_2 = x_1 + x_2$$
  $$J = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\\\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} \end{bmatrix} = \begin{bmatrix} -2 & 1 \\\\ 1 & 1 \end{bmatrix} \implies |\det(J)| = |-3| = 3$$
- 해석: 선형 변환에 의해 원래 파란색 단위 정사각형의 면적이 정확히 3배 넓어진 주황색 평행사변형으로 확대되었습니다!
- 비선형 변환의 경우, 각 점 $\mathbf{x}$ 마다 국소적으로 $|\det(J(\mathbf{x}))|$ 배만큼 미세 부피가 팽창합니다.


## 3. ⚔️ 핵심 예제 전수 분석 (Examples 5.9 ~ 5.11)


### 💡 [Example 5.9: 선형 변환 $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ 의 야코비안 유도]
$A \in \mathbb{R}^{M \times N}, \; \mathbf{x} \in \mathbb{R}^N$ 일 때 $\mathbf{f}(\mathbf{x}) = A\mathbf{x} \in \mathbb{R}^M$:
- $i$번째 출력 성분: $f_i(\mathbf{x}) = \sum_{j=1}^N A_{ij} x_j$
- 편미분: $\frac{\partial f_i}{\partial x_j} = A_{ij}$
- 야코비안 행렬 조립:
  $$\frac{d\mathbf{f}}{d\mathbf{x}} = \begin{bmatrix} A_{11} & \dots & A_{1N} \\\\ \vdots & \ddots & \vdots \\\\ A_{M1} & \dots & A_{MN} \end{bmatrix} = \mathbf{A} \in \mathbb{R}^{M \times N} \quad (\text{Eq 5.68})$$
- 핵심 정리: 선형 사상 $A\mathbf{x}$ 의 도함수는 자기 자신의 변환 행렬 $A$ 와 100% 일치합니다!


### 💡 [Example 5.10: 벡터 합성함수의 연쇄 법칙]
$h(t) = (f \circ g)(t)$, $f(\mathbf{x}) = \exp(x_1 x_2^2)$, $\mathbf{x} = g(t) = \begin{bmatrix} t\cos t \\\\ t\sin t \end{bmatrix}$:
- $\frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times 2}, \quad \frac{\partial g}{\partial t} \in \mathbb{R}^{2 \times 1}$
- 연쇄 법칙 적용 (Eq 5.74):
  $$\frac{dh}{dt} = \frac{\partial f}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial t} = \begin{bmatrix} \exp(x_1 x_2^2) x_2^2 & 2\exp(x_1 x_2^2) x_1 x_2 \end{bmatrix} \begin{bmatrix} \cos t - t\sin t \\\\ \sin t + t\cos t \end{bmatrix}$$
  $$\frac{dh}{dt} = \exp(x_1 x_2^2) \left[ x_2^2(\cos t - t\sin t) + 2x_1 x_2(\sin t + t\cos t) \right]$$


### 💡 [Example 5.11: 선형 회귀 최소제곱 손실함수(Least-Squares Loss) 그래디언트 엄밀 유도]
선형 회귀 모델 $\mathbf{y} = \Phi \boldsymbol{\theta}$ ($\boldsymbol{\theta} \in \mathbb{R}^D, \; \Phi \in \mathbb{R}^{N \times D}, \; \mathbf{y} \in \mathbb{R}^N$):
- 오차 벡터: $\mathbf{e}(\boldsymbol{\theta}) := \mathbf{y} - \Phi \boldsymbol{\theta} \in \mathbb{R}^N$
- 손실 함수: $L(\mathbf{e}) := \Vert \mathbf{e} \Vert^2 = \mathbf{e}^\top \mathbf{e} \in \mathbb{R}$
- 목표: $\frac{\partial L}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{1 \times D}$ 도출.

1. 외부 손실함수 미분:
   $$\frac{\partial L}{\partial \mathbf{e}} = 2\mathbf{e}^\top \in \mathbb{R}^{1 \times N} \quad (\text{Eq 5.81})$$
2. 내부 오차함수 야코비안:
   $$\frac{\partial \mathbf{e}}{\partial \boldsymbol{\theta}} = -\Phi \in \mathbb{R}^{N \times D} \quad (\text{Eq 5.82})$$
3. 연쇄 법칙 결합 (Eq 5.83):
   $$\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{\partial L}{\partial \mathbf{e}} \frac{\partial \mathbf{e}}{\partial \boldsymbol{\theta}} = (2\mathbf{e}^\top) (-\Phi) = -2\mathbf{e}^\top \Phi = \mathbf{-2(\mathbf{y}^\top - \boldsymbol{\theta}^\top \Phi^\top)\Phi \in \mathbb{R}^{1 \times D}}$$


## 4. ⚔️ Section 5.4 도입: 행렬에 대한 미분과 3차원 야코비안 텐서 (Figure 5.7)

행렬 $A \in \mathbb{R}^{4 \times 2}$ 를 벡터 $\mathbf{x} \in \mathbb{R}^3$ 로 미분하면 결과는 3차원 텐서 $\frac{dA}{d\mathbf{x}} \in \mathbb{R}^{4 \times 2 \times 3}$ 가 됩니다:

1. 접근법 1 (편미분 슬라이스 결합): 각 편미분 행렬 $\frac{\partial A}{\partial x_1}, \frac{\partial A}{\partial x_2}, \frac{\partial A}{\partial x_3} \in \mathbb{R}^{4 \times 2}$ 를 3번째 깊이 축으로 이어 붙여 $4 \times 2 \times 3$ 텐서 생성.
2. 접근법 2 (평탄화 후 재구성): $A$ 를 8차원 벡터 $\tilde{\mathbf{a}} \in \mathbb{R}^8$ 로 평탄화(Flatten)하여 야코비안 행렬 $\frac{d\tilde{\mathbf{a}}}{d\mathbf{x}} \in \mathbb{R}^{8 \times 3}$ 을 구한 뒤, 다시 $4 \times 2 \times 3$ 텐서로 형태를 변환(Reshape).


## 🧠 5. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 벡터값 함수 ($\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$): 다변수 입력을 받아 다변수 출력을 내보내는 다차원 사상입니다.
- 야코비안 ($J = \frac{d\mathbf{f}}{d\mathbf{x}} \in \mathbb{R}^{m \times n}$): $m$개 출력과 $n$개 입력 간의 모든 1계 편미분을 격자로 모아놓은 행렬입니다.
- 야코비안 행렬식 ($|\det(J)|$): 다차원 공간 변환 시 국소 미세 부피가 확대/축소되는 팽창 배율입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 다층 심층 신경망(Deep Layer)의 전 단계 미분: 레이어별 벡터 입출력 사이의 선형 변화율을 규명하고 연쇄 법칙으로 손실 그래디언트를 전파하기 위해 사용합니다.
- 확률분포의 변수 변환(Change-of-Variables): 잠재 공간(Latent Space)에서 데이터 공간으로의 확률 밀도 보존 및 적분을 계산하기 위해 야코비안 행렬식을 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 분자 배치(Numerator Layout) vs 분모 배치(Denominator Layout):
  - 분자 배치: $\frac{d\mathbf{f}}{d\mathbf{x}} \in \mathbb{R}^{m \times n}$ 으로 연쇄법칙 $\frac{d\mathbf{g}}{d\mathbf{x}} = \frac{d\mathbf{g}}{d\mathbf{f}} \frac{d\mathbf{f}}{d\mathbf{x}}$ 에서 차원 일치가 자연스러움 (본 교재 채택).
  - 분모 배치: 전치가 붙어 차원 관리가 복잡해질 수 있음.


### 4️⃣ [4단계 실전 AI 연결고리]
- 선형 회귀의 정규방정식 (Normal Equation - Ch 9):
  손실함수 그래디언트 $\frac{\partial L}{\partial \boldsymbol{\theta}} = -2(\mathbf{y}^\top - \boldsymbol{\theta}^\top \Phi^\top)\Phi = \mathbf{0}$ 을 전치하여 풀면 전설적인 정규방정식 $\boldsymbol{\theta}^* = (\Phi^\top \Phi)^{-1}\Phi^\top \mathbf{y}$ 가 단번에 유도됩니다!
- 정규화 흐름 생성 모델 (Normalizing Flows & RealNVP):
  복잡한 데이터 확률분포 $p(\mathbf{x}) = p(\mathbf{z}) |\det(J_{f^{-1}})|$ 를 계산하기 위해 삼각 야코비안(Triangular Jacobian) 구조를 설계하여 행렬식을 $O(D)$ 로 초고속 계산합니다.
- VAE 재매개변수화 트릭 (Reparameterization Trick - Ch 6.7):
  확률적 샘플링 $\mathbf{z} = \boldsymbol{\mu} + \sigma \odot \boldsymbol{\epsilon}$ 을 미분 가능한 결정론적 사상으로 변환하여 야코비안 역전파를 수행합니다.
