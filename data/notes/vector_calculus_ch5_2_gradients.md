# 📐 5.2 Partial Differentiation and Gradients (편미분과 그래디언트, 다변수 연쇄법칙과 그래디언트 체킹)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 5.2 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 다변수 세상의 미분: 왜 "편미분(Partial Derivative)"과 "그래디언트(Gradient)"인가?

우리는 앞선 5.1절에서 하나의 변수($x \in \mathbb{R}$)를 미분하는 방법을 다루었습니다.
하지만 머신러닝의 모든 모델(선형 회귀의 수십 개 가중치, 거대 언어 모델 LLM의 수십억 개 매개변수)은 수많은 다변수($\mathbf{x} \in \mathbb{R}^n$)로 구성된 고차원 공간에서 작동합니다.

- 편미분(Partial Derivative): $n$개의 변수 중 오직 관심 있는 변수 $1$개만 미세하게 움직이고, 나머지 모든 변수는 꽁꽁 얼려둔 채(상수 취급) 순간 변화율을 측정하는 분할정복 미분법입니다.
- 그래디언트(Gradient): 각 독립변수 방향으로의 모든 편미분 값들을 일목요연하게 하나로 모아놓은 "다차원 기울기 벡터"입니다.
- 가장 가파른 상승 방향(Steepest Ascent): 그래디언트 벡터는 고차원 곡면 위에서 "함수값이 가장 빠르게 치솟는 방향과 그 경사도"를 가리키며, 그 반대 방향($-\nabla f$)은 딥러닝 최적화의 핵심인 경사하강법의 이동 경로가 됩니다.


## 1. ⚔️ Section 5.2: 편미분의 정의와 그래디언트 행벡터 표기법 (Definition 5.5)


### 📌 1. 편미분의 수학적 정의 (Definition 5.5 & Eq 5.39)

다변수 스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$ ($\mathbf{x} = [x_1, \dots, x_n]^\top \in \mathbb{R}^n$) 에 대해, $i$번째 변수 $x_i$ 방향으로의 편미분은 다음과 같이 정의됩니다:

$$\frac{\partial f}{\partial x_i} := \lim_{h \to 0} \frac{f(x_1, \dots, x_{i-1}, x_i + h, x_{i+1}, \dots, x_n) - f(x_1, \dots, x_n)}{h} \quad (\text{Eq 5.39})$$

- 핵심 연산 직관: $x_i$ 를 제외한 나머지 모든 $x_j$ ($j \neq i$) 변수들을 평범한 숫자(상수)로 취급하고 5.1절의 일변수 스칼라 미분 공식을 그대로 적용합니다.


### 📌 2. 그래디언트(Gradient)의 정의 (Eq 5.40)

모든 편미분 성분들을 가로로 나란히 모아놓은 벡터를 함수 $f$ 의 그래디언트(Gradient) 또는 스칼라 함수의 야코비안(Jacobian) 이라 부릅니다:

$$\nabla_\mathbf{x} f = \text{grad} f = \frac{df}{d\mathbf{x}} := \begin{bmatrix} \frac{\partial f(\mathbf{x})}{\partial x_1} & \frac{\partial f(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial f(\mathbf{x})}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{1 \times n} \quad (\text{Eq 5.40})$$


### 💡 [★ 왜 MML 교재는 그래디언트를 열벡터가 아닌 "행벡터($1 \times n$)"로 정의하는가?]
일반적인 고교 수학이나 일부 문헌에서는 그래디언트를 세로 열벡터($n \times 1$)로 표기하지만, MML 교재와 고급 선형대수학에서는 반드시 가로 행벡터($1 \times n$) 로 엄밀히 정의합니다. 그 이유는 2가지 절대적인 이점 때문입니다:

1. 벡터 함수 야코비안($m \times n$)으로의 자연스러운 일반화:
   함수의 출력이 $m$차원인 $f: \mathbb{R}^n \to \mathbb{R}^m$ 일 때 야코비안 행렬은 $m \times n$ 크기가 됩니다. 스칼라 함수($m=1$)의 그래디언트는 야코비안의 특수한 경우이므로 자연스럽게 $1 \times n$ 행벡터가 됩니다.
2. 다변수 연쇄 법칙(Chain Rule)의 전치 없는 완벽한 행렬곱 일치:
   그래디언트를 행벡터로 정의하면, 합성함수를 미분할 때 전치($^\top$) 기호를 어색하게 붙이지 않고도 $(1 \times n) \times (n \times k) = (1 \times k)$ 로 행렬 곱셈 차원이 물 흐르듯 완벽히 맞아떨어집니다!


### 💡 [Example 5.6 & 5.7: 편미분과 그래디언트 수치 계산 전수 분석]

1. Example 5.6: 연쇄법칙을 이용한 편미분 계산:
   $f(x, y) = (x + 2y^3)^2$ 에 대해:
   - $x$ 에 대한 편미분 ($y$ 를 상수로 취급):
     $$\frac{\partial f}{\partial x} = 2(x + 2y^3) \cdot \frac{\partial}{\partial x}(x + 2y^3) = 2(x + 2y^3) \cdot 1 = 2(x + 2y^3) \quad (\text{Eq 5.41})$$
   - $y$ 에 대한 편미분 ($x$ 를 상수로 취급):
     $$\frac{\partial f}{\partial y} = 2(x + 2y^3) \cdot \frac{\partial}{\partial y}(x + 2y^3) = 2(x + 2y^3) \cdot (6y^2) = 12y^2(x + 2y^3) \quad (\text{Eq 5.42})$$

2. Example 5.7: 그래디언트 행벡터 조합:
   $f(x_1, x_2) = x_1^2 x_2 + x_1 x_2^3 \in \mathbb{R}$ 에 대해:
   $$\frac{\partial f}{\partial x_1} = 2x_1 x_2 + x_2^3, \quad \frac{\partial f}{\partial x_2} = x_1^2 + 3x_1 x_2^2$$
   $$\frac{df}{d\mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 2x_1 x_2 + x_2^3 & x_1^2 + 3x_1 x_2^2 \end{bmatrix} \in \mathbb{R}^{1 \times 2} \quad (\text{Eq 5.45})$$


## 2. ⚔️ Section 5.2.1: Basic Rules of Partial Differentiation (편미분 기본 연산 법칙)

다변수 벡터 $\mathbf{x} \in \mathbb{R}^n$ 에 대해서도 기본 미분 법칙이 그대로 성립합니다 (Eq 5.46~5.48):

1. 곱의 법칙 (Product Rule: Eq 5.46):
   $$\frac{\partial}{\partial \mathbf{x}} [f(\mathbf{x})g(\mathbf{x})] = \frac{\partial f}{\partial \mathbf{x}} g(\mathbf{x}) + f(\mathbf{x}) \frac{\partial g}{\partial \mathbf{x}}$$
2. 합의 법칙 (Sum Rule: Eq 5.47):
   $$\frac{\partial}{\partial \mathbf{x}} [f(\mathbf{x}) + g(\mathbf{x})] = \frac{\partial f}{\partial \mathbf{x}} + \frac{\partial g}{\partial \mathbf{x}}$$
3. 연쇄 법칙 (Chain Rule: Eq 5.48):
   $$\frac{\partial}{\partial \mathbf{x}} (g \circ f)(\mathbf{x}) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial \mathbf{x}}$$
   - 행렬 곱셈의 차원 맞춤 직관: 첫 번째 인수의 분모 $\partial f$ 와 두 번째 인수의 분자 $\partial f$ 가 서로 맞물려 소거되는 것처럼 행렬 곱의 인접 차원이 정확히 일치하여 최종적으로 $\frac{\partial g}{\partial \mathbf{x}}$ 만 남습니다.


## 3. ⚔️ Section 5.2.2: Multivariate Chain Rule (다변수 연쇄 법칙과 행렬 표현)


### 📌 1. 단일 매개변수 $t$ 에 대한 다변수 연쇄 법칙 (Eq 5.49)

함수 $f(x_1, x_2)$ 가 주어지고, $x_1(t), x_2(t)$ 가 매개변수 $t$ 의 함수일 때:

$$\frac{df}{dt} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} \begin{bmatrix} \frac{\partial x_1(t)}{\partial t} \\\\ \frac{\partial x_2(t)}{\partial t} \end{bmatrix} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t} \quad (\text{Eq 5.49})$$

#### 💡 [Example 5.8: 단일 매개변수 연쇄법칙 수치 전개]
$f(x_1, x_2) = x_1^2 + 2x_2$ 이고 $x_1 = \sin t, \; x_2 = \cos t$ 일 때:
$$\frac{df}{dt} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t} = (2x_1)(\cos t) + (2)(-\sin t) = 2\sin t \cos t - 2\sin t = 2\sin t(\cos t - 1)$$


### 📌 2. 다중 매개변수 $(s, t)$ 에 대한 행렬 연쇄 법칙 (Eq 5.51~5.53)

함수 $f(x_1, x_2)$ 에 대해 $x_1(s, t), x_2(s, t)$ 가 두 변수 $s, t$ 의 함수일 때:
- $s$ 에 대한 편미분: $\frac{\partial f}{\partial s} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial s} + \frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial s}$
- $t$ 에 대한 편미분: $\frac{\partial f}{\partial t} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t}$

이를 그래디언트 행벡터와 야코비안 행렬의 단일 곱셈으로 묶으면:

$$\frac{df}{d(s, t)} = \frac{\partial f}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial (s, t)} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} \begin{bmatrix} \frac{\partial x_1}{\partial s} & \frac{\partial x_1}{\partial t} \\\\ \frac{\partial x_2}{\partial s} & \frac{\partial x_2}{\partial t} \end{bmatrix} \in \mathbb{R}^{1 \times 2} \quad (\text{Eq 5.53})$$


## 4. ⚔️ Gradient Checking: 수치 유한차분 검증 (Remark)

머신러닝 코드를 작성할 때 직접 유도한 해석적 그래디언트(Analytic Gradient $df_i$) 구현에 버그가 없는지 검증하기 위해 유한 차분법(Finite Differences)을 사용합니다.

아주 작은 스텝 $h \approx 10^{-4}$ 에 대해 수치적 차분 몫 근사치 $dh_i$ 를 계산한 후, 상대 오차를 측정합니다:

$$dh_i \approx \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_i - h, \dots, x_n)}{2h}$$

$$\text{Relative Error} = \sqrt{\frac{\sum_i (dh_i - df_i)^2}{\sum_i (dh_i + df_i)^2}} < 10^{-6}$$

- 상대 오차가 $10^{-6}$ 미만이면 해석적 역전파/그래디언트 구현이 수학적으로 완벽히 정확하다고 판정합니다!


## 🧠 5. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 편미분 ($\frac{\partial f}{\partial x_i}$): $n$차원 다변수 함수에서 특정 변수 $x_i$ 를 제외한 나머지 모든 변수를 상수로 고정하고 계산한 순간 변화율입니다.
- 그래디언트 ($\nabla f = \frac{df}{d\mathbf{x}} \in \mathbb{R}^{1 \times n}$): 모든 독립변수 방향으로의 편미분 값들을 모아놓은 행벡터이자, 함수값이 가장 가파르게 증가하는 방향(Steepest Ascent)을 가리키는 벡터입니다.
- 다변수 연쇄 법칙 ($\frac{df}{d(s, t)} = \frac{\partial f}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial (s, t)}$): 합성함수의 그래디언트를 내부/외부 함수의 야코비안 행렬 곱으로 연결하는 규칙입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 고차원 손실 곡면의 나침반: 수억 개의 가중치 공간에서 손실함수(Loss)를 가장 빠르게 줄일 수 있는 방향($-\nabla L$)을 단번에 알아내기 위해 사용합니다.
- 계산의 분할정복: 복잡한 다변수 함수를 $n$개의 독립적인 1차원 스칼라 미분 문제로 쪼개어 단순하게 계산할 수 있습니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 해석적 그래디언트(Analytic Gradient) vs 수치 유한차분(Numerical Gradient):
  - 해석적 그래디언트 (역전파): 연쇄법칙 수식으로 단 한 번에 계산하므로 $O(1)$ 패스로 초고속 계산 가능 (실제 모델 학습에 사용).
  - 수치 유한차분 (Gradient Checking): 모든 변수마다 $f(x+h)$ 를 일일이 호출해야 하므로 $O(n)$ 계산 비용이 발생하지만, 코드의 버그를 100% 잡아내는 검증용으로 필수적입니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- PyTorch Autograd / Backward 패스:
  `loss.backward()` 를 호출하면 신경망의 모든 매개변수 $w_i$ 에 대해 $\frac{\partial \text{Loss}}{\partial w_i}$ 를 다변수 연쇄법칙으로 역방향 전파하여 `w.grad` 에 행벡터 형태로 누적합니다.
- 경사하강법 가중치 갱신 (SGD & Adam):
  $$w \leftarrow w - \eta \nabla_w \text{Loss}$$
  (그래디언트의 반대 방향으로 학습률 $\eta$ 만큼 이동하여 손실 최소화 달성).
- 물리 정보 기반 신경망 (PINN - Physics-Informed Neural Networks):
  미분방정식(PDE)의 편미분 항($\frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial x^2}$)을 신경망 출력의 그래디언트로 직접 계산하여 물리 법칙 손실함수에 결합합니다.
