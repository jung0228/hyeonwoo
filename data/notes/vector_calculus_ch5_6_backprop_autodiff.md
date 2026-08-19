# 📐 5.6 Backpropagation and Automatic Differentiation (역전파 알고리즘, 자동 미분과 계산 그래프)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 5.6 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 딥러닝 혁명의 절대적 심장: 왜 "역전파(Backpropagation)"와 "자동 미분(Autodiff)"인가?

우리는 앞선 5.2~5.5절에서 연쇄 법칙과 행렬 미분 항등식을 배웠습니다.
하지만 수천만~수천억 개의 파라미터를 가진 거대 심층 신경망(Deep Neural Networks, LLM, Transformer)을 학습시킬 때, 종이 위에 수식을 하나하나 손으로 미분하여 구현하는 것은 불가능합니다.

- 수식 폭발(Expression Swell)의 한계: 합성함수 $f(x) = \sqrt{x^2 + \exp(x^2)} + \cos(x^2 + \exp(x^2))$ 를 기호 미분(Symbolic differentiation)으로 그대로 풀면 수식이 기형적으로 길어지고 중복 계산이 폭발합니다.
- 역전파 알고리즘 (Backpropagation, Rumelhart et al., 1986): 연쇄 법칙(Chain Rule)을 뒤에서 앞으로 역방향 전파하면서 중간 계산 결과를 완벽히 재사용(동적계획법 캐싱)하여, 순방향 함수 계산과 동일한 $O(\text{Ops})$ 비용으로 수억 개 매개변수의 그래디언트를 단번에 계산합니다.
- 자동 미분 (Automatic Differentiation / Autograd): PyTorch, JAX, TensorFlow의 근간 엔진으로, 컴퓨터 프로그램을 기본 연산자 단위의 계산 그래프(Computational Graph)로 분해하여 기계 정밀도(Machine Precision) 수준으로 정확한 미분을 전자동 수행합니다.


## 1. ⚔️ Section 5.6.1: Gradients in a Deep Network (심층 신경망의 역전파 메커니즘)


### 📌 1. 심층 다층 신경망의 순방향 전파 (Forward Pass: Figure 5.8 & Eq 5.111~5.114)

입력 데이터 $\mathbf{x}$ 로부터 최종 예측값 $\mathbf{y}$ 를 얻는 과정은 $K$개 레이어의 다단계 함수 합성입니다:

$$\mathbf{y} = (f_K \circ f_{K-1} \circ \dots \circ f_1)(\mathbf{x}) = f_K(f_{K-1}(\dots(f_1(\mathbf{x}))\dots)) \quad (\text{Eq 5.111})$$

- $i$번째 레이어의 출력 $\mathbf{f}_i$:
  $$\mathbf{f}_0 := \mathbf{x}, \quad \mathbf{f}_i := \sigma_i(A_{i-1}\mathbf{f}_{i-1} + \mathbf{b}_{i-1}) \quad (i = 1, \dots, K) \quad (\text{Eq 5.112~5.113})$$
  (여기서 $A_{i-1}$ 은 가중치 행렬, $\mathbf{b}_{i-1}$ 은 편향 벡터, $\sigma_i$ 는 ReLU, Sigmoid, tanh 등의 활성화 함수입니다.)
- 손실 함수 (Loss Function):
  $$L(\boldsymbol{\theta}) = \Vert \mathbf{y} - \mathbf{f}_K(\boldsymbol{\theta}, \mathbf{x}) \Vert^2 \quad (\text{단, } \boldsymbol{\theta} = \{A_0, \mathbf{b}_0, \dots, A_{K-1}, \mathbf{b}_{K-1}\})$$


### 📌 2. 연쇄 법칙을 통한 레이어별 역방향 전파 (Backward Pass: Figure 5.9 & Eq 5.115~5.118)

손실 $L$ 을 각 레이어의 매개변수 $\boldsymbol{\theta}_i = \{A_i, \mathbf{b}_i\}$ 로 미분하기 위해 다변수 연쇄 법칙을 적용합니다:

$$\frac{\partial L}{\partial \boldsymbol{\theta}_{K-1}} = \frac{\partial L}{\partial \mathbf{f}_K} \frac{\partial \mathbf{f}_K}{\partial \boldsymbol{\theta}_{K-1}} \quad (\text{Eq 5.115})$$

$$\frac{\partial L}{\partial \boldsymbol{\theta}_{K-2}} = \frac{\partial L}{\partial \mathbf{f}_K} \frac{\partial \mathbf{f}_K}{\partial \mathbf{f}_{K-1}} \frac{\partial \mathbf{f}_{K-1}}{\partial \boldsymbol{\theta}_{K-2}} \quad (\text{Eq 5.116})$$

$$\frac{\partial L}{\partial \boldsymbol{\theta}_i} = \underbrace{\frac{\partial L}{\partial \mathbf{f}_K} \frac{\partial \mathbf{f}_K}{\partial \mathbf{f}_{K-1}} \dots \frac{\partial \mathbf{f}_{i+2}}{\partial \mathbf{f}_{i+1}}}_{\text{앞선 레이어에서 이미 계산된 그래디언트 } \frac{\partial L}{\partial \mathbf{f}_{i+1}}} \cdot \frac{\partial \mathbf{f}_{i+1}}{\partial \boldsymbol{\theta}_i} \quad (\text{Eq 5.118})$$

- 동적계획법적 재사용(Caching)의 기적:
  상위 레이어에서 하위 레이어로 내려올 때, 이미 계산된 출력 그래디언트 $\frac{\partial L}{\partial \mathbf{f}_{i+1}}$ 를 메모리에 보관해두면 하위 레이어의 파라미터 그래디언트를 구할 때 $\frac{\partial \mathbf{f}_{i+1}}{\partial \boldsymbol{\theta}_i}$ 만 한 번 곱해주면 끝납니다!


## 2. ⚔️ Section 5.6.2: Automatic Differentiation (자동 미분과 계산 그래프)


### 📌 1. 미분 3대 방식 전면 비교

| 미분 방식 | 작동 원리 | 장점 | 치명적 단점 |
| :--- | :--- | :--- | :--- |
| 수치 미분 (Numerical / Finite Diff) | $\lim \frac{f(x+h)-f(x)}{h}$ 수치 대입 | 구현이 극도로 단순함 | 반올림/절단 오차 발생, $N$개 변수마다 $O(N)$ 재실행 |
| 기호 미분 (Symbolic / SymPy) | 사람이 풀듯 대수 공식으로 전개 | 100% 수학적 정확성 | 수식이 기하급수적으로 팽창(수식 폭발 Expression Swell) |
| 자동 미분 (Automatic Differentiation) | 기본 연산 단위 계산 그래프에서 연쇄법칙 수치 전파 | 기계 정밀도 수준 정확, $O(\text{Forward})$ 비용 | 계산 그래프 추적을 위한 추가 메모리 필요 |


### 📌 2. 순방향 모드(Forward Mode) vs 역방향 모드(Reverse Mode / Backprop) 결합 순서 비교

데이터 흐름 그래프 $x \to a \to b \to y$ 에서 $\frac{dy}{dx} = \frac{dy}{db} \frac{db}{da} \frac{da}{dx}$ 일 때 행렬곱 결합 순서의 차이 (Figure 5.10 & Eq 5.120~5.121):

1. 순방향 모드 (Forward Mode AD: Eq 5.121):
   $$\frac{dy}{dx} = \frac{dy}{db} \left( \frac{db}{da} \frac{da}{dx} \right)$$
   - 입력에서 출력 방향(왼쪽에서 오른쪽)으로 그래디언트를 전파합니다.
   - 입력 변수 $N$ 개가 작고 출력 변수 $M$ 개가 클 때 ($N \ll M$) 유리합니다.
2. 역방향 모드 (Reverse Mode AD / Backpropagation: Eq 5.120):
   $$\frac{dy}{dx} = \left( \frac{dy}{db} \frac{db}{da} \right) \frac{da}{dx}$$
   - 출력에서 입력 방향(오른쪽에서 왼쪽)으로 그래디언트를 거슬러 올라가며 전파합니다.
   - 입력 변수 $N$ 개(수억 개 가중치)가 거대하고 최종 출력 $M=1$ (스칼라 손실 Loss) 일 때 ($N \gg M$) 압도적인 계산 효율을 달성합니다! 단 1번의 역방향 패스($O(1)$)로 모든 가중치의 그래디언트를 동시에 추출합니다.


### 💡 [Example 5.14: 계산 그래프(Computation Graph)와 역방향 자동 미분 전수 수치 추적]

함수 $f(x) = \sqrt{x^2 + \exp(x^2)} + \cos(x^2 + \exp(x^2))$ 의 계산 그래프 구축 (Figure 5.11):

1. 순방향 중간 변수 생성 (Forward Pass: Eq 5.123~5.128):
   - $a = x^2$
   - $b = \exp(a)$
   - $c = a + b$
   - $d = \sqrt{c}$
   - $e = \cos(c)$
   - $f = d + e$

2. 기본 연산자별 국소 도함수 (Local Derivatives: Eq 5.129~5.134):
   - $\frac{\partial a}{\partial x} = 2x, \quad \frac{\partial b}{\partial a} = \exp(a), \quad \frac{\partial c}{\partial a} = 1, \quad \frac{\partial c}{\partial b} = 1$
   - $\frac{\partial d}{\partial c} = \frac{1}{2\sqrt{c}}, \quad \frac{\partial e}{\partial c} = -\sin(c), \quad \frac{\partial f}{\partial d} = 1, \quad \frac{\partial f}{\partial e} = 1$

3. 역방향 그래디언트 전파 (Backward Pass: Eq 5.135~5.142):
   - $\frac{\partial f}{\partial f} = 1$
   - $\frac{\partial f}{\partial d} = 1, \quad \frac{\partial f}{\partial e} = 1$
   - $\frac{\partial f}{\partial c} = \frac{\partial f}{\partial d}\frac{\partial d}{\partial c} + \frac{\partial f}{\partial e}\frac{\partial e}{\partial c} = \frac{1}{2\sqrt{c}} - \sin(c)$
   - $\frac{\partial f}{\partial b} = \frac{\partial f}{\partial c}\frac{\partial c}{\partial b} = \frac{\partial f}{\partial c} \cdot 1$
   - $\frac{\partial f}{\partial a} = \frac{\partial f}{\partial b}\frac{\partial b}{\partial a} + \frac{\partial f}{\partial c}\frac{\partial c}{\partial a} = \frac{\partial f}{\partial b}\exp(a) + \frac{\partial f}{\partial c} \cdot 1$
   - $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} = \frac{\partial f}{\partial a} \cdot (2x)$

- 결과: 복잡하고 긴 미분 수식(Eq 5.110)을 직접 풀지 않고도, 각 마디의 덧셈과 곱셈만으로 정확한 $\frac{df}{dx}$ 가 완벽히 계산됩니다!


### 📌 3. 일반 계산 그래프의 역방향 자동 미분 총괄 공식 (Eq 5.143~5.145)

입력 노드 $x_1, \dots, x_d$, 중간 노드 $x_{d+1}, \dots, x_{D-1}$, 최종 출력 노드 $x_D = f$:
1. 순방향 전파 (Forward: Eq 5.143):
   $$x_i = g_i(x_{\text{Pa}(x_i)}) \quad (i = d+1, \dots, D)$$
   ($\text{Pa}(x_i)$ 는 노드 $x_i$ 의 부모 노드 집합).
2. 역방향 전파 (Backward: Eq 5.144~5.145):
   $$\frac{\partial f}{\partial x_D} = 1$$
   $$\frac{\partial f}{\partial x_i} = \sum_{x_j : x_i \in \text{Pa}(x_j)} \frac{\partial f}{\partial x_j} \frac{\partial x_j}{\partial x_i} = \sum_{x_j : x_i \in \text{Pa}(x_j)} \frac{\partial f}{\partial x_j} \frac{\partial g_j}{\partial x_i}$$
   (노드 $x_i$ 로부터 뻗어나간 모든 자식 노드 $x_j$ 경로들의 역방향 그래디언트 합산).


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 역전파 (Backpropagation): 다층 신경망에서 출력 손실의 오차를 출력층부터 입력층까지 역방향으로 전파하며 각 가중치의 그래디언트를 구하는 동적계획법 기반 알고리즘입니다.
- 자동 미분 (Automatic Differentiation): 복잡한 컴퓨터 프로그램을 기본 연산자의 계산 그래프로 분해하여 연쇄 법칙을 컴퓨터 상에서 수치적으로 정확하게 평가하는 기법입니다.
- 역방향 모드 자동 미분 (Reverse Mode AD): 출력이 1개이고 입력이 수억 개인 함수에서 출력 측부터 거슬러 올라가며 단 1회의 역방향 탐색으로 모든 입력의 그래디언트를 구하는 최적 모드입니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 수식 팽창 없이 수억 개 파라미터 동시 학습: 기호 미분의 수식 폭발과 수치 미분의 $O(N)$ 연산 비효율을 극복하고, 순방향 연산 비용의 약 2~3배 수준만으로 모든 가중치의 정확한 기울기를 얻기 위해 사용합니다.
- 복잡한 제어 흐름(if, for)이 포함된 딥러닝 모델의 엔드투엔드 미분: PyTorch Autograd 엔진이 실행 시점에 동적 계산 그래프(Dynamic Computational Graph)를 추적하여 손쉽게 그래디언트를 산출합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- Forward-mode AD vs Reverse-mode AD (Backprop):
  - Forward-mode: 입력 차원 $N$ 번만큼 순방향을 반복해야 하므로 $N \gg 1$ 인 딥러닝에는 비효율적이지만, 중간 텐서를 메모리에 저장할 필요가 없어 메모리 사용량이 $O(1)$ 입니다.
  - Reverse-mode: 역방향 계산을 위해 순방향 패스에서 생성된 모든 중간 활성화 텐서(Activation tensors)를 메모리에 보관해야 하므로 메모리 사용량이 레이어 깊이에 비례($O(K)$)하여 증가합니다 (이를 해결하기 위해 Activation Checkpointing 기법 사용).


### 4️⃣ [4단계 실전 AI 연결고리]
- PyTorch `loss.backward()` 엔진:
  텐서의 `requires_grad=True` 속성을 추적하여 동적 DAG(Directed Acyclic Graph)를 빌드하고, `backward()` 호출 시 C++ 엔진이 Eq 5.145 공식에 따라 리버스 모드 자동 미분을 실행합니다.
- 메모리 절약을 위한 그래디언트 체크포인팅 (Gradient Checkpointing / Rematerialization):
  거대 언어 모델(LLM) 훈련 시 역방향 전파에 필요한 모든 중간 활성화 값을 저장하지 않고, 일부 체크포인트 노드만 저장한 뒤 역전파 시점에 순방향을 재계산(Rematerialize)하여 GPU 메모리를 70% 이상 절약합니다.
- JAX `jax.grad` & `jax.vjp`:
  벡터-야코비안 곱(Vector-Jacobian Product, VJP)을 기반으로 역방향 자동 미분을 함수형 패러다임으로 컴파일하여 TPU/GPU에서 초고속 가속 훈련을 수행합니다.
