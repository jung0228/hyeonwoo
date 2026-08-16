# 📐 05. 추상적 선형변환의 이산 행렬 표현 (Matrix Representation of Linear Mapping)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- 선형변환 (Linear Mapping $T$): 덧셈과 스칼라배 보존 조건($T(u+v) = T(u)+T(v)$, $T(cu) = cT(u)$)을 만족하는 모든 함수 연산자.
- 행렬 표현 (Matrix Representation $M$): 미분($\frac{d}{dx}$)이나 적분($\int$) 같은 추상적 선형 연산자를 컴퓨터가 연산할 수 있는 이산 행렬(Discrete Matrix) 곱셈 형태로 재좌표화하는 기술.


## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.5)

### [Problem 5] Ex 2.5 - 미분 연산자 $T = \frac{d}{dx}$ 의 $3 \times 3$ 행렬 표현 백지 유도
- 문제: 2차 이하 다항식 공간 $P_2 = \{ a_0 + a_1 x + a_2 x^2 \}$ 의 기저 $\mathcal{B} = \{1, x, x^2\}$ 에 대해, 미분 변환 $T(p(x)) = \frac{d}{dx} p(x)$ 의 행렬 표현 $M \in \mathbb{R}^{3 \times 3}$ 을 구하시오.

- 수식 유도 3단계:
  1. 기저 원소별 미분 연산 수행:
     - $T(1) = \frac{d}{dx}(1) = 0$
     - $T(x) = \frac{d}{dx}(x) = 1$
     - $T(x^2) = \frac{d}{dx}(x^2) = 2x$

  2. 결과 다항식을 기저 $\{1, x, x^2\}$ 의 선형결합 계수 벡터로 표현:
     - $T(1) = 0 = \mathbf{0} \cdot 1 + \mathbf{0} \cdot x + \mathbf{0} \cdot x^2 \implies \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \end{bmatrix}$ (1번째 열)
     - $T(x) = 1 = \mathbf{1} \cdot 1 + \mathbf{0} \cdot x + \mathbf{0} \cdot x^2 \implies \begin{bmatrix} 1 \\\\ 0 \\\\ 0 \end{bmatrix}$ (2번째 열)
     - $T(x^2) = 2x = \mathbf{0} \cdot 1 + \mathbf{2} \cdot x + \mathbf{0} \cdot x^2 \implies \begin{bmatrix} 0 \\\\ 2 \\\\ 0 \end{bmatrix}$ (3번째 열)

  3. 계수 열벡터들을 나열하여 행렬 $M$ 완성:
     $$M = \begin{bmatrix} 0 & 1 & 0 \\\\ 0 & 0 & 2 \\\\ 0 & 0 & 0 \end{bmatrix}$$


## 🔍 3. 비판적 맹점 & 실전 AI 연결

### 1) 미분 연산의 정보 손실 맹점 (Kernel 소실)
- 행렬 $M$의 첫 번째 열이 전부 $0$이라는 뜻은, 상수항($1$) 정보가 미분 연산을 통과하면서 0으로 매핑되어 완전히 사라졌다(Kernel 공간으로 손실됨)는 수학적 맹점을 의미함.
- 따라서 미분 연산자 $M$은 역행렬이 존재하지 않는 가역성 파탄 행렬임 ($\det(M) = 0$).

### 2) 실전 AI / 딥러닝 연결 (Neural ODE & PINN)
- Neural ODE (미분방정식 신경망): 미분방정식을 컴퓨터로 풀 때, 연속적 미분 연산자를 가우시안/이산 행렬 $M$으로 변환하여 역전파(Backpropagation)를 연산함.
- Physics-Informed Neural Network (PINN): 물리 법칙 미분 기호를 행렬 곱셈 형태로 이산화하여 학습함.
