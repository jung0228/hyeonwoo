# 📐 2.1 Systems of Linear Equations (선형방정식계)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.1 원문 완전 대조 노트

---

## 1. 🌐 Chapter 2 Intro: 벡터(Vector)의 4가지 유형

수학에서 수학적 대상(Objects)을 정의하고 이들을 조작하는 규칙의 집합을 Algebra(대수학)라 하며, Linear Algebra(선형대수학)는 벡터와 벡터를 조작하는 대수 규칙을 연구하는 학문이다.

본 교재에서는 일반적인 벡터를 굵은 글씨 $\mathbf{x}, \mathbf{y}$ 로 표기하며, "더하기(Addition)가 가능하고 스칼라 곱(Scalar Multiplication)을 했을 때 동일한 종류의 대상이 되는 모든 오브젝트"를 벡터로 정의한다.

### 📌 벡터의 4가지 예시 (Figure 2.1)
1. Geometric Vectors (기하학적 벡터):
   - 방향과 크기를 가지는 화살표/유향 선분 ($\vec{x}, \vec{y}$).
   - $\vec{x} + \vec{y} = \vec{z}$ 이며 스칼라 배 $\lambda \vec{x}$ 역시 동일한 유향 선분임.
2. Polynomials (다항식):
   - 두 다항식을 더해도 다항식이며, 스칼라 $\lambda \in \mathbb{R}$를 곱해도 다항식임. (기하학적 "그림"과 달리 추상적인 개념이지만 수학적 벡터 공리를 충족함)
3. Audio Signals (오디오 신호):
   - 일련의 숫자 시퀀스로 표현되는 오디오 신호는 더하거나 스칼라배를 해도 새로운 오디오 신호가 되므로 벡터임.
4. Elements of $\mathbb{R}^n$ ($n$차원 실수 튜플):
   - 본 교재에서 주로 다루는 추상적 대상. 예: $\mathbf{a} = \begin{bmatrix} 1 \\\\ 2 \\\\ 3 \end{bmatrix} \in \mathbb{R}^3$.
   - 컴퓨터 프로그램의 실수 배열(Array of real numbers) 연산과 1:1로 정확히 대응함.

---

## 2. ⚔️ Section 2.1: Systems of Linear Equations (선형방정식계)

### 📌 Example 2.1 (자원 배분과 최적 생산 계획 모델)

회사에서 자원 $R_1, \dots, R_m$을 사용하여 제품 $N_1, \dots, N_n$을 생산한다.
- 제품 $N_j$ 1단위를 생산하는 데 자원 $R_i$가 $a_{ij}$ 단위만큼 필요함.
- 사용 가능한 총 자원이 $b_i$ 단위일 때, 자원을 남김없이 소비하는 최적 생산량 $x_1, \dots, x_n$을 구하는 계획.

#### 📐 선형방정식계 일반형 (Equation 2.3)
자원 $R_i$에 대한 소비 총량 수식:
$$a_{i1} x_1 + a_{i2} x_2 + \dots + a_{in} x_n = b_i \quad (i = 1, \dots, m)$$

전체 연립 1차 방정식계:
$$\begin{aligned}
a_{11} x_1 + a_{12} x_2 + \dots + a_{1n} x_n &= b_1 \\\\
a_{21} x_1 + a_{22} x_2 + \dots + a_{2n} x_n &= b_2 \\\\
&\vdots \\\\
a_{m1} x_1 + a_{m2} x_2 + \dots + a_{mn} x_n &= b_m
\end{aligned}$$
여기서 $x_1, \dots, x_n$은 미지수(Unknowns)이며, (2.3)을 만족하는 모든 $n$-튜플 $(x_1, \dots, x_n) \in \mathbb{R}^n$이 해(Solution)가 된다.

---

### 📌 Example 2.2 (해의 3가지 가능성 - 해 없음, 유일해, 무수히 많은 해)

실수 선형방정식계의 해는 [1] 해가 없음, [2] 정확히 1개의 유일해, [3] 무수히 많은 해 3가지 경우만 존재한다.

#### 1️⃣ [Case 1: No Solution (해 없음 - Equation 2.4)]
$$\begin{aligned}
x_1 + x_2 + x_3 &= 3 \quad (1) \\\\
x_1 - x_2 + 2x_3 &= 2 \quad (2) \\\\
2x_1 + 3x_3 &= 1 \quad (3)
\end{aligned}$$
- 풀이: (1)식과 (2)식을 더하면 $2x_1 + 3x_3 = 5$가 된다. 하지만 이는 (3)번 식의 $2x_1 + 3x_3 = 1$ 과 모순($5 = 1$)되므로 해가 존재하지 않는다.

#### 2️⃣ [Case 2: Unique Solution (유일해 - Equation 2.5)]
$$\begin{aligned}
x_1 + x_2 + x_3 &= 3 \quad (1) \\\\
x_1 - x_2 + 2x_3 &= 2 \quad (2) \\\\
x_2 + x_3 &= 2 \quad (3)
\end{aligned}$$
- 풀이: (1)식에서 (3)식을 빼면 $x_1 = 1$. (1)+(2)에서 $2x_1 + 3x_3 = 5 \implies x_3 = 1$. (3)식에 대입하면 $x_2 = 1$.
- 결론: 오직 $(1, 1, 1)$ 만이 유일한 해(Unique Solution)가 된다.

#### 3️⃣ [Case 3: Infinite Solutions (무수히 많은 해 - Equation 2.6 & 2.7)]
$$\begin{aligned}
x_1 + x_2 + x_3 &= 3 \quad (1) \\\\
x_1 - x_2 + 2x_3 &= 2 \quad (2) \\\\
2x_1 + 3x_3 &= 5 \quad (3)
\end{aligned}$$
- 풀이: (1)+(2)=(3) 이므로 (3)번 식은 중복(Redundancy)되어 생략 가능함. (1)과 (2)로부터 $2x_1 = 5 - 3x_3$, $2x_2 = 1 + x_3$를 얻음.
- $x_3 = a \in \mathbb{R}$를 자유 변수(Free Variable)로 정의하면, 해집합은 다음과 같이 무수히 많은 해를 가짐:
  $$\left( \frac{5}{2} - \frac{3}{2}a, \; \frac{1}{2} + \frac{1}{2}a, \; a \right), \quad a \in \mathbb{R}$$

---

### 📌 Remark: 기하학적 해석 (Geometric Interpretation)

- 2차원 평면 ($\mathbb{R}^2$): 각 선형방정식은 $x_1 x_2$-평면 상의 직선(Line)을 정의함. 해집합은 이 직선들의 교점(Intersection)임.
  - 직선들이 평행하면 해가 없음 (Empty).
  - 한 점에서 만나면 유일해 (Point).
  - 같은 직선을 나타내면 무수히 많은 해 (Line).
  - Equation (2.8): $4x_1 + 4x_2 = 5$, $2x_1 - 4x_2 = 1$ 의 해집합은 점 $(x_1, x_2) = (1, \frac{1}{4})$.
- 3차원 공간 ($\mathbb{R}^3$): 각 방정식은 3차원 공간 상의 평면(Plane)을 정의하며, 이 평면들의 교집합이 해집합이 됨 (평면, 직선, 점, 또는 해 없음).

---

## 3. 📐 행렬 벡터 표기법 (Matrix-Vector Notation: Eq 2.9 & 2.10)

방정식계 (2.3)을 계수 벡터들의 선형 결합 형태로 컴팩트하게 표기:

$$x_1 \begin{bmatrix} a_{11} \\\\ \vdots \\\\ a_{m1} \end{bmatrix} + x_2 \begin{bmatrix} a_{12} \\\\ \vdots \\\\ a_{m2} \end{bmatrix} + \dots + x_n \begin{bmatrix} a_{1n} \\\\ \vdots \\\\ a_{mn} \end{bmatrix} = \begin{bmatrix} b_1 \\\\ \vdots \\\\ b_m \end{bmatrix}$$

이를 행렬-벡터 곱셈 형태 $Ax = b$ 로 압축 (Equation 2.10):

$$\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1n} \\\\ a_{21} & a_{22} & \dots & a_{2n} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ a_{m1} & a_{m2} & \dots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \\\\ \vdots \\\\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\\\ b_2 \\\\ \vdots \\\\ b_m \end{bmatrix}$$
