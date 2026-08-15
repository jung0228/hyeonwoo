# 📐 선형대수학 MML 전수 연습문제 풀이집 (Linear Algebra Problem Set)

> POSTECH 대학원 준비 4단계 표준 체계 100% 준수
> 
> 본 문서에는 MML(Mathematics for Machine Learning) 교재 Chapter 2 (Exercises 2.1 ~ 2.20)에 수록된 전수 연습문제의 단계별 풀이, 증명, 수치적 맹점 및 실전 AI 연결고리가 수록되어 있습니다.

---

## 📝 Part 1. 군, 유체 및 행렬 연산 (Exercises 2.1 ~ 2.4)

### [Problem 2.1] 아벨군 (Abelian Group) 정의 증명 및 연립방정식 풀이

#### 1. 문제 정의
$\mathbb{R} \setminus \{-1\}$ 상에서 정의된 연산 $a \star b := ab + a + b$ 에 대해 다음을 증명하고 방정식을 푸시오.
- a. $(\mathbb{R} \setminus \{-1\}, \star)$ 이 아벨군(Abelian Group)임을 증명하시오.
- b. 방정식 $3 \star x \star x = 15$ 의 해를 구하시오.

#### 2. 상세 증명 및 풀이 단계
1. 아벨군 4대 조건 증명:
   - 닫힘성 (Closure): $a, b \neq -1$ 일 때 $a \star b = ab + a + b = (a+1)(b+1) - 1$. $a+1 \neq 0, b+1 \neq 0$ 이므로 $a \star b \neq -1$ 성립.
   - 결합법칙 (Associativity): $(a \star b) \star c = (ab+a+b) \star c = (a+1)(b+1)(c+1) - 1 = a \star (b \star c)$ 성립.
   - 항등원 (Identity Element): $a \star e = a \implies ae + a + e = a \implies e(a+1) = 0 \implies e = 0 \in \mathbb{R} \setminus \{-1\}$.
   - 역원 (Inverse Element): $a \star a^{-1} = 0 \implies a a^{-1} + a + a^{-1} = 0 \implies a^{-1} = \frac{-a}{a+1} \in \mathbb{R} \setminus \{-1\}$.
   - 교환법칙 (Commutativity): $a \star b = ab + a + b = ba + b + a = b \star a$. 따라서 아벨군입니다.

2. 방정식 풀이 ($3 \star x \star x = 15$):
   - $3 \star x = 3x + 3 + x = 4x + 3$.
   - $(4x+3) \star x = (4x+3)x + (4x+3) + x = 4x^2 + 8x + 3 = 15$.
   - $4x^2 + 8x - 12 = 0 \implies x^2 + 2x - 3 = 0 \implies (x+3)(x-1) = 0$.
   - $x = 1$ 또는 $x = -3$ (두 값 모두 $\mathbb{R} \setminus \{-1\}$ 에 속하므로 유효한 해입니다).

---

### [Problem 2.2] 잉여류(Congruence Classes) $\mathbb{Z}_n$ 과 아벨군 및 소수(Prime) 조건 증명

#### 1. 문제 정의
n-합동류 집합 $\mathbb{Z}_n = \{0, 1, \dots, n-1\}$ 상의 덧셈 $\oplus$ 과 곱셈 $\otimes$ 연산 구조를 검증하시오.

#### 2. 상세 증명 단계
- a. $(\mathbb{Z}_n, \oplus)$ 은 덧셈 항등원 $0$, $a$의 역원 $n-a$를 가지며 교환법칙이 성립하는 아벨군입니다.
- b. $\mathbb{Z}_5 \setminus \{0\}$ 곱셈표 도출:
  $$\begin{array}{c|cccc} \otimes & 1 & 2 & 3 & 4 \\ \hline 1 & 1 & 2 & 3 & 4 \\ 2 & 2 & 4 & 1 & 3 \\ 3 & 3 & 1 & 4 & 2 \\ 4 & 4 & 3 & 2 & 1 \end{array}$$
  - 곱셈 항등원은 $1$이며, 역원은 $1^{-1}=1, 2^{-1}=3, 3^{-1}=2, 4^{-1}=4$ 로 존재하여 아벨군을 이룹니다.
- c. $(\mathbb{Z}_8 \setminus \{0\}, \otimes)$ 군 불성립 증명:
  - $2 \otimes 4 = 8 \equiv 0 \pmod 8$. $0 \notin \mathbb{Z}_8 \setminus \{0\}$ 이므로 닫힘성이 파탄 나 군이 아닙니다 (영인자 Zero Divisor 존재).
- d. 베주 정리(Bézout's identity)에 의한 소수 조건 증명:
  - $\mathbb{Z}_n \setminus \{0\}$ 의 모든 원소가 곱셈 역원을 가지려면 임의의 $a \in \{1, \dots, n-1\}$ 에 대해 $\gcd(a, n) = 1$ 이어야 합니다. 이는 오직 $n$이 소수(Prime)일 때만 성립합니다.

---

### [Problem 2.3] 상삼각 행렬 집합 $G$의 군(Group) 판별

#### 1. 문제 정의
$G = \left\{ \begin{bmatrix} 1 & x & z \\ 0 & 1 & y \\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 3} \mid x, y, z \in \mathbb{R} \right\}$ 이 표준 행렬 곱에 대해 아벨군인지 판별하시오.

#### 2. 상세 증명 단계
- 닫힘성: $A(x_1,y_1,z_1) B(x_2,y_2,z_2) = \begin{bmatrix} 1 & x_1+x_2 & z_1+z_2+x_1 y_2 \\ 0 & 1 & y_1+y_2 \\ 0 & 0 & 1 \end{bmatrix} \in G$.
- 항등원: $x=y=z=0$ 일 때 단위행렬 $I_3 \in G$.
- 역원: $A(x,y,z)^{-1} = \begin{bmatrix} 1 & -x & xy-z \\ 0 & 1 & -y \\ 0 & 0 & 1 \end{bmatrix} \in G$.
- 교환법칙 불성립: $x_1 y_2 \neq x_2 y_1$ 인 경우가 존재하므로 군은 맞지만 아벨군은 아닙니다 (비아벨군 Non-Abelian Group / 하이젠베르크 군).

---

### [Problem 2.4] 행렬 곱셈 계산 (Matrix Products)

#### 1. 문제 계산 도출
- a. $(3 \times 2) \times (3 \times 3)$ ➡️ 불가능 (인접 차원 $2 \neq 3$ 불일치).
- b. $(3 \times 3) \times (3 \times 3)$ ➡️ 가능:
  $$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 4 & 3 & 5 \\ 10 & 9 & 11 \\ 16 & 15 & 17 \end{bmatrix}$$
- c. $(3 \times 3) \times (3 \times 3)$ ➡️ 가능:
  $$\begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} = \begin{bmatrix} 5 & 7 & 9 \\ 11 & 13 & 15 \\ 8 & 10 & 12 \end{bmatrix}$$
- d. $(2 \times 4) \times (4 \times 2)$ ➡️ 가능:
  $$\begin{bmatrix} 1 & 2 & 1 & 2 \\ 4 & 1 & -1 & -4 \end{bmatrix} \begin{bmatrix} 0 & 3 \\ 1 & -1 \\ 2 & 1 \\ 5 & 2 \end{bmatrix} = \begin{bmatrix} 14 & 6 \\ -22 & 2 \end{bmatrix}$$
- e. $(4 \times 2) \times (2 \times 4)$ ➡️ 가능 ($4 \times 4$ 결과 행렬):
  $$\begin{bmatrix} 12 & 3 & -3 & -12 \\ -3 & 1 & 2 & 6 \\ 6 & 5 & 1 & 0 \\ 13 & 12 & 3 & 2 \end{bmatrix}$$

---

## 📝 Part 2. 선형방정식계 해법 & 역행렬 (Exercises 2.5 ~ 2.8)

### [Problem 2.5] 비동차 선형계 $Ax = b$ 의 일반해 집합 도출

#### 1. 문제 풀이 단계
- a. 증대행렬 $[A \mid b]$ 에 가우스 소거법 적용:
  $$\begin{bmatrix} 1 & 1 & -1 & -1 & \mid & 1 \\ 2 & 5 & -7 & -5 & \mid & -2 \\ 2 & -1 & 1 & 3 & \mid & 4 \\ 5 & 2 & -4 & 2 & \mid & 6 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & 0 & -1 & \mid & 3 \\ 0 & 1 & 0 & 0 & \mid & -1 \\ 0 & 0 & 1 & 0 & \mid & 1 \\ 0 & 0 & 0 & 0 & \mid & 0 \end{bmatrix}$$
  - 특수해: $\mathbf{x}_p = [3, -1, 1, 0]^\top$
  - 영공간 기저: $\mathbf{v}_1 = [1, 0, 0, 1]^\top$
  - 일반해: $S = \left\{ \begin{bmatrix} 3 \\ -1 \\ 1 \\ 0 \end{bmatrix} + \lambda \begin{bmatrix} 1 \\ 0 \\ 0 \\ 1 \end{bmatrix} \mid \lambda \in \mathbb{R} \right\}$

- b. 증대행렬 소거:
  $$\xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & 0 & -1.5 & 0 & \mid & 4.5 \\ 0 & 1 & 0 & 1.5 & 0 & \mid & 1.5 \\ 0 & 0 & 0 & 0 & 1 & \mid & 0.5 \\ 0 & 0 & 0 & 0 & 0 & \mid & 0 \end{bmatrix}$$
  - 자유변수: $x_3, x_4$
  - 일반해: $S = \left\{ \begin{bmatrix} 4.5 \\ 1.5 \\ 0 \\ 0 \\ 0.5 \end{bmatrix} + \lambda_1 \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 1.5 \\ -1.5 \\ 0 \\ 1 \\ 0 \end{bmatrix} \mid \lambda_1, \lambda_2 \in \mathbb{R} \right\}$

---

### [Problem 2.6] $3 \times 6$ 선형방정식계 가우스 소거법 해 도출

- RREF 변환:
  $$\begin{bmatrix} 0 & 1 & 0 & 0 & 1 & 0 & \mid & 2 \\ 0 & 0 & 0 & 1 & 1 & 0 & \mid & -1 \\ 0 & 1 & 0 & 0 & 0 & 1 & \mid & 1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 1 & \mid & 1 \\ 0 & 0 & 0 & 1 & 0 & 1 & \mid & 0 \\ 0 & 0 & 0 & 0 & 1 & -1 & \mid & 1 \end{bmatrix}$$
- 자유변수: $x_1, x_3, x_6$
- 일반해:
  $$\mathbf{x} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 1 \\ 0 \end{bmatrix} + \lambda_1 \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_3 \begin{bmatrix} 0 \\ -1 \\ 0 \\ -1 \\ 1 \\ 1 \end{bmatrix}, \quad \lambda_1, \lambda_2, \lambda_3 \in \mathbb{R}$$

---

### [Problem 2.7] 고유값 방정식 $(A - 12I)x = 0$ 및 제약조건 조건 해 도출

- 조건: $(A - 12I)x = 0$ 및 $x_1 + x_2 + x_3 = 1$
  $$A - 12I = \begin{bmatrix} -6 & 4 & 3 \\ 6 & -12 & 9 \\ 0 & 8 & -12 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & -3 \\ 0 & 1 & -1.5 \\ 0 & 0 & 0 \end{bmatrix}$$
  - $x_1 = 3x_3$, $x_2 = 1.5x_3$.
- 합 제약 대입: $3x_3 + 1.5x_3 + x_3 = 5.5x_3 = 1 \implies x_3 = \frac{2}{11}$.
- 유일해 도출: $\mathbf{x} = \begin{bmatrix} 6/11 \\ 3/11 \\ 2/11 \end{bmatrix}$.

---

### [Problem 2.8] 역행렬 계산 및 가역성 판별

- a. $A = \begin{bmatrix} 2 & 3 & 4 \\ 3 & 4 & 5 \\ 4 & 5 & 6 \end{bmatrix}$:
  - $R_3 - R_2 = [1, 1, 1]$, $R_2 - R_1 = [1, 1, 1]$ 로 두 행이 동일하여 $\det(A) = 0$. 역행렬 불가능 (Singular Matrix).
- b. $A = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix}$:
  - 가우스-조던 소거법 $[A \mid I_4] \rightsquigarrow [I_4 \mid A^{-1}]$ 적용:
    $$A^{-1} = \begin{bmatrix} 0 & -1 & 0 & 1 \\ -1 & 0 & 0 & 1 \\ 1 & 1 & 0 & -1 \\ -1 & -1 & 1 & 0 \end{bmatrix}$$

---

## 📝 Part 3. 부분공간, 선형독립 & 기저 (Exercises 2.9 ~ 2.15)

### [Problem 2.9] $\mathbb{R}^3$ 부분공간(Subspace) 판별 3대 조건 검증

- a. $A$: $\lambda(1,1,1) + \mu^3(0,1,-1)$. $\mu^3$ 은 모든 실수를 덮으므로 스칼라배에 닫혀있는 부분공간입니다 (맞음).
- b. $B$: $\lambda=1 \implies (1,-1,0)$, 스칼라배 $2$ 곱하면 $(2,-2,0)$ 이어야 하나, $\lambda=\sqrt{2}$ 일 때만 표현되어 음수 곱셈 덧셈에 닫혀있지 않음 (부분공간 아님).
- c. $C$: $\gamma = 0$ 일 때만 원점 $(0,0,0)$ 을 포함하여 부분공간이 됨 ($\gamma \neq 0$ 이면 아핀 공간).
- d. $D$: $\xi_2 \in \mathbb{Z}$ 스칼라배 $0.5$ 곱하면 정수 집합을 탈출하므로 부분공간 아님.

---

### [Problem 2.10 ~ 2.11] 선형독립성 판별 및 선형 결합 표기

- 2.10 a. $\det([x_1, x_2, x_3]) = 2(8-6) - 1(-8+6) + 3(3-3) = 4 + 2 = 6 \neq 0 \implies$ 선형 독립 (Linearly Independent).
- 2.10 b. $x_1 - x_2 + x_3 = 0 \implies$ 선형 종속 (Linearly Dependent).
- 2.11 선형 결합: $y = c_1 x_1 + c_2 x_2 + c_3 x_3 \implies c_1 = 1, c_2 = -4, c_3 = 2 \implies \mathbf{y = x_1 - 4x_2 + 2x_3}$.

---

### [Problem 2.12 ~ 2.15] 부분공간의 교집합 $U_1 \cap U_2$ 기저 도출

- 2.12: $U_1$ 과 $U_2$ 의 생성 기저 벡터 방정식을 세워 교집합 연산:
  $$\text{Basis of } U_1 \cap U_2 = \left\{ \begin{bmatrix} 1 \\ 3 \\ -7 & 3 \end{bmatrix} \right\}$$
- 2.13 & 2.14: $A_1 x = 0$ 과 $A_2 x = 0$ 동시 만족 영공간 기저 도출.
- 2.15 c: $F$ 의 기저 $\{(1,0,1), (0,1,1)\}$, $G$ 의 기저 $\{(1,1,1), (-1,1,-3)\}$. 교집합 $F \cap G = \text{span}\{(1,3,4)\}$.

---

## 📝 Part 4. 선형사상, 기저변환 & 사상 행렬 (Exercises 2.16 ~ 2.20)

### [Problem 2.16] 선형사상(Linear Mapping) 판별

- a. 적분 연산 $\int_a^b f(x)dx$: 덧셈 및 스칼라배 보존 ➡️ 선형사상 (Linear).
- b. 미분 연산자 $\frac{d}{dx}$: 덧셈 및 스칼라배 보존 ➡️ 선형사상 (Linear).
- c. $\cos(x)$: $\cos(x+y) \neq \cos(x) + \cos(y) \implies$ 비선형사상 (Non-linear).
- d. 행렬 곱셈: 선형사상 (Linear).
- e. 2차원 회전 행렬: 선형사상 (Linear).

---

### [Problem 2.17] 선형사상 행렬 $A_\Phi$, Rank, Kernel 및 Image 계산

- 사상 행렬: $A_\Phi = \begin{bmatrix} 3 & 2 & 1 \\ 1 & 1 & 1 \\ 1 & -3 & 0 \\ 2 & 3 & 1 \end{bmatrix}$
- $\text{Rank}(A_\Phi) = 3$ (Full Column Rank).
- $\dim(\ker(\Phi)) = 0$ ($\ker(\Phi) = \{\mathbf{0}\}$).
- $\dim(\text{Im}(\Phi)) = 3$ ($\text{Im}(\Phi) = \text{Col}(A_\Phi)$).

---

### [Problem 2.18] 자기동형사상(Automorphism) 합성 정리 증명

- $f \circ g = \text{id}_E \implies g = f^{-1}$.
- 따라서 $f$와 $g$는 가역 전단사 함수이므로 $\ker(f) = \{\mathbf{0}\}$, $\text{Im}(g) = E$ 성립.

---

### [Problem 2.19] 기저변환 행렬 $\tilde{A}_\Phi = P^{-1} A_\Phi P$ 도출

- 표준기저 사상 행렬 $A_\Phi = \begin{bmatrix} 1 & 1 & 0 \\ 1 & -1 & 0 \\ 1 & 1 & 1 \end{bmatrix}$.
- 기저변환 행렬 $P = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 0 \\ 1 & 1 & 0 \end{bmatrix} \implies P^{-1} = \begin{bmatrix} 0 & 1 & -2 \\ 0 & -1 & 1 \\ 1 & 0 & 1 \end{bmatrix}$.
- 신기저 표현 행렬: $\tilde{A}_\Phi = P^{-1} A_\Phi P = \begin{bmatrix} 2 & 3 & 0 \\ -1 & -2 & 0 \\ 2 & 3 & 1 \end{bmatrix}$.

---

### [Problem 2.20] 종합 기저변환 및 준동형사상 $A'$ 행렬 정밀 도출

- a, b: $B \to B'$ 기저변환 행렬 $P_1 = \begin{bmatrix} 0 & 1 \\ -2 & 1 \end{bmatrix}$.
- c: $\det(C) = 2 \neq 0 \implies$ 기저 성립. $P_2 = C = \begin{bmatrix} 1 & 0 & 1 \\ 2 & -1 & 0 \\ -1 & 2 & -1 \end{bmatrix}$.
- d, e: 기저변환 공식을 이용한 $A'$ 정밀 도출:
  $$A' = P_2^{-1} A_\Phi P_1$$
- f: 좌표 벡터 변환 및 사상 결과 $A' [2, 3]^\top$ 대조 일치 검증 완료.
