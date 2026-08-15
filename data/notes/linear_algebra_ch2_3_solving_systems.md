# 📐 2.3 Solving Systems of Linear Equations (선형방정식계의 해법)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.3 원문 완전 대조 노트**

---

## 1. 🌐 Section 2.3.1: Particular and General Solution (특수해와 일반해)

### 📌 일반 연립 1차 방정식계 (Equation 2.37 & 2.38)
$m$개의 방정식과 $n$개의 미지수로 구성된 선형계는 행렬 곱 $Ax = b$ 로 표기됩니다.

$$A \mathbf{x} = \mathbf{b} \iff \begin{bmatrix} 1 & 0 & 8 & -4 \\\\ 0 & 1 & 2 & 12 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \\\\ x_4 \end{bmatrix} = \begin{bmatrix} 42 \\\\ 8 \end{bmatrix} \quad (2.38)$$

위 예제는 2개의 방정식과 4개의 미지수를 가지므로 무수히 많은 해가 존재합니다.

---

### 📌 특수해 (Particular / Special Solution) 및 동차해 (Equation 2.39 ~ 2.43)

1. **특수해 (Particular Solution)**:
   열벡터 $\mathbf{c}_1, \mathbf{c}_2$를 이용하여 우변 $\mathbf{b}$를 만들어내는 하나의 해점입니다 (Eq 2.39):
   $$\mathbf{b} = \begin{bmatrix} 42 \\\\ 8 \end{bmatrix} = 42 \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} + 8 \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} \implies \mathbf{x}_p = \begin{bmatrix} 42 \\\\ 8 \\\\ 0 \\\\ 0 \end{bmatrix}$$

2. **영공간 동차해 (Homogeneous Solution)**:
   비피벗 열 $\mathbf{c}_3, \mathbf{c}_4$를 피벗 열들의 선형 결합으로 표현하여 $A\mathbf{x} = 0$ 이 되는 영벡터 생성 조합을 찾습니다 (Eq 2.40~2.42):
   $$\mathbf{c}_3 = 8\mathbf{c}_1 + 2\mathbf{c}_2 \implies 8\mathbf{c}_1 + 2\mathbf{c}_2 - 1\mathbf{c}_3 + 0\mathbf{c}_4 = \mathbf{0}$$
   $$\mathbf{c}_4 = -4\mathbf{c}_1 + 12\mathbf{c}_2 \implies -4\mathbf{c}_1 + 12\mathbf{c}_2 + 0\mathbf{c}_3 - 1\mathbf{c}_4 = \mathbf{0}$$

3. **일반해 집합 (General Solution: Eq 2.43)**:
   일반해는 특수해 $\mathbf{x}_p$에 동차해 기저들의 임의 스칼라배 선형 결합이 더해진 아핀 공간 형태를 이룹니다:

   $$\text{General Solution: } \{ \mathbf{x} \in \mathbb{R}^4 : \mathbf{x} = \begin{bmatrix} 42 \\\\ 8 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_1 \begin{bmatrix} 8 \\\\ 2 \\\\ -1 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} -4 \\\\ 12 \\\\ 0 \\\\ -1 \end{bmatrix}, \ \lambda_1, \lambda_2 \in \mathbb{R} \}$$

- **Remark (일반해 구하기 3단계)**:
  - [1단계]: $Ax = b$ 의 특수해 $\mathbf{x}_p$ 하나를 구합니다.
  - [2단계]: 동차방정식 $Ax = 0$ 의 모든 해 공간(Kernel)을 구합니다.
  - [3단계]: 1단계와 2단계의 해를 합쳐 일반해를 구성합니다.

---

## 2. ⚔️ Section 2.3.2: Elementary Transformations (기본 행 연산 & 가우스 소거법)

### 📌 기본 행 연산 (Elementary Row Operations) 3가지
해가 변하지 않도록 방정식을 동등하게 변형하는 3가지 연산:
1. 두 행(방정식)의 위치 교환 (Swap)
2. 한 행에 0이 아닌 스칼라 $\lambda \in \mathbb{R} \setminus \{0\}$ 배 곱함
3. 한 행의 스칼라 배를 다른 행에 더함

---

### 📌 Example 2.6 (가우스 소거법 전개 과정: Eq 2.44 ~ 2.47)

다음 증대행렬 $[A \mid b]$ 에 기본 행 연산을 적용합니다:

$$\begin{bmatrix} -2 & 4 & -2 & -1 & 4 & \mid & -3 \\\\ 4 & -8 & 3 & -3 & 1 & \mid & 2 \\\\ 1 & -2 & 1 & -1 & 1 & \mid & 0 \\\\ 1 & -2 & 0 & -3 & 4 & \mid & a \end{bmatrix} \xrightarrow{R_1 \leftrightarrow R_3} \dots \xrightarrow{\text{REF}} \begin{bmatrix} 1 & -2 & 1 & -1 & 1 & \mid & 0 \\\\ 0 & 0 & 1 & -1 & 3 & \mid & -2 \\\\ 0 & 0 & 0 & 1 & -2 & \mid & 1 \\\\ 0 & 0 & 0 & 0 & 0 & \mid & a + 1 \end{bmatrix}$$

- **해의 존재 조건**: 마지막 행 $0 = a + 1$ 에 의해 오직 $a = -1$ 일 때만 해가 존재합니다.
- **특수해 (Eq 2.46)**: $\mathbf{x}_p = [2, 0, -1, 1, 0]^\top$
- **일반해 집합 (Eq 2.47)**:
  $${ \mathbf{x} \in \mathbb{R}^5 : \mathbf{x} = \begin{bmatrix} 2 \\\\ 0 \\\\ -1 \\\\ 1 \\\\ 0 \end{bmatrix} + \lambda_1 \begin{bmatrix} 2 \\\\ 1 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 2 \\\\ 0 \\\\ -1 \\\\ 2 \\\\ 1 \end{bmatrix}, \ \lambda_1, \lambda_2 \in \mathbb{R} }$$

---

### 📌 Definition 2.6 (Row-Echelon Form 행 사다리꼴 REF) & 피벗
1. 영행(모든 성분이 0인 행)은 행렬의 가장 아래쪽에 위치합니다.
2. 0이 아닌 행의 첫 번째 0이 아닌 성분(피벗 Pivot / Leading Coefficient)은 위쪽 행 피벗보다 반드시 오른쪽에 위치합니다.
- **Basic & Free Variables (주변수와 자유변수)**: 피벗 열에 대응하는 변수($x_1, x_3, x_4$)를 주변수(Basic Variables), 나머지 변수($x_2, x_5$)를 자유변수(Free Variables)라 부릅니다.
- **Reduced Row-Echelon Form (기약 행 사다리꼴 RREF)**: REF 상태에서 모든 피벗이 1이고, 피벗이 속한 열의 다른 성분이 모두 0인 형태입니다.

---

### 📌 Example 2.7 (RREF 형태 및 영공간 추출: Eq 2.49 & 2.50)

$$A = \begin{bmatrix} \mathbf{1} & 3 & 0 & 0 & 3 \\\\ 0 & 0 & \mathbf{1} & 0 & 9 \\\\ 0 & 0 & 0 & \mathbf{1} & -4 \end{bmatrix}$$

비피벗 열(2열, 5열)을 피벗 열들(1열, 3열, 4열)의 선형 결합으로 표현하여 동차해 $Ax = 0$ 의 일반해 집합을 추출합니다 (Eq 2.50):

$$\text{Solutions of } Ax = 0: \quad \mathbf{x} = \lambda_1 \begin{bmatrix} 3 \\\\ -1 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\\\ 0 \\\\ 9 \\\\ -4 \\\\ -1 \end{bmatrix}, \quad \lambda_1, \lambda_2 \in \mathbb{R}$$

---

## 3. ⚔️ Section 2.3.3: The Minus-1 Trick & Calculating the Inverse (-1 트릭과 역행렬 계산)

### 📌 -1 Trick (-1 트릭을 통한 영공간 기저 즉시 도출: Eq 2.51 & 2.52)
RREF 형태의 행렬 $A \in \mathbb{R}^{k \times n}$ 에 피벗이 없는 대각 위치마다 $[-1]$ 행을 추가하여 $n \times n$ 확장 행렬 $\tilde{A}$를 만듭니다.

대각선상에 $[-1]$ 이 들어간 열을 그대로 읽어내면 $Ax = 0$ 동차방정식계의 영공간(Kernel / Null Space) 기저 벡터가 즉시 얻어집니다.

---

### 📌 Example 2.8 (-1 Trick 실전 적용: Eq 2.53 ~ 2.55)

Eq (2.49)의 $3 \times 5$ RREF 행렬 $A$에 피벗이 빠진 2행과 5행 위치에 $[-1]$ 행을 대입하여 $5 \times 5$ 확장 행렬 $\tilde{A}$를 구성합니다 (Eq 2.54):

$$\tilde{A} = \begin{bmatrix} 1 & 3 & 0 & 0 & 3 \\\\ 0 & -1 & 0 & 0 & 0 \\\\ 0 & 0 & 1 & 0 & 9 \\\\ 0 & 0 & 0 & 1 & -4 \\\\ 0 & 0 & 0 & 0 & -1 \end{bmatrix}$$

대각선 성분이 $-1$ 인 2열과 5열을 읽어내면 동차해 영공간 기저가 완벽하게 일치합니다 (Eq 2.55).

---

### 📌 가우스-조던 소거법을 통한 역행렬 계산 (Eq 2.56 ~ 2.58)
정방행렬 $A \in \mathbb{R}^{n \times n}$ 의 역행렬 $A^{-1}$ 은 증대행렬 $[A \mid I_n]$ 에 가우스 소거법을 적용하여 $[I_n \mid A^{-1}]$ 형태로 변환하여 구합니다 (Eq 2.56).

### 📌 Example 2.9 (역행렬 계산 수치 예시)

$$[A \mid I_4] = \begin{bmatrix} 1 & 0 & 2 & 0 & \mid & 1 & 0 & 0 & 0 \\\\ 1 & 1 & 0 & 0 & \mid & 0 & 1 & 0 & 0 \\\\ 1 & 2 & 0 & 1 & \mid & 0 & 0 & 1 & 0 \\\\ 1 & 1 & 1 & 1 & \mid & 0 & 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & 0 & 0 & \mid & -1 & 2 & -2 & 2 \\\\ 0 & 1 & 0 & 0 & \mid & 1 & -1 & 2 & -2 \\\\ 0 & 0 & 1 & 0 & \mid & 1 & -1 & 1 & -1 \\\\ 0 & 0 & 0 & 1 & \mid & -1 & 0 & -1 & 2 \end{bmatrix}$$

따라서 $A$의 역행렬은 우변의 행렬로 얻어집니다 (Eq 2.58):
$$A^{-1} = \begin{bmatrix} -1 & 2 & -2 & 2 \\\\ 1 & -1 & 2 & -2 \\\\ 1 & -1 & 1 & -1 \\\\ -1 & 0 & -1 & 2 \end{bmatrix}$$

---

## 4. ⚔️ Section 2.3.4: Algorithms for Solving Systems (선형계 알고리즘 & 반복법)

### 📌 최소제곱법 유사역행렬 (Moore-Penrose Pseudo-inverse: Eq 2.59)
해가 존재하지 않거나 정방행렬이 아닐 때, 열들이 선형 독립이라는 가정하에 무어-펜로즈 유사역행렬을 사용하여 최소제곱해를 구합니다:

$$Ax = b \iff A^\top A x = A^\top b \iff x = (A^\top A)^{-1} A^\top b \quad (2.59)$$

---

### 📌 수치적 연산 복잡도 & 반복적 해법 (Stationary & Krylov Subspace Methods: Eq 2.60)
- **가우스 소거법의 한계**: 수천 개 미지수까지는 가우스 소거법으로 해결 가능하지만, 수백만 개 미지수의 대형 시스템에서는 복잡도가 $\mathcal{O}(n^3)$ 으로 폭발하여 비실용적입니다.
- **반복법 (Iterative Methods)**: 대규모 연립방정식계는 다음과 같은 반복 수식 $x^{(k+1)} = C x^{(k)} + d$ 를 통해 수렴시키는 수치 해법을 사용합니다:
  - **정류 반복법 (Stationary Iterative Methods)**: Jacobi method, Gauss-Seidel method, Richardson method, SOR.
  - **크릴로프 부분공간법 (Krylov Subspace Methods)**: Conjugate Gradients (CG), GMRES, BiCGSTAB.
