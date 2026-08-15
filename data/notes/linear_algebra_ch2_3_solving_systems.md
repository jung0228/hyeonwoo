# 📐 2.3 Solving Systems of Linear Equations (선형방정식계의 해법)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.3 원문 완전 대조 노트

---

## 1. 🌐 Section 2.3.1: Particular and General Solution (특수해와 일반해)

### 📌 일반 연립 1차 방정식계 (Equation 2.37 & 2.38)
$m$개의 방정식과 $n$개의 미지수로 구성된 선형계는 행렬 곱 $Ax = b$ 로 표기됩니다.

$$A \mathbf{x} = \mathbf{b} \iff \begin{bmatrix} 1 & 0 & 8 & -4 \\\\ 0 & 1 & 2 & 12 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \\\\ x_4 \end{bmatrix} = \begin{bmatrix} 42 \\\\ 8 \end{bmatrix} \quad (2.38)$$

위 예제는 2개의 방정식과 4개의 미지수를 가지므로 무수히 많은 해가 존재합니다.

---

### 📌 특수해 (Particular Solution $\mathbf{x}_p$) 및 동차해 ($\mathbf{x}_h$) 직관 해설 (Eq 2.39 ~ 2.43)

#### 1️⃣ [1단계 개념 정의: $\mathbf{x}_p$와 $\mathbf{x}_h$는 무엇인가?]
선형계 $A\mathbf{x} = \mathbf{b}$ 의 무수히 많은 해는 기준 정답 점 1개 ($\mathbf{x}_p$) 와 0으로 소멸하는 유령 조합들 ($\mathbf{x}_h$) 의 합으로 구성됩니다:

$$\mathbf{x} = \mathbf{x}_p + \mathbf{x}_h$$

- 특수해 ($\mathbf{x}_p$, Particular Solution):
  - 영단어 Particular(특정한)의 앞글자 $p$를 딴 기호입니다.
  - $A \mathbf{x}_p = \mathbf{b}$ 를 100% 만족하는 단 하나의 가장 쉬운 대표 정답 점(기준점)입니다.
- 동차해 ($\mathbf{x}_h$, Homogeneous Solution):
  - 영단어 Homogeneous(동차)의 앞글자 $h$를 딴 기호입니다.
  - $A \mathbf{x}_h = \mathbf{0}$ 이 되어 행렬 $A$에 의해 0으로 사라지는 영공간(Kernel) 기저들의 선형 결합입니다.

---

#### 2️⃣ [2단계 수치 유도: 숫자는 어디서 나왔는가?]

주어진 행렬식 (Eq 2.38):
$$\begin{bmatrix} 1 & 0 & 8 & -4 \\\\ 0 & 1 & 2 & 12 \end{bmatrix} \begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \\\\ x_4 \end{bmatrix} = \begin{bmatrix} 42 \\\\ 8 \end{bmatrix}$$

1. 특수해 $\mathbf{x}_p = \begin{bmatrix} 42 \\\\ 8 \\\\ 0 \\\\ 0 \end{bmatrix}$ 의 도출:
   - 피벗 열인 1열 $\mathbf{c}_1 = \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}$ 과 2열 $\mathbf{c}_2 = \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}$ 만으로 우변 $\mathbf{b} = \begin{bmatrix} 42 \\\\ 8 \end{bmatrix}$ 을 만듭니다 (Eq 2.39):
     $$\mathbf{b} = 42\mathbf{c}_1 + 8\mathbf{c}_2 + 0\mathbf{c}_3 + 0\mathbf{c}_4 \implies \mathbf{x}_p = \begin{bmatrix} 42 \\\\ 8 \\\\ 0 \\\\ 0 \end{bmatrix}$$

2. 동차해 유령 조합 $\mathbf{v}_1, \mathbf{v}_2$ 의 도출:
   - 비피벗 3열 $\mathbf{c}_3 = \begin{bmatrix} 8 \\\\ 2 \end{bmatrix}$ 을 피벗 열들로 표현: $8\mathbf{c}_1 + 2\mathbf{c}_2 - 1\mathbf{c}_3 + 0\mathbf{c}_4 = \mathbf{0} \implies \mathbf{v}_1 = \begin{bmatrix} 8 \\\\ 2 \\\\ -1 \\\\ 0 \end{bmatrix}$
   - 비피벗 4열 $\mathbf{c}_4 = \begin{bmatrix} -4 \\\\ 12 \end{bmatrix}$ 을 피벗 열들로 표현: $-4\mathbf{c}_1 + 12\mathbf{c}_2 + 0\mathbf{c}_3 - 1\mathbf{c}_4 = \mathbf{0} \implies \mathbf{v}_2 = \begin{bmatrix} -4 \\\\ 12 \\\\ 0 \\\\ -1 \end{bmatrix}$

3. 일반해 집합 (General Solution: Eq 2.43):
   $$\text{General Solution: } \{ \mathbf{x} \in \mathbb{R}^4 : \mathbf{x} = \begin{bmatrix} 42 \\\\ 8 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_1 \begin{bmatrix} 8 \\\\ 2 \\\\ -1 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} -4 \\\\ 12 \\\\ 0 \\\\ -1 \end{bmatrix}, \ \lambda_1, \lambda_2 \in \mathbb{R} \}$$

---

#### 3️⃣ [3단계 기하학적 직관]
![4D to 2D Hand-Drawn Blueprint Sketch Transformation Mapping](sketch_4d_to_2d_kernel_mapping.jpg)

- 선형 변환의 찌그러짐 ($A : \mathbb{R}^4 \to \mathbb{R}^2$):  
  왼쪽의 거대한 4차원 입력 공간 $\mathbb{R}^4$ 전체가 행렬 $A$라는 사상(Mapping) 변환기를 통과하면서 오른쪽의 2차원 출력 평면 $\mathbb{R}^2$ 으로 납작하게 압착(Projection)되어 찌그러집니다.
- 영공간(Kernel)으로 사라진 2차원:  
  4차원 중에서 무려 2개의 차원이 변환기를 통과하는 순간 0점으로 완전히 짓눌려 소멸(Null Space / Kernel)합니다. 바로 이 짓눌려 사라진 2차원 평면 전체가 $Ax = b$ 의 무수히 많은 해 공간(Affine Subspace)을 형성하게 됩니다!

---

#### 4️⃣ [4단계 실전 AI 연결고리]
- LLM / 신경망의 해 공간: 초거대 언어모델(LLM)은 파라미터 수(미지수 $n$)가 입력 조건(방정식 $m$)보다 훨씬 많아 자유변수가 무수히 많은 선형계입니다.
- AI가 학습된다는 것은 해 평면 위에서 하나의 최적 특수해 $\mathbf{x}_p$ 를 찾아내는 과정이며, 동차해 공간 $\mathbf{x}_h$ 의 존재 덕분에 파라미터가 약간 흔들려도 동일한 출력을 내놓는 영공간 강건성(Null-space Robustness)을 가집니다.

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

- 해의 존재 조건: 마지막 행 $0 = a + 1$ 에 의해 오직 $a = -1$ 일 때만 해가 존재합니다.
- 특수해 (Eq 2.46): $\mathbf{x}_p = [2, 0, -1, 1, 0]^\top$
- 일반해 집합 (Eq 2.47):
  $${ \mathbf{x} \in \mathbb{R}^5 : \mathbf{x} = \begin{bmatrix} 2 \\\\ 0 \\\\ -1 \\\\ 1 \\\\ 0 \end{bmatrix} + \lambda_1 \begin{bmatrix} 2 \\\\ 1 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 2 \\\\ 0 \\\\ -1 \\\\ 2 \\\\ 1 \end{bmatrix}, \ \lambda_1, \lambda_2 \in \mathbb{R} }$$

---

### 📌 Definition 2.6 (Row-Echelon Form 행 사다리꼴 REF) & 피벗
1. 영행(모든 성분이 0인 행)은 행렬의 가장 아래쪽에 위치합니다.
2. 0이 아닌 행의 첫 번째 0이 아닌 성분(피벗 Pivot / Leading Coefficient)은 위쪽 행 피벗보다 반드시 오른쪽에 위치합니다.
- Basic & Free Variables (주변수와 자유변수): 피벗 열에 대응하는 변수($x_1, x_3, x_4$)를 주변수(Basic Variables), 나머지 변수($x_2, x_5$)를 자유변수(Free Variables)라 부릅니다.
- Reduced Row-Echelon Form (기약 행 사다리꼴 RREF): REF 상태에서 모든 피벗이 1이고, 피벗이 속한 열의 다른 성분이 모두 0인 형태입니다.

---

### 📌 Example 2.7 (RREF 형태 및 영공간 추출: Eq 2.49 & 2.50)

$$A = \begin{bmatrix} \mathbf{1} & 3 & 0 & 0 & 3 \\\\ 0 & 0 & \mathbf{1} & 0 & 9 \\\\ 0 & 0 & 0 & \mathbf{1} & -4 \end{bmatrix}$$

비피벗 열(2열, 5열)을 피벗 열들(1열, 3열, 4열)의 선형 결합으로 표현하여 동차해 $Ax = 0$ 의 일반해 집합을 추출합니다 (Eq 2.50):

$$\text{Solutions of } Ax = 0: \quad \mathbf{x} = \lambda_1 \begin{bmatrix} 3 \\\\ -1 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\\\ 0 \\\\ 9 \\\\ -4 \\\\ -1 \end{bmatrix}, \quad \lambda_1, \lambda_2 \in \mathbb{R}$$

---

### 📌 -1 Trick (-1 트릭을 통한 영공간 기저 즉시 도출 직관 해설: Eq 2.51 ~ 2.55)

#### 1️⃣ [-1 트릭이란 무엇인가?]
$Ax = 0$ 이 되는 영공간(Kernel) 기저를 구할 때, 연립방정식을 이항하고 풀 필요 없이 행렬 대각선 빈자리에 $[-1]$ 행을 억지로 쓱 삽입한 뒤, 그 세로 줄을 그대로 베껴 쓰면 1초 만에 답이 튀어나오는 매직 꼼수(Trick)입니다.

---

#### 2️⃣ [-1 트릭 실전 2단계 조작법 (Example 2.8)]

주어진 $3 \times 5$ 기약 행 사다리꼴(RREF) 행렬 $A$:
$$A = \begin{bmatrix} \mathbf{1} & 3 & 0 & 0 & 3 \\\\ 0 & 0 & \mathbf{1} & 0 & 9 \\\\ 0 & 0 & 0 & \mathbf{1} & -4 \end{bmatrix}$$

1. [1단계: 대각선 빠진 행에 `-1` 억지 삽입]:
   - 대각선 $a_{11}, a_{22}, a_{33}, a_{44}, a_{55}$ 중 피벗이 빠진 2행과 5행에 `[0 ... -1 ... 0]` 행을 추가해 $5 \times 5$ 확장 행렬 $\tilde{A}$를 만듭니다 (Eq 2.54):
     $$\tilde{A} = \begin{bmatrix} 1 & 3 & 0 & 0 & 3 \\\\ 0 & \mathbf{-1} & 0 & 0 & 0 \\\\ 0 & 0 & 1 & 0 & 9 \\\\ 0 & 0 & 0 & 1 & -4 \\\\ 0 & 0 & 0 & 0 & \mathbf{-1} \end{bmatrix}$$

2. [2단계: `-1` 이 있는 세로 줄을 그대로 복사하기]:
   - 대각선 성분이 $-1$ 인 2번째 열과 5번째 열을 세로 그대로 베껴 씁니다:
     $$\mathbf{v}_1 = \begin{bmatrix} 3 \\\\ \mathbf{-1} \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 3 \\\\ 0 \\\\ 9 \\\\ -4 \\\\ \mathbf{-1} \end{bmatrix}$$

3. [최종 정답 자동 완성]:
   - 이 두 세로 열이 곧바로 $Ax = 0$ 의 영공간(Kernel) 기저가 됩니다!
     $$\text{Solutions of } Ax = 0: \quad \mathbf{x} = \lambda_1 \begin{bmatrix} 3 \\\\ -1 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\\\ 0 \\\\ 9 \\\\ -4 \\\\ -1 \end{bmatrix}$$

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
- 가우스 소거법의 한계: 수천 개 미지수까지는 가우스 소거법으로 해결 가능하지만, 수백만 개 미지수의 대형 시스템에서는 복잡도가 $\mathcal{O}(n^3)$ 으로 폭발하여 비실용적입니다.
- 반복법 (Iterative Methods): 대규모 연립방정식계는 다음과 같은 반복 수식 $x^{(k+1)} = C x^{(k)} + d$ 를 통해 수렴시키는 수치 해법을 사용합니다:
  - 정류 반복법 (Stationary Iterative Methods): Jacobi method, Gauss-Seidel method, Richardson method, SOR.
  - 크릴로프 부분공간법 (Krylov Subspace Methods): Conjugate Gradients (CG), GMRES, BiCGSTAB.
