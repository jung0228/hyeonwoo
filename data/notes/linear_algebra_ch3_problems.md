# 📐 MML Chapter 3 해석기하학 전수 연습문제 풀이집 (Exercises 3.1 ~ 3.10)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Chapter 3 Analytic Geometry 전수 문제 풀이집
> 
> 본 문서에는 MML Chapter 3 (Exercises 3.1 ~ 3.10) 전 문항의 단계별 수학적 증명, 수치 풀이, 기하학적 직관 및 실전 AI 연결고리가 수록되어 있습니다.


## 📝 [Problem 3.1] 내적(Inner Product) 3대 공리 증명

### 1. 문제 정의
$\mathbb{R}^2$ 상의 임의의 두 벡터 $\mathbf{x} = [x_1, x_2]^\top, \mathbf{y} = [y_1, y_2]^\top$ 에 대해 다음과 같이 정의된 연산이 내적임을 증명하시오:
$$\langle \mathbf{x}, \mathbf{y} \rangle := x_1 y_1 - (x_1 y_2 + x_2 y_1) + 2 x_2 y_2$$

### 2. 단계별 정밀 증명
1. 이선형성 (Bilinearity)
   - 연산을 행렬 곱 형태로 표현하면 다음과 같습니다:
     $$\langle \mathbf{x}, \mathbf{y} \rangle = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} 1 & -1 \\\\ -1 & 2 \end{bmatrix} \begin{bmatrix} y_1 \\\\ y_2 \end{bmatrix} = \mathbf{x}^\top A \mathbf{y}$$
   - 행렬 곱 $\mathbf{x}^\top A \mathbf{y}$ 은 행렬 대수의 분배법칙에 의해 양쪽 인자에 대해 선형성을 만족하므로 이선형 사상입니다.

2. 대칭성 (Symmetry)
   - 행렬 $A = \begin{bmatrix} 1 & -1 \\\\ -1 & 2 \end{bmatrix}$ 에 대해 $A^\top = A$ 이므로 대칭 행렬입니다.
   - $\langle \mathbf{y}, \mathbf{x} \rangle = y_1 x_1 - (y_1 x_2 + y_2 x_1) + 2 y_2 x_2 = \langle \mathbf{x}, \mathbf{y} \rangle$ 성립.

3. 양의 정정성 (Positive Definiteness)
   - $\langle \mathbf{x}, \mathbf{x} \rangle = x_1^2 - 2 x_1 x_2 + 2 x_2^2 = (x_1 - x_2)^2 + x_2^2 \ge 0$.
   - 제곱의 합이 0이 되는 조건: $(x_1 - x_2)^2 = 0$ 이고 $x_2^2 = 0 \iff x_2 = 0, x_1 = 0 \iff \mathbf{x} = \mathbf{0}$.
   - 따라서 0이 아닌 모든 $\mathbf{x} \neq \mathbf{0}$ 에 대해 $\langle \mathbf{x}, \mathbf{x} \rangle > 0$ 이며 $\langle \mathbf{0}, \mathbf{0} \rangle = 0$ 이 성립합니다.

- 결론: 3대 공리를 모두 만족하므로 유효한 내적입니다.


## 📝 [Problem 3.2] 비대칭 행렬의 내적 성립 여부 판별

### 1. 문제 정의
$A = \begin{bmatrix} 2 & 0 \\\\ 1 & 2 \end{bmatrix}$ 에 대해 $\langle \mathbf{x}, \mathbf{y} \rangle := \mathbf{x}^\top A \mathbf{y}$ 가 $\mathbb{R}^2$ 상의 내적인지 판별하시오.

### 2. 단계별 풀이 및 반례 증명
- 대칭성 검증:
  $$A^\top = \begin{bmatrix} 2 & 1 \\\\ 0 & 2 \end{bmatrix} \neq \begin{bmatrix} 2 & 0 \\\\ 1 & 2 \end{bmatrix} = A$$
- 반례 제시: $\mathbf{x} = \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}, \mathbf{y} = \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}$ 일 때,
  $$\langle \mathbf{x}, \mathbf{y} \rangle = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\\\ 1 & 2 \end{bmatrix} \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\\\ 2 \end{bmatrix} = 0$$
  $$\langle \mathbf{y}, \mathbf{x} \rangle = \begin{bmatrix} 0 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\\\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \end{bmatrix} \begin{bmatrix} 2 \\\\ 1 \end{bmatrix} = 1$$
- $\langle \mathbf{x}, \mathbf{y} \rangle \neq \langle \mathbf{y}, \mathbf{x} \rangle$ 이므로 대칭성 공리를 위반합니다.
- 결론: 내적이 아닙니다.


## 📝 [Problem 3.3] 도트곱 및 가중치 행렬 내적 하에서의 두 점 간 거리 계산

### 1. 문제 정의
$\mathbf{x} = \begin{bmatrix} 1 \\\\ 2 \\\\ 3 \end{bmatrix}, \mathbf{y} = \begin{bmatrix} -1 \\\\ -1 \\\\ 0 \end{bmatrix}$ 에 대해 다음 내적 기준에서의 거리 $d(\mathbf{x}, \mathbf{y})$ 를 계산하시오:
- a. 표준 도트 곱 $\langle \mathbf{x}, \mathbf{y} \rangle := \mathbf{x}^\top \mathbf{y}$
- b. 가중치 내적 $\langle \mathbf{x}, \mathbf{y} \rangle := \mathbf{x}^\top A \mathbf{y}, \quad A := \begin{bmatrix} 2 & 1 & 0 \\\\ 1 & 3 & -1 \\\\ 0 & -1 & 2 \end{bmatrix}$

### 2. 단계별 수치 계산
차이 벡터: $\mathbf{d} = \mathbf{x} - \mathbf{y} = \begin{bmatrix} 1 - (-1) \\\\ 2 - (-1) \\\\ 3 - 0 \end{bmatrix} = \begin{bmatrix} 2 \\\\ 3 \\\\ 3 \end{bmatrix}$

- a. 표준 도트 곱 거리:
  $$d(\mathbf{x}, \mathbf{y}) = \sqrt{\mathbf{d}^\top \mathbf{d}} = \sqrt{2^2 + 3^2 + 3^2} = \sqrt{4 + 9 + 9} = \sqrt{22} \approx 4.690$$

- b. 가중치 내적 $A$ 하에서의 거리:
  $$A \mathbf{d} = \begin{bmatrix} 2 & 1 & 0 \\\\ 1 & 3 & -1 \\\\ 0 & -1 & 2 \end{bmatrix} \begin{bmatrix} 2 \\\\ 3 \\\\ 3 \end{bmatrix} = \begin{bmatrix} 2(2) + 1(3) + 0 \\\\ 1(2) + 3(3) - 1(3) \\\\ 0 - 1(3) + 2(3) \end{bmatrix} = \begin{bmatrix} 7 \\\\ 8 \\\\ 3 \end{bmatrix}$$
  $$\mathbf{d}^\top A \mathbf{d} = \begin{bmatrix} 2 & 3 & 3 \end{bmatrix} \begin{bmatrix} 7 \\\\ 8 \\\\ 3 \end{bmatrix} = 2(7) + 3(8) + 3(3) = 14 + 24 + 9 = 47$$
  $$d(\mathbf{x}, \mathbf{y}) = \sqrt{47} \approx 6.856$$


## 📝 [Problem 3.4] 도트곱 및 가중치 행렬 내적 하에서의 사잇각 계산

### 1. 문제 정의
$\mathbf{x} = \begin{bmatrix} 1 \\\\ 2 \end{bmatrix}, \mathbf{y} = \begin{bmatrix} -1 \\\\ -1 \end{bmatrix}$ 에 대해 사잇각 $\omega$ 를 계산하시오:
- a. 표준 도트 곱 $\langle \mathbf{x}, \mathbf{y} \rangle := \mathbf{x}^\top \mathbf{y}$
- b. 가중치 내적 $\langle \mathbf{x}, \mathbf{y} \rangle := \mathbf{x}^\top B \mathbf{y}, \quad B := \begin{bmatrix} 2 & 1 \\\\ 1 & 3 \end{bmatrix}$

### 2. 단계별 수치 계산
- a. 표준 도트 곱:
  $$\langle \mathbf{x}, \mathbf{y} \rangle = 1(-1) + 2(-1) = -3$$
  $$\Vert\mathbf{x}\Vert = \sqrt{1^2 + 2^2} = \sqrt{5}, \quad \Vert\mathbf{y}\Vert = \sqrt{(-1)^2 + (-1)^2} = \sqrt{2}$$
  $$\cos\omega = \frac{-3}{\sqrt{5}\sqrt{2}} = \frac{-3}{\sqrt{10}} \implies \omega = \arccos\left(-\frac{3}{\sqrt{10}}\right) \approx 2.820 \text{ rad} \approx 161.57^\circ$$

- b. 가중치 내적 $B$:
  $$B \mathbf{y} = \begin{bmatrix} 2 & 1 \\\\ 1 & 3 \end{bmatrix} \begin{bmatrix} -1 \\\\ -1 \end{bmatrix} = \begin{bmatrix} -3 \\\\ -4 \end{bmatrix}$$
  $$\langle \mathbf{x}, \mathbf{y} \rangle_B = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} -3 \\\\ -4 \end{bmatrix} = -3 - 8 = -11$$
  $$\Vert\mathbf{x}\Vert_B^2 = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 2 & 1 \\\\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1 \\\\ 2 \end{bmatrix} = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 4 \\\\ 7 \end{bmatrix} = 18 \implies \Vert\mathbf{x}\Vert_B = \sqrt{18} = 3\sqrt{2}$$
  $$\Vert\mathbf{y}\Vert_B^2 = \begin{bmatrix} -1 & -1 \end{bmatrix} \begin{bmatrix} -3 \\\\ -4 \end{bmatrix} = 7 \implies \Vert\mathbf{y}\Vert_B = \sqrt{7}$$
  $$\cos\omega = \frac{-11}{\sqrt{18}\sqrt{7}} = \frac{-11}{\sqrt{126}} \implies \omega = \arccos\left(\frac{-11}{\sqrt{126}}\right) \approx 2.941 \text{ rad} \approx 168.49^\circ$$


## 📝 [Problem 3.5] 5차원 부분공간 기저 판별 및 직교 정사영/거리 계산

### 1. 문제 정의
$U = \text{span}[\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3, \mathbf{u}_4] \subseteq \mathbb{R}^5, \mathbf{x} = [-1, -9, -1, 4, 1]^\top$
- $\mathbf{u}_1 = [0, -1, 2, 0, 2]^\top, \mathbf{u}_2 = [1, -3, 1, -1, 2]^\top, \mathbf{u}_3 = [-3, 4, 1, 2, 1]^\top, \mathbf{u}_4 = [-1, -3, 5, 0, 7]^\top$
- a. 직교 정사영 $\pi_U(\mathbf{x})$ 도출
- b. 최단거리 $d(\mathbf{x}, U)$ 도출

### 2. 단계별 풀이
1. 선형 종속성 검사 및 기저 축출:
   - $\mathbf{u}_1 + 2\mathbf{u}_2 + \mathbf{u}_3 = [0+2-3, -1-6+4, 2+2+1, 0-2+2, 2+4+1]^\top = [-1, -3, 5, 0, 7]^\top = \mathbf{u}_4$.
   - 따라서 $\mathbf{u}_4$ 는 종속 벡터이며, $U$ 의 3차원 기저는 $B = [\mathbf{u}_1 \mid \mathbf{u}_2 \mid \mathbf{u}_3]$ 입니다.

2. 정규방정식 $B^\top B \boldsymbol{\lambda} = B^\top \mathbf{x}$ 구축:
   $$B^\top B = \begin{bmatrix} 9 & 9 & 0 \\\\ 9 & 16 & -14 \\\\ 0 & -14 & 31 \end{bmatrix}, \quad B^\top \mathbf{x} = \begin{bmatrix} 9 \\\\ 23 \\\\ -25 \end{bmatrix}$$
   - 연립방정식 풀이: $\lambda_1 = -3, \; \lambda_2 = 4, \; \lambda_3 = 1 \implies \boldsymbol{\lambda} = \begin{bmatrix} -3 \\\\ 4 \\\\ 1 \end{bmatrix}$.

3. 정사영점 $\pi_U(\mathbf{x})$ 계산 (소문제 a):
   $$\pi_U(\mathbf{x}) = -3\mathbf{u}_1 + 4\mathbf{u}_2 + \mathbf{u}_3 = \begin{bmatrix} 1 \\\\ -5 \\\\ -1 \\\\ -2 \\\\ 3 \end{bmatrix}$$

4. 거리 $d(\mathbf{x}, U)$ 계산 (소문제 b):
   $$\mathbf{x} - \pi_U(\mathbf{x}) = \begin{bmatrix} -1 \\\\ -9 \\\\ -1 \\\\ 4 \\\\ 1 \end{bmatrix} - \begin{bmatrix} 1 \\\\ -5 \\\\ -1 \\\\ -2 \\\\ 3 \end{bmatrix} = \begin{bmatrix} -2 \\\\ -4 \\\\ 0 \\\\ 6 \\\\ -2 \end{bmatrix}$$
   $$d(\mathbf{x}, U) = \sqrt{(-2)^2 + (-4)^2 + 0^2 + 6^2 + (-2)^2} = \sqrt{4 + 16 + 36 + 4} = \sqrt{60} = 2\sqrt{15} \approx 7.746$$


## 📝 [Problem 3.6] 일반 내적 공간에서의 기저 정사영 및 거리

### 1. 문제 정의
$\mathbb{R}^3$ 내적 $A = \begin{bmatrix} 2 & 1 & 0 \\\\ 1 & 2 & -1 \\\\ 0 & -1 & 2 \end{bmatrix}, U = \text{span}[\mathbf{e}_1, \mathbf{e}_3]$ 일 때:
- a. $\mathbf{e}_2$ 의 $U$ 위로의 직교 정사영 $\pi_U(\mathbf{e}_2)$ 도출
- b. 거리 $d(\mathbf{e}_2, U)$ 도출
- c. 기하학적 상황 서술

### 2. 단계별 풀이
기저 행렬 $B = [\mathbf{e}_1 \mid \mathbf{e}_3] = \begin{bmatrix} 1 & 0 \\\\ 0 & 0 \\\\ 0 & 1 \end{bmatrix}$.
- $B^\top A B = \begin{bmatrix} 2 & 0 \\\\ 0 & 2 \end{bmatrix}, \quad B^\top A \mathbf{e}_2 = \begin{bmatrix} 1 \\\\ -1 \end{bmatrix}$.
- 좌표: $\boldsymbol{\lambda} = (B^\top A B)^{-1} B^\top A \mathbf{e}_2 = \begin{bmatrix} 1/2 \\\\ -1/2 \end{bmatrix}$.

- a. 사영점:
  $$\pi_U(\mathbf{e}_2) = \frac{1}{2}\mathbf{e}_1 - \frac{1}{2}\mathbf{e}_3 = \begin{bmatrix} 1/2 \\\\ 0 \\\\ -1/2 \end{bmatrix}$$

- b. 거리 계산:
  $$\mathbf{e} = \mathbf{e}_2 - \pi_U(\mathbf{e}_2) = \begin{bmatrix} -1/2 \\\\ 1 \\\\ 1/2 \end{bmatrix}$$
  $$A \mathbf{e} = \begin{bmatrix} 0 \\\\ 1 \\\\ 0 \end{bmatrix} \implies d(\mathbf{e}_2, U) = \sqrt{\mathbf{e}^\top A \mathbf{e}} = \sqrt{\begin{bmatrix} -1/2 & 1 & 1/2 \end{bmatrix} \begin{bmatrix} 0 \\\\ 1 \\\\ 0 \end{bmatrix}} = \sqrt{1} = 1$$

- c. 기하학적 서술:
  표준 도트 곱에서는 $\mathbf{e}_2$ 가 $xz$ 평면($U$)과 완벽히 수직이어서 사영이 $\mathbf{0}$ 이지만, 비대각 결합 성분($A_{12}=1, A_{23}=-1$)이 존재하는 내적 $A$ 하에서는 $\mathbf{e}_2$ 축이 $xz$ 평면 쪽으로 기울어져 사영 벡터가 $\frac{1}{2}\mathbf{e}_1 - \frac{1}{2}\mathbf{e}_3$ 로 0이 아니게 나타납니다.


## 📝 [Problem 3.7] 사영 사상 $\pi$ 와 여사영 $(\text{id}_V - \pi)$ 의 상(Image) & 핵(Kernel) 정리 증명

### 1. 문제 정의
- a. $\pi$ 가 정사영 $\iff \text{id}_V - \pi$ 가 정사영임을 증명하시오.
- b. $\text{Im}(\text{id}_V - \pi)$ 와 $\ker(\text{id}_V - \pi)$ 를 $\text{Im}(\pi)$ 와 $\ker(\pi)$ 로 표현하시오.

### 2. 정밀 증명
- a. $Q = \text{id}_V - \pi$ 라 할 때:
  $$Q^2 = (\text{id}_V - \pi)(\text{id}_V - \pi) = \text{id}_V - 2\pi + \pi^2$$
  $$Q^2 = Q \iff \text{id}_V - 2\pi + \pi^2 = \text{id}_V - \pi \iff \pi^2 = \pi$$
  따라서 $\pi$ 가 멱등 사영인 것은 $\text{id}_V - \pi$ 가 멱등 사영인 것과 완전히 동치입니다.

- b. 핵과 상의 상호 교환 관계 도출:
  1. $\ker(\text{id}_V - \pi) = \{\mathbf{x} \in V \mid \mathbf{x} - \pi(\mathbf{x}) = \mathbf{0}\} = \{\mathbf{x} \in V \mid \pi(\mathbf{x}) = \mathbf{x}\} = \text{Im}(\pi)$.
  2. $\text{Im}(\text{id}_V - \pi) = \ker(\pi)$.
     - $\mathbf{x} \in \ker(\pi) \implies (\text{id}_V - \pi)(\mathbf{x}) = \mathbf{x} \in \text{Im}(\text{id}_V - \pi)$.
     - $\mathbf{y} \in \text{Im}(\text{id}_V - \pi) \implies \mathbf{y} = \mathbf{x} - \pi(\mathbf{x}) \implies \pi(\mathbf{y}) = \pi(\mathbf{x}) - \pi^2(\mathbf{x}) = \mathbf{0} \implies \mathbf{y} \in \ker(\pi)$.
  - 결론: $\ker(\text{id}_V - \pi) = \text{Im}(\pi)$, $\text{Im}(\text{id}_V - \pi) = \ker(\pi)$.


## 📝 [Problem 3.8] 그람-슈미트 직교화를 통한 정규직교기저(ONB) 구축

### 1. 문제 정의
$\mathbf{b}_1 = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix}, \mathbf{b}_2 = \begin{bmatrix} -1 \\\\ 2 \\\\ 0 \end{bmatrix}$ 로 생성된 2차원 부분공간의 정규직교기저 $\mathcal{C} = (\mathbf{c}_1, \mathbf{c}_2)$ 를 구하시오.

### 2. 단계별 수치 풀이
1. 첫 번째 단위 기저 $\mathbf{c}_1$:
   $$\mathbf{u}_1 = \mathbf{b}_1 = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix}, \quad \Vert\mathbf{u}_1\Vert = \sqrt{1+1+1} = \sqrt{3} \implies \mathbf{c}_1 = \frac{1}{\sqrt{3}} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$

2. 두 번째 직교 벡터 $\mathbf{u}_2$ 및 정규화 $\mathbf{c}_2$:
   $$\mathbf{b}_1^\top \mathbf{b}_2 = 1(-1) + 1(2) + 1(0) = 1$$
   $$\mathbf{u}_2 = \mathbf{b}_2 - \frac{\mathbf{b}_1^\top \mathbf{b}_2}{\Vert\mathbf{b}_1\Vert^2}\mathbf{b}_1 = \begin{bmatrix} -1 \\\\ 2 \\\\ 0 \end{bmatrix} - \frac{1}{3}\begin{bmatrix} 1 \\\\ 1 \\\\ 1 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} -4 \\\\ 5 \\\\ -1 \end{bmatrix}$$
   $$\Vert\mathbf{u}_2\Vert = \frac{1}{3}\sqrt{(-4)^2 + 5^2 + (-1)^2} = \frac{\sqrt{42}}{3} \implies \mathbf{c}_2 = \frac{1}{\sqrt{42}} \begin{bmatrix} -4 \\\\ 5 \\\\ -1 \end{bmatrix}$$

- 검산: $\mathbf{c}_1^\top \mathbf{c}_2 = \frac{1}{\sqrt{126}} (-4 + 5 - 1) = 0$, $\Vert\mathbf{c}_1\Vert = 1, \Vert\mathbf{c}_2\Vert = 1$.


## 📝 [Problem 3.9] 코시-슈바르츠 부등식을 이용한 부등식 증명

### 1. 문제 정의
$x_1, \dots, x_n > 0$ 이며 $\sum_{i=1}^n x_i = 1$ 일 때, 코시-슈바르츠 부등식 $(\mathbf{u}^\top \mathbf{v})^2 \le \Vert\mathbf{u}\Vert^2 \Vert\mathbf{v}\Vert^2$ 을 사용하여 다음을 증명하시오:
- a. $\sum_{i=1}^n x_i^2 \ge \frac{1}{n}$
- b. $\sum_{i=1}^n \frac{1}{x_i} \ge n^2$

### 2. 정밀 증명
- a. 벡터 선택: $\mathbf{u} = [x_1, \dots, x_n]^\top, \mathbf{v} = [1, \dots, 1]^\top$
  $$\left( \sum_{i=1}^n x_i \cdot 1 \right)^2 \le \left( \sum_{i=1}^n x_i^2 \right) \left( \sum_{i=1}^n 1^2 \right)$$
  $$1^2 \le \left( \sum_{i=1}^n x_i^2 \right) \cdot n \implies \sum_{i=1}^n x_i^2 \ge \frac{1}{n} \quad (\text{등호 성립 조건: } x_1 = \dots = x_n = 1/n)$$

- b. 벡터 선택: $\mathbf{u} = [\sqrt{x_1}, \dots, \sqrt{x_n}]^\top, \mathbf{v} = [1/\sqrt{x_1}, \dots, 1/\sqrt{x_n}]^\top$
  $$u_i v_i = \sqrt{x_i} \cdot \frac{1}{\sqrt{x_i}} = 1 \implies \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n 1 = n$$
  $$n^2 = (\mathbf{u}^\top \mathbf{v})^2 \le \left( \sum_{i=1}^n (\sqrt{x_i})^2 \right) \left( \sum_{i=1}^n \left(\frac{1}{\sqrt{x_i}}\right)^2 \right) = \left( \sum_{i=1}^n x_i \right) \left( \sum_{i=1}^n \frac{1}{x_i} \right)$$
  $$\sum_{i=1}^n x_i = 1 \text{ 이므로 } n^2 \le 1 \cdot \sum_{i=1}^n \frac{1}{x_i} \implies \sum_{i=1}^n \frac{1}{x_i} \ge n^2$$


## 📝 [Problem 3.10] $30^\circ$ 2차원 회전 변환 계산

### 1. 문제 정의
$\mathbf{x}_1 = \begin{bmatrix} 2 \\\\ 3 \end{bmatrix}, \mathbf{x}_2 = \begin{bmatrix} 0 \\\\ -1 \end{bmatrix}$ 벡터를 $30^\circ$ 회전 변환하시오.

### 2. 단계별 수치 계산
회전 행렬: $R(30^\circ) = \begin{bmatrix} \cos 30^\circ & -\sin 30^\circ \\\\ \sin 30^\circ & \cos 30^\circ \end{bmatrix} = \frac{1}{2} \begin{bmatrix} \sqrt{3} & -1 \\\\ 1 & \sqrt{3} \end{bmatrix}$

1. $\mathbf{x}_1$ 회전:
   $$R(30^\circ) \mathbf{x}_1 = \frac{1}{2} \begin{bmatrix} \sqrt{3} & -1 \\\\ 1 & \sqrt{3} \end{bmatrix} \begin{bmatrix} 2 \\\\ 3 \end{bmatrix} = \begin{bmatrix} \sqrt{3} - \frac{3}{2} \\\\ 1 + \frac{3\sqrt{3}}{2} \end{bmatrix} \approx \begin{bmatrix} 0.232 \\\\ 3.598 \end{bmatrix}$$

2. $\mathbf{x}_2$ 회전:
   $$R(30^\circ) \mathbf{x}_2 = \frac{1}{2} \begin{bmatrix} \sqrt{3} & -1 \\\\ 1 & \sqrt{3} \end{bmatrix} \begin{bmatrix} 0 \\\\ -1 \end{bmatrix} = \begin{bmatrix} \frac{1}{2} \\\\ -\frac{\sqrt{3}}{2} \end{bmatrix} \approx \begin{bmatrix} 0.500 \\\\ -0.866 \end{bmatrix}$$
