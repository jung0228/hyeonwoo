# 📐 3.3 & 3.4 Lengths, Distances, Angles, and Orthogonality (길이, 거리, 각도 및 직교성)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 3.3 & 3.4 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 3.2절(Inner Products)과의 연결 및 빌드업

우리는 지난 3.2절(Inner Products)에서 벡터 공간 상에 내적(Inner Product) $\langle \mathbf{x}, \mathbf{y} \rangle = \hat{\mathbf{x}}^\top A \hat{\mathbf{y}}$ 및 대칭 양의 정정 행렬(SPD)을 정의했습니다.

3.3절과 3.4절은 내적(Inner Product)이라는 단 하나의 핵심 기준만 정의되면, 공간 안의 모든 기하학적 구조(길이, 거리, 각도, 직교성)가 수식적으로 완벽히 유도되어 완성된다는 것을 보여줍니다.


### 🔗 내적 하나로 완성되는 4가지 핵심 기하학적 개념

1. 내적이 유도하는 벡터의 길이 (Length / Induced Norm)
   - 자기 자신과의 내적에 제곱근을 취하면 벡터의 크기인 노름이 유도됩니다.
   - 수식: $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$

2. 내적이 유도하는 두 벡터 간의 거리 (Distance / Metric)
   - 두 벡터의 차이에 대해 유도된 노름을 적용하면 벡터 간의 물리적 거리가 도출됩니다.
   - 수식: $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$

3. 내적이 유도하는 두 벡터 사이의 각도 (Angle)
   - 코시-슈바르츠 부등식을 적용하여 두 벡터 사이의 기하학적 각도가 유일하게 정의됩니다.
   - 수식: $\cos\omega = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}$

4. 내적이 유도하는 두 벡터의 직교성 (Orthogonality)
   - 내적값이 정확히 0이 될 때 두 벡터는 상호 직교 관계에 놓이게 됩니다.
   - 수식: $\langle \mathbf{x}, \mathbf{y} \rangle = 0 \iff \mathbf{x} \perp \mathbf{y}$


## 1. ⚔️ Section 3.3: Lengths and Distances (내적에 의해 유도되는 길이와 거리)


### 📌 1. 내적이 유도하는 노름 (Induced Norm: Eq 3.16)

모든 내적(Inner Product)은 다음과 같이 벡터의 노름(길이)을 자연스럽게 유도(Induce)합니다:

$$\|\mathbf{x}\| := \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle} \quad (\text{Eq 3.16})$$

- 주의 (Remark): 모든 내적은 노름을 유도하지만, 모든 노름이 내적으로부터 유도되는 것은 아닙니다!
  - 예시: 맨해튼 노름($\ell_1$ norm: $\|\mathbf{x}\|_1 = \sum |x_i|$)은 대응하는 내적이 존재하지 않는 대표적인 노름입니다.


### 📌 2. 코시-슈바르츠 부등식 (Cauchy-Schwarz Inequality: Eq 3.17)

내적 공간 $(V, \langle \cdot, \cdot \rangle)$ 상에서 유도된 노름 $\|\cdot\|$ 은 무조건 다음 코시-슈바르츠 부등식을 만족합니다:

$$|\langle \mathbf{x}, \mathbf{y} \rangle| \le \|\mathbf{x}\| \|\mathbf{y}\| \quad (\text{Eq 3.17})$$

- 기하학적 의미: 두 벡터의 내적 절대값은 각 벡터의 길이를 곱한 것보다 절대 클 수 없습니다.


### 📌 3. 선택하는 내적에 따른 길이의 변화 (Example 3.5 & Eq 3.18~3.20)

동일한 벡터 $\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 이라도 어떤 내적을 채택하느냐에 따라 측정되는 물리적 길이가 달라집니다!

1. 표준 도트 곱 적용 시:
   $$\|\mathbf{x}\| = \sqrt{\mathbf{x}^\top \mathbf{x}} = \sqrt{1^2 + 1^2} = \sqrt{2} \approx 1.414 \quad (\text{Eq 3.18})$$

2. 가중치 행렬 $A = \begin{bmatrix} 1 & -1/2 \\ -1/2 & 1 \end{bmatrix}$ 내적 적용 시:
   $$\langle \mathbf{x}, \mathbf{x} \rangle = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & -1/2 \\ -1/2 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 1 - 1 + 1 = 1 \implies \|\mathbf{x}\| = \sqrt{1} = 1 \quad (\text{Eq 3.20})$$

- 직관적 비교: 동일한 벡터라도 이 내적 공간에서는 도트 곱 공간보다 더 "짧게" 측정됩니다.


### 📌 4. 거리(Distance)와 거리함수(Metric: Definition 3.6 & Eq 3.21~3.23)

내적 공간 $(V, \langle \cdot, \cdot \rangle)$ 상의 두 벡터 $\mathbf{x}, \mathbf{y}$ 사이의 거리 $d(\mathbf{x}, \mathbf{y})$ 는 유도된 노름으로 정의됩니다:

$$d(\mathbf{x}, \mathbf{y}) := \|\mathbf{x} - \mathbf{y}\| = \sqrt{\langle \mathbf{x} - \mathbf{y}, \mathbf{x} - \mathbf{y} \rangle} \quad (\text{Eq 3.21})$$

- 거리함수(Metric)의 3대 공리:
  1. 양의 정정성: $d(\mathbf{x}, \mathbf{y}) \ge 0$ 이며, $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$
  2. 대칭성: $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$
  3. 삼각 부등식: $d(\mathbf{x}, \mathbf{z}) \le d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$

- 내적 vs 거리의 상반된 방향성 (Remark):
  - 두 벡터가 서로 매우 비슷할수록 내적 $\langle \mathbf{x}, \mathbf{y} \rangle$ 은 큰 값(High Similarity)을 가집니다.
  - 반대로 두 벡터가 가까울수록 거리 $d(\mathbf{x}, \mathbf{y})$ 는 0에 가까운 작은 값(Low Distance)을 가집니다.


## 2. ⚔️ Section 3.4: Angles and Orthogonality (각도, 직교성 및 직교행렬)


### 📌 1. 두 벡터 사이의 각도 (Angle: Eq 3.24~3.25 & Example 3.6)

코시-슈바르츠 부등식에 의해 $-1 \le \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|} \le 1$ 이 항상 보장되므로, 두 벡터 사이의 각도 $\omega \in [0, \pi]$ 는 유일하게 결정됩니다:

$$\cos\omega = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|} \quad (\text{Eq 3.25})$$

- Example 3.6 수치 연산: $\mathbf{x} = [1, 1]^\top, \mathbf{y} = [1, 2]^\top$ 에 대한 도트 곱 내적 시:
  $$\cos\omega = \frac{1\cdot 1 + 1\cdot 2}{\sqrt{2} \sqrt{5}} = \frac{3}{\sqrt{10}} \implies \omega = \arccos\left(\frac{3}{\sqrt{10}}\right) \approx 0.32 \text{ rad} \approx 18^\circ$$


### 📌 2. 직교성과 정규직교성 (Orthogonality & Orthonormality: Definition 3.7)

- 직교 (Orthogonal): 두 벡터의 내적값이 0일 때 두 벡터는 직교한다고 정의하며 기호로 $\mathbf{x} \perp \mathbf{y}$ 라 씁니다.
  $$\mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y} \rangle = 0$$

- 정규직교 (Orthonormal): 직교하면서 동시에 각각의 길이가 1인 단위 벡터인 경우입니다.
  $$\mathbf{x} \perp \mathbf{y} \quad \text{and} \quad \|\mathbf{x}\| = 1, \; \|\mathbf{y}\| = 1$$

- 영벡터의 성질: 영벡터 $\mathbf{0}$ 은 공간 상의 모든 벡터와 직교합니다.


### 📌 3. 내적에 따른 직교성의 변화 (Example 3.7 & Eq 3.27~3.28)

$\mathbf{x} = [1, 1]^\top, \mathbf{y} = [-1, 1]^\top$ 두 벡터에 대해:

1. 표준 도트 곱 기준:
   $$\mathbf{x}^\top \mathbf{y} = -1 + 1 = 0 \implies 90^\circ \text{ (직교 성립)}$$

2. 가중치 내적 $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \mathbf{y}$ 기준:
   $$\langle \mathbf{x}, \mathbf{y} \rangle = 1(-1)\cdot 2 + 1(1)\cdot 1 = -2 + 1 = -1 \neq 0 \implies \cos\omega = -\frac{1}{3} \implies \omega \approx 109.5^\circ$$

- 교훈: 어떤 내적 기준에서 직교하던 두 벡터도 다른 내적 기준에서는 직교하지 않을 수 있습니다!


### 📌 4. 직교 행렬 (Orthogonal Matrix: Definition 3.8 & Eq 3.29~3.30)

정방행렬 $A \in \mathbb{R}^{n \times n}$ 의 열벡터들이 서로 정규직교(Orthonormal) 집합을 이룰 때 $A$ 를 직교 행렬이라 부릅니다:

$$A A^\top = I = A^\top A \implies A^{-1} = A^\top \quad (\text{Eq 3.29~3.30})$$

- 핵심 장점: 복잡한 역행렬 계산($O(n^3)$) 대신 단순 전치($A^\top$)만 취하면 $O(1)$ 로 역행렬이 구해집니다.


### 📌 5. 직교 행렬 변환의 2대 불변 성질 (Length & Angle Preservation: Eq 3.31~3.32)

직교 행렬 $A$ 로 벡터 공간을 변환하더라도 길이(Distance)와 각도(Angle)가 100% 보존됩니다:

1. 길이 보존 (Length Preservation):
   $$\|A \mathbf{x}\|_2^2 = (A \mathbf{x})^\top (A \mathbf{x}) = \mathbf{x}^\top A^\top A \mathbf{x} = \mathbf{x}^\top I \mathbf{x} = \mathbf{x}^\top \mathbf{x} = \|\mathbf{x}\|_2^2 \quad (\text{Eq 3.31})$$

2. 각도 보존 (Angle Preservation):
   $$\cos\omega_{new} = \frac{(A \mathbf{x})^\top (A \mathbf{y})}{\|A \mathbf{x}\| \|A \mathbf{y}\|} = \frac{\mathbf{x}^\top A^\top A \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \cos\omega_{orig} \quad (\text{Eq 3.32})$$

- 기하학적 본질: 직교 행렬에 의한 변환은 공간 전체의 형태를 찌그러뜨리지 않는 순수한 회전(Rotation) 및 반사(Flip) 변환입니다.


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 유도된 노름 & 거리: 내적 $\langle \mathbf{x}, \mathbf{x} \rangle$ 으로 유도되는 벡터의 크기 $\|\mathbf{x}\|$ 와 차이 $\|\mathbf{x} - \mathbf{y}\|$.
- 직교 행렬 (Orthogonal Matrix): $A^{-1} = A^\top$ 을 만족하는 행렬로, 공간의 정규직교 기저 축들을 회전/반사 변환시키는 행렬.


### 2️⃣ [2단계 왜 쓰는가?]
- 고차원 데이터 공간에서 데이터 간의 유사도(Cosine Similarity)를 측정하고, 변환 후에도 데이터 간의 거리와 각도가 왜곡되지 않는 안정적인 회전 변환을 수행하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 거리(Metric) vs 내적(Inner Product)의 역방향 직관:
  - 내적값 $\langle \mathbf{x}, \mathbf{y} \rangle$ 이 크다 ➡️ 두 벡터의 방향이 비슷하다 (유사도 High)
  - 거리 $d(\mathbf{x}, \mathbf{y})$ 가 작다 ➡️ 두 벡터의 위치가 가깝다 (거리 Low)
- 직교 행렬의 역행렬 계산 효율:
  - 일반 행렬의 역행렬 연산 복잡도는 $O(n^3)$ 으로 매우 비싸지만, 직교 행렬은 전치 행렬 $A^\top$ 만 취하면 되므로 $O(1)$ 복잡도로 완벽한 역행렬 연산이 가능합니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- 코사인 유사도 (Cosine Similarity - NLP/추천시스템): 텍스트 임베딩 벡터 간의 유사도를 측정할 때 내적 공식 $\cos\omega = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$ 을 그대로 사용합니다.
- PCA 및 SVD (Ch 4.5 & Ch 10): 특이값 분해 $X = U \Sigma V^\top$ 에서 우측/좌측 특이 벡터 행렬 $U, V$ 가 모두 직교 행렬이므로, 데이터의 거리를 훼손하지 않는 주성분 회전 변환이 가능해집니다.
