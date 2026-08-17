# 📐 3.5 & 3.6 Orthonormal Basis and Orthogonal Complement (정규직교기저와 직교 여공간)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 3.5 & 3.6 전수 분석 & 4단계 정밀 해설 노트


## 🌐 0. 3.3/3.4절과의 연결 및 자연스러운 빌드업: 왜 "정규직교기저"와 "직교 여공간"으로 확장하는가?

우리는 지난 3.3절과 3.4절에서 내적(Inner Product)을 통해 벡터의 길이(Length), 두 벡터 간의 거리(Distance), 그리고 각도(Angle)와 직교성(Orthogonality)을 엄밀하게 정의했습니다.

이제 자연스럽게 떠오르는 다음 질문은 이것입니다:
"공간 전체를 생성하는 기저(Basis) 축들을 잡을 때, 축들끼리 서로 완벽히 90도로 직교하고 각 축의 길이마저 깔끔하게 1로 정규화되어 있다면 얼마나 계산이 단순해질까?"

이 질문의 답이 바로 3.5절의 정규직교기저(Orthonormal Basis, ONB)입니다.

나아가 3.6절에서는 어떤 부분공간 $U$ 가 주어졌을 때 그 부분공간과 완전히 90도로 직교하는 나머지 보완 공간인 직교 여공간(Orthogonal Complement, $U^\perp$)을 정의함으로써, 고차원 데이터를 중요 공간과 잔차(노이즈) 공간으로 완벽히 수직 분해하는 기하학적 토대를 완성합니다.


## 1. ⚔️ Section 3.5: Orthonormal Basis (정규직교기저)


### 📌 1. 정규직교기저의 엄밀한 수학적 정의 (Definition 3.9 & Eq 3.33~3.34)

$n$차원 벡터 공간 $V$ 와 기저 집합 $\{\mathbf{b}_1, \dots, \mathbf{b}_n\}$ 에 대해, 모든 $i, j = 1, \dots, n$ 에서 다음 두 조건이 만족될 때 이 기저를 정규직교기저(Orthonormal Basis, ONB)라고 부릅니다:

1. 직교성 조건 (Orthogonality: Eq 3.33)
   $$\langle \mathbf{b}_i, \mathbf{b}_j \rangle = 0 \quad (i \neq j)$$

2. 정규화 조건 (Normalization: Eq 3.34)
   $$\langle \mathbf{b}_i, \mathbf{b}_i \rangle = 1 \iff \|\mathbf{b}_i\| = 1$$

- 직교 기저(Orthogonal Basis): 직교성 조건(Eq 3.33)만 만족하고 길이가 1이 아닌 기저를 의미합니다.
- 정규직교기저(ONB): 직교성과 정규화 조건(Eq 3.33과 3.34)을 동시에 모두 만족하여, 모든 기저 벡터가 서로 수직이면서 길이가 정확히 1인 단위 벡터들로 구성된 기저입니다.


### 📌 2. 그람-슈미트 직교화 과정 (Gram-Schmidt Process: Strang 2003)

임의의 정규화되지 않고 삐뚤어진 기저 집합 $\{\tilde{\mathbf{b}}_1, \dots, \tilde{\mathbf{b}}_n\}$ 이 주어졌을 때, 순차적으로 직교하는 성분만 남겨서 완벽한 정규직교기저 $\{\mathbf{b}_1, \dots, \mathbf{b}_n\}$ 를 만들어내는 생성적 알고리즘입니다.


#### 💡 [그람-슈미트 3단계 작동 원리 알고리즘]

1. 1단계: 첫 번째 벡터 정규화
   - 첫 번째 벡터는 방향을 그대로 유지한 채 단위 길이로 만듭니다:
     $$\mathbf{u}_1 = \tilde{\mathbf{b}}_1, \quad \mathbf{b}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}$$

2. 2단계: 이전 기저로의 정사영(그림자) 성분 빼기 (직교화)
   - 두 번째 벡터에서 첫 번째 기저 $\mathbf{b}_1$ 방향으로 드리운 그림자 성분을 빼버리면, $\mathbf{b}_1$ 과 완벽히 수직인 순수 잔차 성분 $\mathbf{u}_2$ 만 남습니다:
     $$\mathbf{u}_2 = \tilde{\mathbf{b}}_2 - \langle \tilde{\mathbf{b}}_2, \mathbf{b}_1 \rangle \mathbf{b}_1, \quad \mathbf{b}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|}$$

3. $k$단계: 일반화 점화식
   - $k$번째 벡터 $\tilde{\mathbf{b}}_k$ 에서 이전에 구해둔 모든 정규직교기저들($\mathbf{b}_1, \dots, \mathbf{b}_{k-1}$)로의 그림자 성분을 전부 빼서 직교 벡터 $\mathbf{u}_k$ 를 구한 뒤 정규화합니다:
     $$\mathbf{u}_k = \tilde{\mathbf{b}}_k - \sum_{j=1}^{k-1} \langle \tilde{\mathbf{b}}_k, \mathbf{b}_j \rangle \mathbf{b}_j, \quad \mathbf{b}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$


#### 💡 [손으로 직접 푸는 2차원 수치 예제]

삐뚤어진 두 기저 벡터 $\tilde{\mathbf{b}}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \tilde{\mathbf{b}}_2 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ 에 표준 도트 곱 내적을 적용하여 정규직교기저를 만들어 봅시다!

##### 1단계: 첫 번째 정규직교기저 $\mathbf{b}_1$ 구하기
- $\mathbf{u}_1 = \tilde{\mathbf{b}}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- 크기(노름): $\|\mathbf{u}_1\| = \sqrt{1^2 + 1^2} = \sqrt{2}$
- 정규화된 첫 번째 기저:
  $$\mathbf{b}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

##### 2단계: 두 번째 벡터에서 $\mathbf{b}_1$ 방향 그림자 빼기 ($\mathbf{u}_2$ 구하기)
- 그림자 계수(내적): 
  $$\langle \tilde{\mathbf{b}}_2, \mathbf{b}_1 \rangle = \begin{bmatrix} 1 & 2 \end{bmatrix} \left( \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right) = \frac{1 + 2}{\sqrt{2}} = \frac{3}{\sqrt{2}}$$
- $\mathbf{b}_1$ 방향 그림자 벡터:
  $$\langle \tilde{\mathbf{b}}_2, \mathbf{b}_1 \rangle \mathbf{b}_1 = \frac{3}{\sqrt{2}} \left( \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right) = \frac{3}{2} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$$
- 그림자를 뺀 순수 수직 잔차 $\mathbf{u}_2$:
  $$\mathbf{u}_2 = \tilde{\mathbf{b}}_2 - \langle \tilde{\mathbf{b}}_2, \mathbf{b}_1 \rangle \mathbf{b}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ 0.5 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

##### 3단계: 두 번째 정규직교기저 $\mathbf{b}_2$ 정규화
- 크기(노름): $\|\mathbf{u}_2\| = \sqrt{(-0.5)^2 + 0.5^2} = \sqrt{0.5} = \frac{1}{\sqrt{2}}$
- 정규화된 두 번째 기저:
  $$\mathbf{b}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|} = \frac{\frac{1}{2} \begin{bmatrix} -1 \\ 1 \end{bmatrix}}{\frac{1}{\sqrt{2}}} = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

##### 🎯 직교성 및 정규성 최종 검산
- 직교성: $\mathbf{b}_1^\top \mathbf{b}_2 = \left(\frac{1}{\sqrt{2}}\right)\left(\frac{1}{\sqrt{2}}\right) (1\cdot(-1) + 1\cdot 1) = \frac{1}{2}(-1 + 1) = \mathbf{0}$ (완벽 수직!)
- 정규성: $\|\mathbf{b}_1\| = 1, \|\mathbf{b}_2\| = 1$ (완벽 단위 길이!)
- 이렇게 삐뚤어져 있던 기저가 서로 90도로 수직인 완벽한 정규직교기저(ONB)로 변환되었습니다!


### 📌 3. 정규직교기저의 대표 예시 (Example 3.8 & Eq 3.35)

1. 표준 유클리드 공간 $\mathbb{R}^n$ 의 표준기저(Canonical Basis)
   - 표준기저 $\mathbf{e}_1, \dots, \mathbf{e}_n$ 은 도트 곱 내적에 대해 가장 대표적인 정규직교기저입니다.

2. $\mathbb{R}^2$ 공간의 45도 회전 정규직교기저 (Eq 3.35)
   $$\mathbf{b}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad \mathbf{b}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$
   - 내적 확인: $\mathbf{b}_1^\top \mathbf{b}_2 = \frac{1}{2}(1 - 1) = 0$ (서로 수직 직교)
   - 길이 확인: $\|\mathbf{b}_1\| = \sqrt{\frac{1}{2} + \frac{1}{2}} = 1$, $\|\mathbf{b}_2\| = \sqrt{\frac{1}{2} + \frac{1}{2}} = 1$ (단위 길이)
   - 따라서 두 벡터는 $\mathbb{R}^2$ 의 훌륭한 정규직교기저를 형성합니다.


## 2. ⚔️ Section 3.6: Orthogonal Complement (직교 여공간)


### 📌 1. 직교 여공간의 정의 (Orthogonal Complement: $U^\perp$)

$D$차원 벡터 공간 $V$ 와 $M$차원 선형 부분공간 $U \subseteq V$ 에 대해, $U$ 안의 모든 벡터와 직교하는 $V$ 안의 모든 벡터들을 모아놓은 부분집합을 $U$ 의 직교 여공간(Orthogonal Complement)이라 부르며 $U^\perp$ 로 표기합니다.

- 차원 관계: $U^\perp$ 의 차원은 정확히 $(D - M)$ 차원이 됩니다.
  $$\text{dim}(U) + \text{dim}(U^\perp) = M + (D - M) = D = \text{dim}(V)$$

- 영벡터 교집합 성질: $U$ 와 $U^\perp$ 가 공유하는 공통 원소는 오직 영벡터 하나뿐입니다.
  $$U \cap U^\perp = \{\mathbf{0}\}$$


### 📌 2. 공간의 유일한 직교 분해 (Unique Orthogonal Decomposition: Eq 3.36)

$U$ 의 기저를 $(\mathbf{b}_1, \dots, \mathbf{b}_M)$ 이라 하고, $U^\perp$ 의 기저를 $(\mathbf{b}_1^\perp, \dots, \mathbf{b}_{D-M}^\perp)$ 라 하면, 전체 공간 $V$ 안의 임의의 모든 벡터 $\mathbf{x} \in V$ 는 다음과 같이 $U$ 성분과 $U^\perp$ 성분의 합으로 오직 유일하게 분해됩니다:

$$\mathbf{x} = \sum_{m=1}^M \lambda_m \mathbf{b}_m + \sum_{j=1}^{D-M} \psi_j \mathbf{b}_j^\perp \quad (\lambda_m, \psi_j \in \mathbb{R}, \text{ Eq 3.36})$$

- 앞부분 $\sum_{m=1}^M \lambda_m \mathbf{b}_m$: 부분공간 $U$ 위로 내려앉은 직교 정사영 성분
- 뒷부분 $\sum_{j=1}^{D-M} \psi_j \mathbf{b}_j^\perp$: $U$ 와 90도를 이루며 튕겨 나간 수직 잔차(오차) 성분


### 📌 3. 법선 벡터(Normal Vector)를 이용한 평면 및 초평면 기술 (Figure 3.7)

3차원 공간 $\mathbb{R}^3$ ($D=3$) 에서 원점을 지나는 2차원 평면 $U$ ($M=2$) 가 주어졌을 때:

- 직교 여공간 $U^\perp$ 는 $3 - 2 = 1$차원의 직선 공간이 됩니다.
- 이때 $U^\perp$ 를 생성하는 길이 1인 단위 기저 벡터 $\mathbf{w}$ ($\|\mathbf{w}\| = 1$) 를 평면 $U$ 의 법선 벡터(Normal Vector)라고 부릅니다.
- 법선 벡터 $\mathbf{w}$ 와 내적해서 0이 되는 모든 점들의 모임이 곧 평면 $U$ 가 됩니다:
  $$U = \{\mathbf{x} \in \mathbb{R}^3 \mid \langle \mathbf{w}, \mathbf{x} \rangle = 0\}$$

- 확장: 일반적인 $n$차원 벡터 공간이나 어파인 공간에서 $(n-1)$차원 초평면(Hyperplane)을 수식으로 간결하게 정의할 때 직교 여공간의 법선 벡터가 핵심적으로 사용됩니다.


## 🧠 3. 4단계 정밀 개념 해설


### 1️⃣ [1단계 개념 정의]
- 정규직교기저 (ONB): 서로 직교하면서 길이가 모두 1인 가장 이상적인 직교 좌표계 기저 축 모음입니다.
- 직교 여공간 ($U^\perp$): 주어진 부분공간 $U$ 에 완벽히 수직인 나머지 공간이며, 전체 공간 $V$ 를 $V = U \oplus U^\perp$ 로 직합 분해합니다.


### 2️⃣ [2단계 왜 쓰는가?]
- 좌표 계산의 압도적 단순화: 정규직교기저 하에서는 임의의 벡터 $\mathbf{x}$ 의 좌표 계수를 구할 때 복잡한 연립방정식 역행렬을 풀 필요 없이, 단순히 기저와의 내적 $\lambda_i = \langle \mathbf{x}, \mathbf{b}_i \rangle$ 딱 한 번으로 즉시 계산됩니다.
- 노이즈와 신호의 완벽한 분리: 데이터를 주요 정보 부분공간 $U$ 와 수직 오차 공간 $U^\perp$ 로 찌그러짐 없이 분리하기 위해 사용합니다.


### 3️⃣ [3단계 상황별 직관 & Trade-off]
- 비직교 기저 vs 정규직교기저(ONB)의 좌표 계산 복잡도 차이:
  - 비직교 기저에서는 기저가 바뀔 때마다 $O(n^3)$ 의 행렬 반전 연산이 필요합니다.
  - 정규직교기저에서는 기저 행렬이 직교 행렬($Q^\top Q = I$)이 되므로, 전치 행렬 곱셈 $O(n^2)$ 만으로 모든 좌표 변환과 투영이 즉각 완료됩니다.


### 4️⃣ [4단계 실전 AI 연결고리]
- PCA (주성분 분석 - Ch 10): 고차원 데이터를 가장 분산이 큰 $M$차원 정규직교기저 주성분 공간 $U$ 로 투영하고, 버려지는 직교 여공간 $U^\perp$ 의 크기를 최소화하여 정보 손실을 최소화하는 알고리즘입니다.
- SVM (서포트 벡터 머신 - Ch 12): 데이터를 이진 분류하는 초평면(Decision Boundary)을 정의할 때, 경계면의 방향을 결정하는 가중치 벡터 $\mathbf{w}$ 가 바로 초평면의 법선 벡터(Normal Vector) 역할을 수행합니다.
- 선형 회귀 (Linear Regression - Ch 9): 타겟 데이터 $\mathbf{y}$ 를 특성 행렬의 열공간 $U = \text{Col}(X)$ 위로 직교 정사영시키고, 예측 오차 $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ 가 열공간의 직교 여공간 $U^\perp$ 에 놓이도록 가중치를 최적화합니다.
