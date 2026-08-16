# 📐 2.7 Linear Mappings (선형사상과 기저변환)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.7 전수 분석 & 4단계 정밀 해설 노트

## 🌐 0. 지난 노트(2.6절)와의 연결 및 빌드업: 왜 "선형사상"과 "기저변환"을 배우는가?

우리는 지난 2.6절에서 벡터 공간의 최소 뼈대인 기저(Basis)와 공간의 알짜배기 크기인 차원(Dimension) 및 행렬의 계수(Rank)를 배웠습니다.

이제 2.7절에서는 공간을 가만히 두지 않고 한 벡터 공간 $V$ 에서 다른 벡터 공간 $W$ 로 벡터들을 변형시키고 이동시키는 규칙(함수)을 다룹니다.
이것이 바로 선형사상(Linear Mapping) 또는 선형변환(Linear Transformation)입니다.

또한, "동일한 선형변환이라도 관찰하는 기저(좌표축)를 바꾸면 행렬의 모양이 어떻게 일목요원하게 달라지는가?" 를 다루는 기저변환(Basis Change)과, 
"변환 과정에서 0으로 찌그러져 사라지는 차원과 살아서 이동하는 차원의 보존 법칙"인 Rank-Nullity 정리를 교재의 모든 예제와 수식을 포함하여 완벽 해부합니다.

## 1. ⚔️ Section 2.7: Linear Mappings (선형사상의 분류와 주요 정리)

### 📌 1. 선형사상(Linear Mapping)의 수학적 정의 (Definition 2.15)
두 실수 벡터 공간 $V, W$ 에 대해 사상(Mapping) $\Phi : V \to W$ 가 모든 $\mathbf{x}, \mathbf{y} \in V$ 와 실수 스칼라 $\lambda, \psi \in \mathbb{R}$ 에 대해 다음을 만족하면 선형사상(Linear Mapping / Homomorphism / Linear Transformation)이라 부릅니다:

$$\Phi(\lambda \mathbf{x} + \psi \mathbf{y}) = \lambda \Phi(\mathbf{x}) + \psi \Phi(\mathbf{y}) \quad (\text{Eq 2.87})$$

- 중합의 원리 (Superposition Principle): 덧셈 보존 $\Phi(\mathbf{x}+\mathbf{y}) = \Phi(\mathbf{x}) + \Phi(\mathbf{y})$ 과 스칼라배 보존 $\Phi(\lambda \mathbf{x}) = \lambda \Phi(\mathbf{x})$ 가 동시에 성립함을 의미합니다.

### 📌 2. 단사(Injective), 전사(Surjective), 전단사(Bijective) 정의 (Definition 2.16)
- 단사 (Injective / One-to-One): $\forall \mathbf{x}, \mathbf{y} \in V : \Phi(\mathbf{x}) = \Phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$. (서로 다른 입력은 무조건 서로 다른 출력으로 매핑됨).
- 전사 (Surjective / Onto): $\Phi(V) = W$. (공역 $W$ 의 모든 원소가 적어도 하나의 화살을 맞음).
- 전단사 (Bijective): 단사이면서 동시에 전사인 경우 (1:1 완전 대응 역변환 $\Phi^{-1}$ 존재).

### 📌 3. 선형사상의 4가지 분류 체계 (Special Mappings)
1. 동형사상 (Isomorphism): $\Phi : V \to W$ 가 선형이면서 전단사(Bijective)인 사상.
2. 단형사상 (Endomorphism): $\Phi : V \to V$ 자기 자신으로 가는 선형사상.
3. 자기도형사상 (Automorphism): $\Phi : V \to V$ 자기 자신으로 가면서 전단사(Bijective)인 선형사상.
4. 항등사상 (Identity Mapping): $\text{id}_V : V \to V, \mathbf{x} \mapsto \mathbf{x}$.

### 📌 4. MML 원문 예제 백지 해부 (Example 2.19: 복소수 동형사상)
$$\Phi : \mathbb{R}^2 \to \mathbb{C}, \quad \Phi(\mathbf{x}) = x_1 + i x_2$$

$$\Phi\left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \right) = (x_1+y_1) + i(x_2+y_2) = (x_1+ix_2) + (y_1+iy_2) = \Phi\left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) + \Phi\left( \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} \right)$$

$$\Phi\left( \lambda \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \lambda x_1 + i \lambda x_2 = \lambda(x_1 + ix_2) = \lambda \Phi\left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right)$$

- 정리 2.17 (Theorem 2.17 - Axler 2015): 유한차원 벡터 공간 $V$ 와 $W$ 가 동형사상(Isomorphic)일 필요충분조건은 $\text{dim}(V) = \text{dim}(W)$ 입니다. 이 정리에 의해 차원이 같은 행렬 공간 $\mathbb{R}^{m \times n}$ 과 벡터 공간 $\mathbb{R}^{mn}$ 은 수학적으로 동일 취급이 가능합니다.

## 2. ⚔️ Section 2.7.1: Matrix Representation of Linear Mappings (선형사상의 행렬 표현)

### 📌 1. 좌표 및 좌표벡터 (Definition 2.18 & Eq 2.90~2.91)
$n$차원 공간 $V$ 의 순서기저 $\mathcal{B} = (\mathbf{b}_1, \dots, \mathbf{b}_n)$ 에 대해 모든 벡터 $\mathbf{x} \in V$ 는 다음과 같이 유일하게 선형결합됩니다:

$$\mathbf{x} = \alpha_1 \mathbf{b}_1 + \dots + \alpha_n \mathbf{b}_n$$

이때 스칼라 열벡터 $\boldsymbol{\alpha} = [\alpha_1, \dots, \alpha_n]^\top \in \mathbb{R}^n$ 을 기저 $\mathcal{B}$ 에 대한 $\mathbf{x}$ 의 좌표벡터(Coordinate Vector)라 부릅니다.

### 📌 2. 원문 기저 변환 좌표 예제 (Example 2.20 & Figure 2.8, 2.9)
$\mathbb{R}^2$ 표준기저 $(e_1, e_2)$ 에서 좌표 $[2, 3]^\top$ 인 벡터 $\mathbf{x} = 2e_1 + 3e_2$ 에 대해:
새로운 기저 $b_1 = [1, -1]^\top, b_2 = [1, 1]^\top$ 를 잡으면 동일한 벡터 $\mathbf{x}$ 는 다음과 같이 좌표가 재표현됩니다:

$$\mathbf{x} = -\frac{1}{2} b_1 + \frac{5}{2} b_2 \implies \text{새로운 좌표벡터: } \frac{1}{2} [-1, 5]^\top$$

### 📌 3. 변환 행렬 (Transformation Matrix: Definition 2.19 & Eq 2.92~2.94)
선형사상 $\Phi : V \to W$ 와 기저 $\mathcal{B} = (\mathbf{b}_1, \dots, \mathbf{b}_n)$, $\mathcal{C} = (\mathbf{c}_1, \dots, \mathbf{c}_m)$ 에 대해, 변환 행렬 $A_\Phi \in \mathbb{R}^{m \times n}$ 의 $j$번째 열벡터는 기저 벡터 $\Phi(\mathbf{b}_j)$ 의 $\mathcal{C}$ 기준 좌표벡터입니다:

$$\hat{\mathbf{y}} = A_\Phi \hat{\mathbf{x}} \quad (\text{Eq 2.94})$$

#### 💡 [쉬운 언어로 풀어쓴 변환 행렬 $A_\Phi$ 의 직관적 본질]
이 문장과 수식의 진짜 의미는 다음과 같습니다:

1. "기저 벡터들이 어디로 이동했는지 그 결과표를 열(Column)로 적어놓은 행렬":
   출발지 공간 $V$ 의 뼈대(기저 $\mathbf{b}_1, \dots, \mathbf{b}_n$)들이 이동 규칙 $\Phi$ 를 타고 도착지 공간 $W$ 로 건너갔을 때, 도착지의 뼈대($\mathbf{c}_1, \dots, \mathbf{c}_m$)를 기준으로 어디에 떨어졌는지 그 위치 숫자를 열(Column)로 하나씩 세워놓은 결과표가 바로 변환 행렬 $A_\Phi$ 입니다.
2. 왜 쓰는가?:
   출발지 공간에는 무수히 많은 무한개의 벡터가 살고 있어서 일일이 이동 위치를 계산할 수 없습니다. 하지만 공간의 뼈대인 기저 벡터들($\mathbf{b}_1, \dots, \mathbf{b}_n$)이 어디로 가는지 그 착륙 지점만 알면, 세상 모든 벡터의 이동 결과는 단순한 행렬 곱셈 $\hat{\mathbf{y}} = A_\Phi \hat{\mathbf{x}}$ 딱 한 번으로 자동 계산됩니다!
3. 딥러닝 Linear Layer ($Y = W X$) 와의 연결:
   인공신경망의 가장 기본 레이어인 `nn.Linear(in_features=4, out_features=2)` 의 가중치 행렬 $W$ ($2 \times 4$ 행렬)는 입력 4차원 공간의 뼈대 4개가 2차원 출력 공간으로 건너갔을 때의 착륙 지점 좌표를 4개의 열(Column)로 적어놓은 변환 행렬 $A_\Phi$ 바로 그 자체입니다!

- Example 2.21 수치 계산 (Eq 2.95~2.96):
  $$\Phi(\mathbf{b}_1) = \mathbf{c}_1 - \mathbf{c}_2 + 3\mathbf{c}_3 - \mathbf{c}_4, \quad \Phi(\mathbf{b}_2) = 2\mathbf{c}_1 + \mathbf{c}_2 + 7\mathbf{c}_3 + 2\mathbf{c}_4, \quad \Phi(\mathbf{b}_3) = 3\mathbf{c}_2 + \mathbf{c}_3 + 4\mathbf{c}_4$$
  $$A_\Phi = \begin{bmatrix} \mathbf{1} & \mathbf{2} & \mathbf{0} \\ \mathbf{-1} & \mathbf{1} & \mathbf{3} \\ \mathbf{3} & \mathbf{7} & \mathbf{1} \\ \mathbf{-1} & \mathbf{2} & \mathbf{4} \end{bmatrix}$$

- Example 2.22 (2차원 벡터 변환 행렬 기하학: Eq 2.97 & Figure 2.10):
  - 회전 행렬 $A_1 = \begin{bmatrix} \cos(\pi/4) & -\sin(\pi/4) \\ \sin(\pi/4) & \cos(\pi/4) \end{bmatrix}$ (45도 회전)
  - 늘리기 행렬 $A_2 = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$ (수평축 2배 신장)
  - 합성 행렬 $A_3 = \frac{1}{2} \begin{bmatrix} 3 & -1 \\ 1 & -1 \end{bmatrix}$ (반사, 회전, 신장의 복합 작용)

## 3. ⚔️ Section 2.7.2: Basis Change (기저변환 상세 유도)

### 📌 1. 기저변환 정리 및 증명 (Theorem 2.20 & Proof Eq 2.105~2.112)
기저 $\mathcal{B}, \mathcal{C}$ 에서 기저 $\tilde{\mathcal{B}}, \tilde{\mathcal{C}}$ 로 바뀔 때 변환 행렬의 관계:

$$\tilde{A}_\Phi = T^{-1} A_\Phi S \quad (\text{Eq 2.105})$$

#### 💡 [실전 해설 1] 기저변환 행렬 $S$ 와 $T$ 는 대체 어떻게 구하며 무슨 뜻인가?
- $S$ 행렬 구하기 (출발 공간 $V$):
  출발 공간 $V$ 의 새로운 기저 벡터 $\tilde{\mathcal{B}} = (\tilde{\mathbf{b}}_1, \dots, \tilde{\mathbf{b}}_n)$ 들을 옛날 기저($\mathcal{B}$) 관점의 좌표로 써서 열(Column)로 그대로 세워 나열하면 $S$ 행렬이 완결됩니다:
  $$S = \begin{bmatrix} \mid & \mid & \mid \\ \tilde{\mathbf{b}}_1 & \dots & \tilde{\mathbf{b}}_n \\ \mid & \mid & \mid \end{bmatrix}$$
- $T$ 행렬 구하기 (도착 공간 $W$):
  도착 공간 $W$ 의 새로운 기저 벡터 $\tilde{\mathcal{C}} = (\tilde{\mathbf{c}}_1, \dots, \tilde{\mathbf{c}}_m)$ 들을 옛날 기저($\mathcal{C}$) 관점의 좌표로 써서 열(Column)로 그대로 세워 나열하면 $T$ 행렬이 완결됩니다:
  $$T = \begin{bmatrix} \mid & \mid & \mid \\ \tilde{\mathbf{c}}_1 & \dots & \tilde{\mathbf{c}}_m \\ \mid & \mid & \mid \end{bmatrix}$$

- "옛날 기저 관점의 좌표로 쓴다"는 것의 진짜 의미:
  우리가 평소에 사물을 보고 종이에 적는 숫자 $\tilde{\mathbf{b}}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 은 이미 표준기저(옛날 기저 $e_1, e_2$) 기준으로 $1 e_1 + 1 e_2$ 라고 읽어낸 좌표입니다!
  따라서 옛날 기저가 표준기저일 때에는 우리가 보고 적은 새 기저 벡터 숫자 $[1, 1]^\top, [1, -1]^\top$ 그 자체를 열로 세워놓기만 하면 곧바로 $S = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$ 번역기가 완성됩니다.
  (만약 옛날 기저가 표준기저가 아니라 찌그러진 축 $\mathbf{b}_1, \mathbf{b}_2$ 라면, $\tilde{\mathbf{b}}_1 = c_1 \mathbf{b}_1 + c_2 \mathbf{b}_2$ 의 계수 $[c_1, c_2]^\top$ 를 구해서 세워야 합니다).

- $S$ 행렬의 수치적 작동 검증 예시:
  새 기저 $\tilde{\mathbf{b}}_1 = [1, 1]^\top, \tilde{\mathbf{b}}_2 = [1, -1]^\top$ 일 때 $S = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$ 가 됩니다.
  새 좌표계 기준으로 "1번 축 2칸, 2번 축 3칸"($\hat{\mathbf{x}}_{\tilde{\mathcal{B}}} = [2, 3]^\top$) 에 서 있다는 영희의 위치를 $S$ 에 곱해봅시다:
  $$S \hat{\mathbf{x}}_{\tilde{\mathcal{B}}} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 2+3 \\ 2-3 \end{bmatrix} = \begin{bmatrix} 5 \\ -1 \end{bmatrix}$$
  실제 위치 $2 \tilde{\mathbf{b}}_1 + 3 \tilde{\mathbf{b}}_2 = (5, -1)$ 과 정확히 일치합니다! 즉 $S$ 는 새로운 눈으로 말한 좌표를 옛날 눈이 알아듣는 실제 좌표로 해석해 주는 번역기입니다.

#### 💡 [실전 해설 2] 왜 하필 $T$ 에만 역행렬($T^{-1}$)을 곱하는가? (환율/무역 비유)
이것은 데이터가 이동하는 화살표 경로(경로 합성)의 방향 때문입니다!

- 비유: 한국 원화($\tilde{\mathcal{B}}$)를 들고 미국 달러 상품($\tilde{\mathcal{C}}$)을 사고 싶은데, 우리에겐 엔화($\mathcal{B}$)를 엔화 상품($\mathcal{C}$)으로 바꿔주는 오래된 기계($A_\Phi$)만 있는 상황.
1. 1단계 ($S$ 곱하기): 원화($\tilde{\mathcal{B}}$)를 엔화($\mathcal{B}$)로 환전합니다. ($S$ 는 원래 '원화 ➡️ 엔화' 표이므로 방향 일치 ➡️ 그냥 $S$ 곱함).
2. 2단계 ($A_\Phi$ 곱하기): 엔화($\mathcal{B}$)를 기계에 넣어 엔화 상품($\mathcal{C}$)을 받습니다.
3. 3단계 ($T^{-1}$ 곱하기): 손에 든 엔화 상품($\mathcal{C}$)을 달러 상품($\tilde{\mathcal{C}}$)으로 교환해야 합니다!
   준비된 협정표 $T$ 는 원래 '달러 상품 ➡️ 엔화 상품' 방향으로 정의되어 있기 때문에, 반대 방향인 '엔화 상품 ➡️ 달러 상품' 으로 바꾸려면 반드시 역표인 역행렬 $T^{-1}$ 을 곱해야만 최종 달러 상품($\tilde{\mathcal{C}}$)을 얻을 수 있습니다!

- 증명 핵심 요약 (Proof Eq 2.108~2.110):
  $$\Phi(\tilde{\mathbf{b}}_j) = \sum_{k=1}^m \tilde{a}_{kj} \tilde{\mathbf{c}}_k = \sum_{l=1}^m \left( \sum_{k=1}^m t_{lk} \tilde{a}_{kj} \right) \mathbf{c}_l$$
  $$\Phi(\tilde{\mathbf{b}}_j) = \Phi\left(\sum_{i=1}^n s_{ij} \mathbf{b}_i\right) = \sum_{l=1}^m \left( \sum_{i=1}^n a_{li} s_{ij} \right) \mathbf{c}_l$$
  계수를 비교하면 $T \tilde{A}_\Phi = A_\Phi S \implies \tilde{A}_\Phi = T^{-1} A_\Phi S$.

### 📌 2. 행렬의 동등성(Equivalence)과 닮음(Similarity) 정의 (Definition 2.21 & 2.22)
- 동등성 (Equivalence): 정칙행렬 $S, T$ 에 대해 $\tilde{A} = T^{-1} A S$ 만족.
- 닮음 (Similarity): 정칙행렬 $S$ 에 대해 $\tilde{A} = S^{-1} A S$ 만족 (동형사상 $V \to V$).

### 📌 3. MML 원문 기저변환 실전 계산 (Example 2.24 & Eq 2.117~2.121)
$A_\Phi = \begin{bmatrix} 1 & 2 & 0 \\ -1 & 1 & 3 \\ 3 & 7 & 1 \\ -1 & 2 & 4 \end{bmatrix}$, 새 기저 $\tilde{\mathcal{B}}, \tilde{\mathcal{C}}$ 에 대한 기저변환 행렬 $S, T$:

$$S = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}, \quad T = \begin{bmatrix} 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$\tilde{A}_\Phi = T^{-1} A_\Phi S = \begin{bmatrix} -4 & -4 & -2 \\ 6 & 0 & 0 \\ 4 & 8 & 4 \\ 1 & 6 & 3 \end{bmatrix}$$

## 4. ⚔️ Section 2.7.3: Image and Kernel (상과 영공간 및 실전 예제)

### 📌 1. 정의 및 성질 (Definition 2.23 & Remark p.58~59)
- 영공간 (Kernel / Null Space: Eq 2.122): $\text{ker}(\Phi) := \{ \mathbf{v} \in V \mid \Phi(\mathbf{v}) = \mathbf{0}_W \}$. $\text{ker}(\Phi) = \{\mathbf{0}\}$ 일 필요충분조건은 $\Phi$ 가 단사(Injective)인 것입니다.
- 상 (Image / Range / Column Space: Eq 2.123): $\text{Im}(\Phi) := \{ \mathbf{w} \in W \mid \exists \mathbf{v} \in V : \Phi(\mathbf{v}) = \mathbf{w} \} = \text{span}[\mathbf{a}_1, \dots, \mathbf{a}_n]$.

### 📌 2. MML 원문 영공간/상 구하기 예제 백지 분석 (Example 2.25 & Eq 2.125~2.128)
$$\Phi : \mathbb{R}^4 \to \mathbb{R}^2, \quad \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} \mapsto \begin{bmatrix} 1 & 2 & -1 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = \begin{bmatrix} x_1 + 2x_2 - x_3 \\ x_1 + x_4 \end{bmatrix}$$

1. 상 $\text{Im}(\Phi)$ 계산:
   $$\text{Im}(\Phi) = \text{span}\left[ \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ 0 \end{bmatrix}, \begin{bmatrix} -1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix} \right] = \mathbb{R}^2$$

2. 영공간 $\text{ker}(\Phi)$ 계산 ($A\mathbf{x} = \mathbf{0}$ RREF 변환):
   $$\begin{bmatrix} 1 & 2 & -1 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & -1/2 & -1/2 \end{bmatrix}$$
   - 비피벗 열 $\mathbf{a}_3 = -\frac{1}{2}\mathbf{a}_2 \implies \mathbf{a}_3 + \frac{1}{2}\mathbf{a}_2 = \mathbf{0}$
   - 비피벗 열 $\mathbf{a}_4 = \mathbf{a}_1 - \frac{1}{2}\mathbf{a}_2 \implies \mathbf{a}_1 - \frac{1}{2}\mathbf{a}_2 - \mathbf{a}_4 = \mathbf{0}$
   $$\text{ker}(\Phi) = \text{span}\left[ \begin{bmatrix} 0 \\ 1/2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -1 \\ 1/2 \\ 0 \\ 1 \end{bmatrix} \right]$$

### 📌 3. Rank-Nullity 정리 (Theorem 2.24 & Eq 2.129)
$$\dim(\text{ker}(\Phi)) + \dim(\text{Im}(\Phi)) = \dim(V)$$

- Example 2.25 차원 검증: $\dim(\text{ker}(\Phi)) + \dim(\text{Im}(\Phi)) = 2 + 2 = 4 = \dim(\mathbb{R}^4)$ (완벽 성립!).

## 🚀 5. 4단계 실전 AI / 머신러닝 연결고리
- Autoencoder의 Latent Space & Dimensionality Reduction:
  - 인코더 $\Phi : \mathbb{R}^D \to \mathbb{R}^d$ ($D \gg d$) 과정에서 $\dim(\text{ker}(\Phi)) = D - d$ 만큼의 원본 차원이 찌그러져 사라집니다. AI 모델은 이 Rank-Nullity 정리의 지배를 받는 손실 차원을 최소화하기 위해 가장 유의미한 주성분 상(Image / Subspace)으로 데이터를 정사영(Projection)하는 가중치 $W$ 를 학습하게 됩니다.
