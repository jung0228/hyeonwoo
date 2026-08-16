# 📐 2.7 Linear Mappings (선형사상과 기저변환)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.7 원문 완전 대조 스토리텔링 노트

---

## 1. 🌐 서론: 왜 "선형사상(Linear Mapping)"을 행렬로 다루는가?

벡터 공간의 구조를 깨뜨리지 않고 다른 공간으로 옮겨주는 변환을 선형사상(Linear Mapping)이라 합니다.
이 장에서는 사상(Mapping)과 행렬(Matrix)이 완벽하게 1:1로 매핑되는 원리, 기저가 바뀔 때 좌표 변환 행렬이 어떻게 바뀌는지(Basis Change), 그리고 사상에 의해 영으로 사라지는 공간(Kernel/Nullspace)과 살아남는 공간(Image/Column Space)의 관계를 규명하는 Rank-Nullity 정리를 공부합니다.

---

## 2. ⚔️ Section 2.7.1: Linear Mappings & Matrix Representation (선형사상과 사상 행렬)

### 📌 1. 선형사상 분류 용어 정립 (Definition 2.18 & p.49)
두 실수 벡터 공간 $V, W$ 간의 선형사상 $\Phi : V \to W$ ($\Phi(\mathbf{x} + \mathbf{y}) = \Phi(\mathbf{x}) + \Phi(\mathbf{y})$, $\Phi(\lambda \mathbf{x}) = \lambda \Phi(\mathbf{x})$) 의 분류:

- Isomorphism (동형사상): $\Phi : V \to W$ 가 전단사(Bijective)인 선형사상 (두 공간의 구조가 완전히 동일함).
- Endomorphism (단형사상): $\Phi : V \to V$ 자기 자신으로의 선형사상.
- Automorphism (자기동형사상): $\Phi : V \to V$ 자기 자신으로의 전단사(Bijective) 선형사상 (가역 행렬과 일치).

---

### 📌 2. 기저 변환 행렬 (Basis Change: Theorem 2.20 & Eq 2.103~2.104)
정렬된 기저 $B = (\mathbf{b}_1, \dots, \mathbf{b}_n)$ 에 대한 좌표를 $\hat{\mathbf{x}}$, 새로운 기저 $\tilde{B} = (\tilde{\mathbf{b}}_1, \dots, \tilde{\mathbf{b}}_n)$ 에 대한 좌표를 $\tilde{\mathbf{x}}$ 라 할 때:

$$\tilde{A}_\Phi = T^{-1} A_\Phi S$$

- $S$: 정의역 $V$ 에서 $B \to \tilde{B}$ 기저변환 행렬
- $T$: 공역 $W$ 에서 $C \to \tilde{C}$ 기저변환 행렬
- Similarity Transformation (닮음 변환): $V = W$ 일 때 $\tilde{A}_\Phi = P^{-1} A_\Phi P$ 형태가 되며, 이는 대각화(Diagonalization) 및 고유값 분해(Eigendecomposition)의 핵심 수학적 기반이 됩니다!

---

## 3. ⚔️ Section 2.7.2: Kernel and Image (영공간과 상)

### 📌 1. Kernel (Nullspace) 과 Image (Column Space) 의 정의 (Definition 2.22 & Eq 2.124)
선형사상 $\Phi : V \to W$ (사상 행렬 $A \in \mathbb{R}^{m \times n}$) 에 대해:

- Kernel (Nullspace / 영공간): $W$ 의 영벡터 $\mathbf{0}_W$ 로 사상되는 $V$ 안의 모든 벡터들의 집합.
  $$\ker(\Phi) = \{\mathbf{x} \in V \mid \Phi(\mathbf{x}) = \mathbf{0}_W\} \subseteq V \quad (\text{Width } n \text{ 차원 부분공간})$$
  - *의미*: 동차 방정식계 $A\mathbf{x} = \mathbf{0}$ 의 일반해 공간.

- Image (Column Space / 상): $V$ 의 벡터들이 사상되어 도달할 수 있는 $W$ 안의 모든 결과 벡터들의 집합.
  $$\text{Im}(\Phi) = \{\Phi(\mathbf{x}) \in W \mid \mathbf{x} \in V\} = \text{span}[\mathbf{a}_1, \dots, \mathbf{a}_n] \subseteq W \quad (\text{Height } m \text{ 차원 부분공간})$$
  - *의미*: 행렬 $A$ 의 열벡터들이 만들어내는 Column Space.

---

### 📌 2. Rank-Nullity 정리 (차원 정리: Theorem 2.24 & Eq 2.129)
선형대수학 최고 핵심 정리 중 하나인 Rank-Nullity Theorem (Fundamental Theorem of Linear Mappings):

$$\text{dim}(\ker(\Phi)) + \text{dim}(\text{Im}(\Phi)) = \text{dim}(V)$$

$$\text{Nullity}(A) + \text{Rank}(A) = n \quad (\text{입력 공간 } V \text{ 의 전체 차원 수})$$

- 직관적 해석: 
  $n$차원 입력 공간 전체는 선형사상을 통과하면서 "영으로 찌그러져 사라지는 차원 $\text{dim}(\ker(\Phi))$" 과 "사상되어 살아남는 차원 $\text{dim}(\text{Im}(\Phi))$" 두 개로 정확하게 등분 분할됩니다!
