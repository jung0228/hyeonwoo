# 📐 2.7 Linear Mappings (선형사상과 기저변환)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.7 원문 완전 복사 & 심층 가공 노트


## 📖 Part 1. MML 교재 원문 (Textbook Original Text)

### 1. Definition 2.15 (Linear Mapping) & Definition 2.16 (Injective, Surjective, Bijective)
```text
Definition 2.15 (Linear Mapping).
For vector spaces V, W, a mapping \Phi : V -> W is called a linear mapping (or vector space homomorphism / linear transformation) if
\forall x, y in V, \forall \lambda, \psi in R: \Phi(\lambda x + \psi y) = \lambda \Phi(x) + \psi \Phi(y). (Eq 2.87)

Special Mappings:
- Isomorphism: \Phi : V -> W linear and bijective.
- Endomorphism: \Phi : V -> V linear.
- Automorphism: \Phi : V -> V linear and bijective.

Definition 2.16 (Injective, Surjective, Bijective):
- Injective if \forall x, y in V: \Phi(x) = \Phi(y) ==> x = y.
- Surjective if \Phi(V) = W.
- Bijective if it is both injective and surjective.
```

### 2. Theorem 2.20 (Basis Change / Basis Transformation Matrix)
```text
Theorem 2.20 (Basis Change).
For a linear mapping \Phi : V -> W, ordered bases B = (b1, ..., bn), B_tilde = (b1_tilde, ..., bn_tilde) of V and C = (c1, ..., cm), C_tilde = (c1_tilde, ..., cm_tilde) of W, and a transformation matrix A_\Phi with respect to B and C, the transformation matrix A_\Phi_tilde with respect to B_tilde and C_tilde is given by
A_\Phi_tilde = T^{-1} A_\Phi S   (Eq 2.116)
where S is the transformation matrix of id_V mapping B_tilde to B, and T is the transformation matrix of id_W mapping C_tilde to C.
```

### 3. Definition 2.23 (Kernel and Image) & Remark (Column Space / Null Space)
```text
Definition 2.23 (Image and Kernel).
For \Phi : V -> W, we define:
- ker(\Phi) := \Phi^{-1}(0_W) = {v in V : \Phi(v) = 0_W}  (Eq 2.122)
- Im(\Phi) := \Phi(V) = {w in W | \exists v in V : \Phi(v) = w} (Eq 2.123)

Remark (Null Space and Column Space):
- Im(\Phi) = {Ax : x in R^n} = span[a1, ..., an] in R^m (Column Space of A).
- rk(A) = dim(Im(\Phi)).
- ker(\Phi) is the general solution to homogeneous system Ax = 0 (Null Space of A).
```

### 4. Theorem 2.24 (Rank-Nullity Theorem)
```text
Theorem 2.24 (Rank-Nullity Theorem).
For vector spaces V, W and a linear mapping \Phi : V -> W it holds that
dim(ker(\Phi)) + dim(Im(\Phi)) = dim(V). (Eq 2.129)

Direct Consequences:
- If dim(Im(\Phi)) < dim(V), then ker(\Phi) is non-trivial (dim(ker(\Phi)) >= 1).
- If dim(V) = dim(W), then \Phi is injective <==> \Phi is surjective <==> \Phi is bijective.
```


## 🧠 Part 2. 한국어 정밀 가공 & 개념 설명 (Deep Interpretation)

### 📌 1. [개념 정의] 선형사상, 영공간(Kernel), 상(Image)이란 무엇인가?
- 선형사상 (Linear Mapping): 공간의 선형성(가산성 + 스칼라배)을 보존하면서 한 공간 $V$ 의 벡터를 다른 공간 $W$ 로 변환시키는 함수/행렬 사상입니다.
- Kernel (영공간): 사상을 거친 결과 영벡터 $\mathbf{0}$ 으로 무참히 찌그러져 사멸하는 입력 벡터들의 집합입니다.
- Image (상 / Column Space): 사상을 지나 결과 공간 $W$ 상에 실제로 살아남아 도달한 결과물들의 표현 범위입니다.

### 📌 2. [존재 이유 & 직관] 왜 Rank-Nullity 정리가 위대한가?
- 차원 등분 보존 법칙: $n$차원 입력 공간 $V$ 전체는 선형 변환을 거칠 때 "영으로 소실된 차원(Kernel)" + "살아남은 결과 차원(Image)" 으로 단 1차원의 오차도 없이 완벽하게 분할 보전됩니다!
$$\text{dim}(\ker(\Phi)) + \text{dim}(\text{Im}(\Phi)) = n = \text{dim}(V)$$

### 📌 3. [상황별 Trade-off & 맹점] 기저변환(Basis Change)과 유사 변환
- $\tilde{A}_\Phi = P^{-1} A_\Phi P$ 의 본질: 시점을 바꾸면(기저 변경) 복잡해 보이던 사상 행렬이 아주 단순한 대각 행렬(Diagonal Matrix)로 변신할 수 있습니다 (고유값 분해 및 대각화의 핵심원리).

### 📌 4. [실전 AI 연결고리]
- Autoencoder 라텐트 정보 손실 파악: 고차원 데이터 $X \in \mathbb{R}^D$ 가 인코더 $W$ 를 지나 잠재 공간 $z \in \mathbb{R}^d$ ($d \ll D$) 로 압축될 때, Rank-Nullity 정리로 인해 차원 차이 $(D - d)$ 만큼의 정보가 영공간(Kernel)으로 사라지게 됩니다.
