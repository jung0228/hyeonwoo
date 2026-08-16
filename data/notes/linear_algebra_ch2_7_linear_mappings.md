# 📐 2.7 Linear Mappings (선형사상과 기저변환)

> POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.7 전수 분석 & 4단계 정밀 해설 노트

## 🌐 0. 지난 노트(2.6절)와의 연결 및 빌드업: 왜 "선형사상"과 "기저변환"을 배우는가?

우리는 2.6절에서 벡터 공간의 최소 뼈대인 기저(Basis)와 공간의 알짜배기 크기인 차원(Dimension) 및 행렬의 계수(Rank)를 배웠습니다.

이제 2.7절에서는 공간을 가만히 두지 않고 한 벡터 공간 $V$ 에서 다른 벡터 공간 $W$ 로 벡터들을 변형시키고 이동시키는 규칙(함수)을 다룹니다.
이것이 바로 선형사상(Linear Mapping) 또는 선형변환(Linear Transformation)입니다.

또한, "동일한 선형변환이라도 관찰하는 기저(좌표축)를 바꾸면 행렬의 모양이 어떻게 일목요원하게 달라지는가?" 를 다루는 기저변환(Basis Change)과, 
"변환 과정에서 0으로 찌그러져 사라지는 차원과 살아서 이동하는 차원의 보존 법칙"인 Rank-Nullity 정리를 공부합니다.

## 1. ⚔️ Section 2.7.1: Linear Mappings & Special Classifications (선형사상의 분류)

### 📌 1. 선형사상(Linear Mapping)의 수학적 정의 (Definition 2.15)
두 벡터 공간 $V, W$ 에 대해 사상(Mapping) $\Phi : V \to W$ 가 모든 $x, y \in V$ 와 스칼라 $\lambda, \psi \in \mathbb{R}$ 에 대해 다음을 만족하면 선형사상(Linear Mapping)이라 부릅니다:

$$\Phi(\lambda x + \psi y) = \lambda \Phi(x) + \psi \Phi(y) \quad (\text{Eq 2.87})$$

- 중합의 원리 (Superposition Principle): 덧셈 보존 $\Phi(x+y) = \Phi(x) + \Phi(y)$ 과 스칼라배 보존 $\Phi(\lambda x) = \lambda \Phi(x)$ 가 동시에 성립함을 의미합니다.

### 📌 2. 선형사상의 4가지 분류 체계 (Special Mappings)
1. 동형사상 (Isomorphism): $\Phi : V \to W$ 가 선형이면서 전단사(Bijective: 일대일 대응)인 사상. 두 공간 $V$ 와 $W$ 가 수학적으로 완벽히 동일한 구조임을 뜻합니다.
2. 단형사상 (Endomorphism): $\Phi : V \to V$ 자기 자신으로 가는 선형사상.
3. 자기도형사상 (Automorphism): $\Phi : V \to V$ 자기 자신으로 가면서 전단사(Bijective)인 선형사상 (역변환 가능).

### 📌 3. 단사(Injective), 전사(Surjective), 전단사(Bijective) 정의 (Definition 2.16)
- 단사 (Injective / One-to-One): $\forall x, y \in V : \Phi(x) = \Phi(y) \implies x = y$. (서로 다른 입력은 무조건 서로 다른 출력으로 매핑됨).
- 전사 (Surjective / Onto): $\Phi(V) = W$. (공역 $W$ 의 모든 원소가 적어도 하나의 화살을 맞음).
- 전단사 (Bijective): 단사이면서 동시에 전사인 경우 (1:1 완전 대응).

## 2. ⚔️ Section 2.7.2: Basis Change / Transformation Matrix (기저변환 정리)

### 📌 1. 기저변환 정리 (Theorem 2.20 & Eq 2.116)
벡터 공간 $V$ 의 순서기저 $\mathcal{B}, \tilde{\mathcal{B}}$ 와 $W$ 의 순서기저 $\mathcal{C}, \tilde{\mathcal{C}}$ 가 주어지고, 기저 $\mathcal{B}, \mathcal{C}$ 에 대한 선형사상 $\Phi$ 의 행렬 표현이 $A_\Phi$ 일 때, 새로운 기저 $\tilde{\mathcal{B}}, \tilde{\mathcal{C}}$ 에 대한 행렬 표현 $\tilde{A}_\Phi$ 는 다음과 같이 주어집니다:

$$\tilde{A}_\Phi = T^{-1} A_\Phi S \quad (\text{Eq 2.116})$$

- 여기서 $S$ 는 $V$ 공간에서 기저 $\tilde{\mathcal{B}}$ 를 $\mathcal{B}$ 로 변환하는 기저변환 행렬이며, $T$ 는 $W$ 공간에서 기저 $\tilde{\mathcal{C}}$ 를 $\mathcal{C}$ 로 변환하는 기저변환 행렬입니다.
- 닮음 행렬 (Similar Matrices): 특히 $V = W$ 인 자기도형사상에서는 $\tilde{A} = P^{-1} A P$ 형태가 되며, 두 행렬 $A$ 와 $\tilde{A}$ 를 닮음 행렬이라 부릅니다.

## 3. ⚔️ Section 2.7.3: Kernel and Image (영공간과 상)

### 📌 1. 영공간(Kernel / Null Space)과 상(Image / Column Space)의 정의 (Definition 2.23)
선형사상 $\Phi : V \to W$ 에 대하여:

1. 영공간 (Kernel / Null Space: Eq 2.122):
   $$\text{ker}(\Phi) := \{ v \in V \mid \Phi(v) = \mathbf{0}_W \}$$
   - 의미: 변환 결과 영벡터 $\mathbf{0}$ 으로 찌그러져 사라지는 $V$ 안의 모든 입력 벡터들의 집합입니다. 동차계 $A\mathbf{x} = \mathbf{0}$ 의 해공간과 완벽히 일치합니다.

2. 상 (Image / Column Space: Eq 2.123):
   $$\text{Im}(\Phi) := \{ w \in W \mid \exists v \in V : \Phi(v) = w \}$$
   - 의미: $V$ 의 모든 벡터들이 변환을 통해 $W$ 상에 실제로 도달하는 출력 도달 범위입니다. 행렬 $A$ 의 열벡터들의 스팬 $\text{span}[\mathbf{a}_1, \dots, \mathbf{a}_n]$ (열공간)과 일치합니다.

### 📌 2. Rank-Nullity 정리 (차원 정리: Theorem 2.24 & Eq 2.129)
유한차원 벡터 공간 $V, W$ 와 선형사상 $\Phi : V \to W$ 에 대해 다음 차원 보존 법칙이 무조건 성립합니다:

$$\dim(\text{ker}(\Phi)) + \dim(\text{Im}(\Phi)) = \dim(V) \quad (\text{Eq 2.129})$$

$$\text{Nullity}(\Phi) + \text{Rank}(\Phi) = \dim(V)$$

- 정리의 파생결과 (Direct Consequences):
  1. 만약 $\dim(\text{Im}(\Phi)) < \dim(V)$ 이면, 영공간은 비자명합니다 ($\dim(\text{ker}(\Phi)) \ge 1$). 즉, 정보가 찌그러져 손실되는 차원이 반드시 존재합니다.
  2. $\dim(V) = \dim(W)$ 인 경우: $\Phi$ 가 단사(Injective) $\iff \Phi$ 가 전사(Surjective) $\iff \Phi$ 가 전단사(Bijective) 가 모두 동치가 됩니다.

## 🚀 4. 4단계 실전 AI / 머신러닝 연결고리
- Autoencoder의 Latent Space & Dimensionality Reduction:
  - 인코더 $\Phi : \mathbb{R}^D \to \mathbb{R}^d$ ($D \gg d$) 과정에서 $\dim(\text{ker}(\Phi)) = D - d$ 만큼의 원본 차원이 찌그러져 사라집니다. AI 모델은 이 Rank-Nullity 정리의 지배를 받는 손실 차원을 최소화하기 위해 가장 유의미한 주성분 상(Image / Subspace)으로 데이터를 정사영(Projection)하는 가중치 $W$ 를 학습하게 됩니다.
