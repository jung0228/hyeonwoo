# 📐 2.4 Vector Spaces (벡터 공간과 부분공간)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.4 원문 완전 대조 노트**

---

## 1. 🌐 Section 2.4.1: Groups (군과 대수적 구조)

### 📌 Definition 2.7 (Group 군)
집합 $G$ 와 그 상에서 정의된 이항 연산 $\otimes : G \times G \to G$ 에 대해 $G := (G, \otimes)$ 가 다음 4가지 대수적 조건(Group Axioms)을 충족할 때 군(Group)이라 부릅니다:

1. **닫힘성 (Closure)**: $\forall x, y \in G : x \otimes y \in G$
2. **결합법칙 (Associativity)**: $\forall x, y, z \in G : (x \otimes y) \otimes z = x \otimes (y \otimes z)$
3. **항등원 (Neutral Element)**: $\exists e \in G \ \forall x \in G : x \otimes e = x \text{ and } e \otimes x = x$
4. **역원 (Inverse Element)**: $\forall x \in G \ \exists y \in G : x \otimes y = e \text{ and } y \otimes x = e$ (역원은 $x^{-1}$ 로 표기합니다)

- **Abelian Group (아벨군/교환군)**: 추가적으로 교환법칙($x \otimes y = y \otimes x$)이 성립하면 아벨군이라 부릅니다.

---

### 📌 Example 2.10 (군의 예시 및 판별)
- $(\mathbb{Z}, +)$: 아벨군입니다.
- $(\mathbb{N}_0, +)$: 항등원 0은 존재하지만 역원이 없으므로 군이 아닙니다.
- $(\mathbb{Z}, \cdot)$: 항등원 1은 존재하지만 $\pm 1$ 이외 원소의 곱셈 역원이 없으므로 군이 아닙니다.
- $(\mathbb{R}, \cdot)$: 0의 곱셈 역원이 존재하지 않으므로 군이 아닙니다.
- $(\mathbb{R} \setminus \{0\}, \cdot)$: 아벨군입니다.
- $(\mathbb{R}^n, +), (\mathbb{Z}^n, +)$: 성분별 덧셈(Componentwise Addition)에 대해 항등원 $e = (0, \dots, 0)$, 역원 $(-x_1, \dots, -x_n)$ 을 가지는 아벨군입니다 (Eq 2.61).
- $(\mathbb{R}^{m \times n}, +)$: 행렬 덧셈에 대해 아벨군입니다.

---

### 📌 Definition 2.8 (General Linear Group 일반선형군)
가역(Invertible/Regular)인 $n \times n$ 정방행렬의 집합은 행렬 곱셈에 대해 군을 형성하며, 이를 일반선형군(General Linear Group) $GL(n, \mathbb{R})$ 이라 부릅니다.
- 행렬 곱셈은 교환법칙이 성립하지 않으므로 $GL(n, \mathbb{R})$ 은 아벨군이 아닙니다.

---

## 2. ⚔️ Section 2.4.2: Vector Spaces (벡터 공간의 정의와 성질)

### 📌 Definition 2.9 (Vector Space 실수 벡터 공간)
실수 벡터 공간 $V = (V, +, \cdot)$ 은 집합 $V$ 와 내부 연산인 덧셈($+$) 및 외부 연산인 스칼라배($\cdot$) 가 정의된 구조입니다:

$$+ : V \times V \to V \quad (2.62)$$
$$\cdot : \mathbb{R} \times V \to V \quad (2.63)$$

다음 8가지 공리(Axioms)를 충족해야 합니다:
1. $(V, +)$ 이 아벨군(Abelian Group)을 이룹니다 (영벡터 $0 = [0, \dots, 0]^\top$ 포함).
2. **분배법칙 1**: $\forall \lambda \in \mathbb{R}, x, y \in V : \lambda \cdot (x + y) = \lambda \cdot x + \lambda \cdot y$
3. **분배법칙 2**: $\forall \lambda, \psi \in \mathbb{R}, x \in V : (\lambda + \psi) \cdot x = \lambda \cdot x + \psi \cdot x$
4. **외부 연산 결합법칙**: $\forall \lambda, \psi \in \mathbb{R}, x \in V : \lambda \cdot (\psi \cdot x) = (\lambda \psi) \cdot x$
5. **스칼라 항등원**: $\forall x \in V : 1 \cdot x = x$

---

### 📌 Remark (벡터 곱셈에 대한 명확한 구분)
두 벡터의 성분별 곱 $ab$ 는 일반적인 표준 벡터 공간의 연산으로 정의되지 않습니다.
- **Outer Product (외적)**: $ab^\top \in \mathbb{R}^{n \times n}$ (행렬 생성)
- **Inner Product (내적/Dot Product)**: $a^\top b \in \mathbb{R}$ (스칼라 생성)

---

### 📌 Example 2.11 (대표적인 벡터 공간의 종류)
- $V = \mathbb{R}^n$: 표준 성분별 덧셈과 스칼라배에 대해 대표적인 $n$차원 벡터 공간입니다.
- $V = \mathbb{R}^{m \times n}$: $m \times n$ 행렬들의 집합도 행렬 덧셈과 스칼라배에 대해 벡터 공간을 이룹니다 ($\mathbb{R}^{mn}$ 과 동등).
- **Column Vector 표기 관례 (Eq 2.64)**: $\mathbb{R}^n, \mathbb{R}^{n \times 1}$ 은 동일하게 열벡터(Column Vector) $x = [x_1, \dots, x_n]^\top$ 로 표기하며, 행벡터(Row Vector)는 전치 $x^\top \in \mathbb{R}^{1 \times n}$ 로 구분합니다.

---

## 3. ⚔️ Section 2.4.3: Vector Subspaces (부분공간)

### 📌 Definition 2.10 (Vector Subspace 벡터 부분공간)
벡터 공간 $V = (V, +, \cdot)$ 의 공집합이 아닌 부분집합 $U \subseteq V$ 가 기존 $V$의 연산에 대해 스스로 벡터 공간을 이룰 때 $U$를 $V$의 부분공간(Subspace)이라 부르며 $U \subseteq V$ 로 표기합니다.

---

### 📌 부분공간 판별 3대 필수 조건 (Subspace Test)
부분집합 $U \subseteq V$ 가 부분공간인지 확인하기 위한 3가지 판별 조건:

1. **비어있지 않음 (Non-empty)**: $U \neq \emptyset$, 특히 영벡터를 포함해야 함 ($0 \in U$).
2. **외부 연산(스칼라배) 닫힘성**: $\forall \lambda \in \mathbb{R} \ \forall x \in U : \lambda x \in U$.
3. **내부 연산(덧셈) 닫힘성**: $\forall x, y \in U : x + y \in U$.

---

### 📌 Example 2.12 (부분공간의 대표적 예시 및 맹점)
- **자명한 부분공간 (Trivial Subspaces)**: 모든 벡터 공간 $V$ 에 대해 자기 자신 $V$ 와 영공간 $\{0\}$ 은 자명한 부분공간입니다.
- **Figure 2.6 분석 (2차원 공간 $\mathbb{R}^2$ 부분집합 판별)**:
  - A, C: 스칼라배 및 덧셈 닫힘성 위반 (부분공간 아님).
  - B: 영벡터 $(0,0)$ 을 포함하지 않음 (부분공간 아님).
  - D: 원점을 지나는 직선 ➡️ **부분공간이 맞음**.
- **선형방정식계 해공간과 부분공간의 관계**:
  - **동차 방정식계 $Ax = 0$ 의 해집합**: 무조건 $\mathbb{R}^n$ 의 부분공간(Subspace / Kernel)을 이룹니다.
  - **비동차 방정식계 $Ax = b \ (b \neq 0)$ 의 해집합**: 영벡터가 포함되지 않으므로 부분공간이 아니며, 이동된 **아핀 공간(Affine Subspace)**이 됩니다.
  - **교집합 성질**: 임의의 부분공간들의 교집합(Intersection)은 항상 자기 자신도 부분공간이 됩니다.

- **Remark**: $\mathbb{R}^n$ 의 모든 부분공간 $U$ 는 어떤 동차 선형방정식계 $Ax = 0$ 의 해공간으로 표현할 수 있습니다.
