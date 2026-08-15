# 📐 2.4 Vector Spaces & Subspaces (벡터공간과 부분공간)

> **POSTECH 대학원 지정 교재 《MML (Mathematics for Machine Learning)》 Section 2.4 완전 해부**

---

## 1. ⚔️ 4단계 개념 구조화

### 1️⃣ [1단계 명확한 개념 정의]
- **벡터 공간 (Vector Space)**: 덧셈과 스칼라배 8가지 연산 공리(닫힘성, 결합성, 항등원, 역원 등)를 만족하는 벡터 집합 $V$.
- **부분공간 (Vector Subspaces)**: $V$의 부분집합 $U \subseteq V$가 그 자체로 벡터 공간이 되기 위한 **3대 필수 조건**:
  1. 원점 포함: $\mathbf{0} \in U$
  2. 덧셈 닫힘: $\forall u, v \in U \implies u + v \in U$
  3. 스칼라배 닫힘: $\forall u \in U, c \in \mathbb{R} \implies c u \in U$

---

## 🔍 2. ★ MML 교재 원문 심층 해부: Example 2.12 (Vector Subspaces)

MML 교재에서 $\mathbb{R}^2$ 평면 상의 4가지 부분집합 $A, B, C, D$의 예시를 통해 부분공간 판별 조건을 직관적으로 증명함.

- **[Case A] $U_A = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_2 = x_1 + 1\}$ (원점을 지나지 않는 직선)**
  - ❌ **부분공간 탈락!**
  - **이유**: $(0, 0)$을 대입하면 $0 = 0 + 1$ 모순 ➡️ **원점(Zero Vector)을 포함하지 않음 ($\mathbf{0} \notin U_A$)**.
- **[Case B] $U_B = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_1 \ge 0, x_2 \ge 0\}$ (1사분면 전체)**
  - ❌ **부분공간 탈락!**
  - **이유**: 스칼라배에 대해 닫혀있지 않음. $(1, 1) \in U_B$이지만 $c = -1$을 곱하면 $(-1, -1) \notin U_B$.
- **[Case C] $U_C = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_1 x_2 = 0\}$ ($x_1$축과 $x_2$축의 합집합)**
  - ❌ **부분공간 탈락!**
  - **이유**: 덧셈에 대해 닫혀있지 않음. $(1, 0) \in U_C$이고 $(0, 1) \in U_C$이지만 둘을 더한 $(1, 1) \notin U_C$ ($1 \cdot 1 = 1 \neq 0$).
- **[Case D] $U_D = \{(x_1, x_2) \in \mathbb{R}^2 \mid x_2 = -2 x_1\}$ (원점을 지나는 직선)**
  - ✅ **완벽한 벡터 부분공간 (Subspace)!**
  - **이유**: 원점 포함 + 덧셈 닫힘 + 스칼라배 닫힘 3대 공리 완벽 충족!
