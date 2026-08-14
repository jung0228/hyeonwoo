# 📘 [MML Ch 2 전수 풀이] 선형독립, Basis, Rank-Nullity 수식 유도

## 📝 Problem 1. (MML Ch 2.4 - Linear Independence & Basis)
> **[문제]** 
> 3차원 공간 $\mathbb{R}^3$의 세 벡터 $v_1 = \begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, v_3 = \begin{bmatrix} 1 \\ 0 \\ a \end{bmatrix}$ 가 있다.
> 1) 세 벡터가 선형독립이 되기 위한 $a$의 조건을 구하시오.
> 2) $a=-2$ 일 때, 열공간의 차원 $\text{Rank}(A)$와 Nullity를 구하시오.

### ✍️ [백지 수식 유도]
1) $A = [v_1, v_2, v_3]$ 의 $\det(A) = 1(a) + 1(2) = a+2 \ne 0 \implies \mathbf{a \ne -2}$.
2) $a=-2$ 일 때 가우스 소거법 결과 Pivot이 2개 ➡️ **$\text{Rank}(A) = \mathbf{2}$**, Rank-Nullity 정리에 의해 **$\text{Nullity}(A) = 3 - 2 = \mathbf{1}$**.

### 🧠 [개념 4단계 & 인사이트]
- **[1. 개념정의]**: 선형독립은 어떤 벡터도 다른 벡터의 선형결합으로 표현 불가능한 최소 상태.
- **[2. 존재이유]**: 공간을 중복(Redundancy) 없이 최소한의 기저(Basis)로 고유하게 좌표화하기 위함.
- **[3. Trade-off/직관]**: $a=-2$가 되면 3차원 축 중 하나가 찌그러져 2차원 평면으로 압축되고 1차원 영공간으로 정보가 날아감.
- **[4. AI연결]**: 신경망 가중치 행렬 $\det(W)=0$ 시 차원 고갈(Rank Collapse) 현상의 수학적 이유.

---

## 📝 Problem 2. (MML Ch 2.7 - Linear Mappings & Rank-Nullity)
> **[문제]**
> $A = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 1 & 3 & 1 & 1 \end{bmatrix}$ 의 $\text{Kernel}(T)$ 의 기저와 차원, 그리고 $\text{Rank}(A)$를 구하시오.

### ✍️ [백지 수식 유도]
1) $Ax = 0$ 가우스 소거 후 자유변수 $x_3, x_4$ 기준 해공간 계산:
   - **$\text{Kernel}(T)$ 기저**: $\left\{ [2, -1, 1, 0]^T, [-1, 0, 0, 1]^T \right\}$ ➡️ **$\text{Nullity}(A) = \mathbf{2}$**
2) **$\text{Rank}(A) = \mathbf{2}$** ➡️ $\text{Rank} + \text{Nullity} = 2 + 2 = 4$ (Domain $\mathbb{R}^4$ 차원 일치!).
