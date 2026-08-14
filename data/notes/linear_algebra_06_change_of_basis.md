# 📐 06. 기저변환과 행렬의 대각화 (Change of Basis & Diagonalization)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- **기저변환 행렬 ($P_{\mathcal{E} \leftarrow \mathcal{B}}$)**: 표준 좌표계 $\mathcal{E}$의 벡터를 새로운 관점의 기저 $\mathcal{B}$ 상의 좌표로 변환하는 변환 행렬 ($P = [b_1 \mid b_2]$).
- **행렬의 대각화 ($\tilde{A} = P^{-1} A P = \Lambda$)**: 축들이 서로 얽혀 있는(Coupled) 표준 공간을, 축들이 완벽히 독립(Decoupled)된 고유벡터 기저 공간으로 관점을 바꿔 **행렬을 대각행렬(Diagonal Matrix)로 재좌표화하는 연산**.

---

## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.5)

### [Problem 6] Ex 2.6 - 기저변환과 대각화 백지 수식 유도
- **문제**: $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ 를 새로운 고유기저 $\mathcal{B} = \left\{ \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ -1 \end{bmatrix} \right\}$ 에 대해 기저변환한 행렬 $\tilde{A} = P^{-1} A P$ 를 구하시오.

- **수식 유도 3단계**:
  1. **변환행렬 $P$ 및 역행렬 $P^{-1}$ 구하기**:
     $$P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad P^{-1} = \frac{1}{-2} \begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

  2. **$\tilde{A} = P^{-1} A P$ 대입 연산**:
     $$A P = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix}$$
     $$\tilde{A} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix} = \mathbf{\begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}}$$

  3. **결과 해석**: 고유벡터 축 상에서 행렬 $A$가 **완벽한 대각행렬 $\text{diag}(3, 1)$** 로 변환됨!

---

## 🔍 3. 비판적 맹점 & 실전 AI 연결

### 1) $O(n^3) \rightarrow O(n)$ 연산 절감 맹점
- 복잡한 행렬 거듭제곱 연산 $A^k$를 직접 곱하려면 $O(k n^3)$의 엄청난 계산량이 들지만, 대각화를 수행하면 $\tilde{A}^k = \begin{bmatrix} 3^k & 0 \\ 0 & 1^k \end{bmatrix}$ 로 **$O(n)$ 연산으로 파격 절감**됨.
- **맹점**: $P, P^{-1}$을 구하는 1회성 초기 변환 비용이 $O(n^3)$으로 비싸므로, 연산이 반복될 때만 이득임.

### 2) 실전 AI 연결 (Transformer Multi-Head Attention)
- Multi-Head Attention이 입력 $X$를 가중치 $W_Q, W_K, W_V$를 통해 서로 다른 서브스페이스(Subspace)로 기저변환(Change of Basis)하여 다양한 관점의 서사를 독립적으로 추출함.
