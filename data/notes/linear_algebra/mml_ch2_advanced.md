# 📘 [MML Ch 2 고난도 풀이] 기저변환(Basis Change) & 어파인 공간(Affine Subspace)

## 📝 Ex 2.6 - Change of Basis & Diagonalization
> **[문제]** 
> $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ 를 기저 $\mathcal{B} = \left\{ \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ -1 \end{bmatrix} \right\}$ 로 기저변환한 행렬 $\tilde{A} = P^{-1} A P$ 를 구하시오.

### ✍️ [백지 수식 풀이]
1. 변환행렬 $P = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, $P^{-1} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$
2. $\tilde{A} = P^{-1} A P = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix} = \mathbf{\begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}}$ (대각화 완료!)

---

## 📝 Ex 2.7 - Affine Subspaces ($Ax = b$)
> **[문제]** 
> $Ax = b$ 의 특수해 $x_p = [1, 0, 2]^T$ 와 $\text{Kernel}(A)$ 의 기저 $v_1 = [2, 1, 0]^T$ 일 때, 해집합 $S$가 벡터공간이 아님을 증명하시오.

### ✍️ [백지 수식 풀이]
- $S = \{ x_p + c_1 v_1 \mid c_1 \in \mathbb{R} \}$ 에 대해, **영벡터 $0 \notin S$** (원점을 지나지 않음!).
- $\therefore S$는 벡터공간이 아닌, 원점에서 $x_p$만큼 평행이동된 **어파인 공간(Affine Subspace)**이다.
