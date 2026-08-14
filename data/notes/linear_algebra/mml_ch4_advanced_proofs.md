# 📘 [MML Ch 4 고난도 증명] Spectral Theorem & SVD 수학적 유도

## 📝 Ex 4.1 - Spectral Theorem & Orthogonal Eigendecomposition
> **[문제]** 
> 대칭행렬 $A = A^T$ 의 서로 다른 고유값 $\lambda_1 \ne \lambda_2$ 에 대응하는 고유벡터 $x_1, x_2$ 가 서로 수직($x_1 \perp x_2$)함을 증명하고 $A = Q \Lambda Q^T$ 를 유도하시오.

### ✍️ [백지 수식 증명]
1. $\lambda_1 (x_1^T x_2) = (A x_1)^T x_2 = x_1^T A^T x_2 = x_1^T A x_2 = x_1^T (\lambda_2 x_2) = \lambda_2 (x_1^T x_2)$
2. $(\lambda_1 - \lambda_2) (x_1^T x_2) = 0$ 이고 $\lambda_1 \ne \lambda_2$ 이므로 **$x_1^T x_2 = 0 \implies x_1 \perp x_2$**.
3. 정규직교행렬 $Q^T Q = I$ 에 대해 **$A = Q \Lambda Q^T$**.

---

## 📝 Ex 4.2 - SVD & $A^TA$ Eigendecomposition 증명
> **[문제]** 
> $A = U \Sigma V^T$ 일 때 $A^TA$와 $AA^T$를 대입하여 $V, U, \sigma_i$ 간의 관계를 유도하시오.

### ✍️ [백지 수식 증명]
1. $A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = \mathbf{V (\Sigma^T \Sigma) V^T}$
2. $A A^T = (U \Sigma V^T) (U \Sigma V^T)^T = U \Sigma V^T V \Sigma^T U^T = \mathbf{U (\Sigma \Sigma^T) U^T}$
3. $\mathbf{V}$는 $A^T A$의 고유벡터, $\mathbf{U}$는 $AA^T$의 고유벡터, 특이값 $\mathbf{\sigma_i = \sqrt{\lambda_i (A^TA)}}$.
