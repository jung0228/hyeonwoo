# 📘 [MML Ch 2 추상행렬 일반증명] 숫자를 몰라도 성립하는 정리 3선

## 📝 Proof 1. Rank-Nullity 정리 일반 증명 ($\text{Rank}(T) + \text{Nullity}(T) = n$)
- **기저 확장(Basis Extension)**: $\text{Kernel}(T)$의 기저 $\{v_1, \dots, v_k\}$를 전체 도메인 $V$의 기저 $\{v_1, \dots, v_k, v_{k+1}, \dots, v_n\}$로 확장.
- **Image 기저 증명**: 임의의 $v$에 대해 $T(v) = \sum_{i=k+1}^n c_i T(v_i)$ 이므로 $\{T(v_{k+1}), \dots, T(v_n)\}$이 $\text{Image}(T)$를 생성하고 선형독립임을 증명.
- $\therefore \text{Rank}(T) = n - k = n - \text{Nullity}(T) \implies \mathbf{\text{Rank} + \text{Nullity} = n}$.

## 📝 Proof 2. 행렬곱의 Rank 상한 정리 ($\text{Rank}(AB) \le \min(\text{Rank}(A), \text{Rank}(B))$)
1. $\text{Col}(AB) \subseteq \text{Col}(A) \implies \text{Rank}(AB) \le \text{Rank}(A)$.
2. $\text{Kernel}(B) \subseteq \text{Kernel}(AB) \implies \text{Nullity}(B) \le \text{Nullity}(AB)$.
3. Rank-Nullity 정리에 의해 $n - \text{Rank}(B) \le n - \text{Rank}(AB) \implies \text{Rank}(AB) \le \text{Rank}(B)$.
4. $\therefore \mathbf{\text{Rank}(AB) \le \min(\text{Rank}(A), \text{Rank}(B))}$.

## 📝 Proof 3. 블록 대각 행렬 역행렬 증명 ($M = \text{diag}(A, B)$)
- $M M^{-1} = \begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix} \begin{bmatrix} A^{-1} & 0 \\ 0 & B^{-1} \end{bmatrix} = \begin{bmatrix} A A^{-1} & 0 \\ 0 & B B^{-1} \end{bmatrix} = \begin{bmatrix} I_m & 0 \\ 0 & I_n \end{bmatrix} = I_{m+n}$.
- 성분과 무관하게 역행렬 유일성에 의해 성립.
