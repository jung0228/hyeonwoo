# 📐 03. 부분공간, Kernel/Image, 그리고 차원 정리 (Subspaces & Rank-Nullity)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- **Kernel (영공간)**: 변환을 지나 0으로 매핑되어 손실되는 공간 ($Ax = 0$).
- **Image (열공간)**: 변환 후 살아남아 도달할 수 있는 출력 공간 ($\text{Col}(A)$).
- **Rank-Nullity 정리**: $\text{Rank}(A) + \text{Nullity}(A) = n$ (입력 전체 차원 보존 법칙).
- **부분공간 차원 정리**: $\dim(U+W) = \dim(U) + \dim(W) - \dim(U \cap W)$.

---

## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.4 & 2.7)

### [Problem 4] Ex 2.4 - 부분공간 차원 정리 수식 유도
- **문제**: $U = \text{Span}([1,0,0,0]^T, [0,1,0,0]^T)$, $W = \text{Span}([0,1,1,0]^T, [0,0,1,1]^T)$ 에 대해 차원 정리 검증.
- **수식 유도**:
  - $U \cap W$ 의 연립방정식 풀이 ➡️ 영벡터 $[0,0,0,0]^T$ 뿐이므로 $\dim(U \cap W) = 0$.
  - $U+W$ 의 피벗 개수 ➡️ $\dim(U+W) = 3$.
  - **검증**: $3 = 2 + 2 - 0 \implies 3 = 3$.
- **비판적 맹점 지적**: 중복되는 기저를 제거하지 않으면 선형종속이 되므로 1번 차감해야 함. $U \cup W$ 는 부분공간이 아니므로 반드시 합공간 $U+W$ 로 서술해야 함.

### [Problem 5] Ex 2.7 - Kernel/Image 백지 유도 및 직합
- **문제**: $A = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 1 & 3 & 1 & 1 \end{bmatrix}$ 의 Kernel/Image 기저 유도.
- **수식 유도**:
  - $Ax = 0$ 소거 ➡️ 자유변수 $x_3, x_4 \implies \text{Kernel}$ 기저 2개 ($\text{Nullity}=2$).
  - 피벗 열 2개 $\implies \text{Rank}=2$.
  - **검증**: $2 + 2 = 4 = n$ (Domain $\mathbb{R}^4$ 완벽 일치!).

---

## 🔍 3. 비판적 맹점 & 실전 AI 연결
- **Information Loss in Kernel**: Kernel로 들어간 정보는 0으로 매핑되어 영구 손실 ➡️ ResNet의 **Skip Connection ($y = F(x) + x$)** 으로 손실 강제 차단.
- **Direct Sum ($U \oplus W$)**: $\dim(U \cap W) = 0$ 일 때만 중복이 $0$이 되어 벡터가 유일(Uniquely)하게 분해됨.
