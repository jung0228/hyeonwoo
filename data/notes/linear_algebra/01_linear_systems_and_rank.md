# 📐 01. 선형방정식계, Pivot, 그리고 Rank (Linear Systems & Rank)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- **Pivot (피벗)**: 가우스 소거법 후 각 행에서 0이 아닌 숫자로 처음 등장하는 대장 요소.
- **Rank (계수)**: 피벗의 개수이자, 행렬이 형성하는 실제 독립된 공간의 차원 $\dim(\text{Col}(A))$.
- **라우셰-카펠리 정리 (Rouché–Capelli Theorem)**:
  - $\text{Rank}(A) = \text{Rank}([A|b]) = n \implies$ **유일해 (Unique Solution)**
  - $\text{Rank}(A) = \text{Rank}([A|b]) < n \implies$ **무수히 많은 해 (Infinite Solutions)**
  - $\text{Rank}(A) < \text{Rank}([A|b]) \implies$ **해 없음 (Inconsistent System)**

---

## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.1 ~ 2.4)

### [Problem 1] Ex 2.1 - 선형계 가우스 소거법 및 유일해 유도
- **연립방정식**:
  $$\begin{aligned} x_1 + 2x_2 + x_3 &= 1 \\ 2x_1 + 3x_2 + 4x_3 &= 3 \\ x_1 + 4x_2 - 2x_3 &= -1 \end{aligned}$$
- **증대행렬 가우스 소거**:
  $$[A | b] = \begin{bmatrix} 1 & 2 & 1 & | & 1 \\ 2 & 3 & 4 & | & 3 \\ 1 & 4 & -2 & | & -1 \end{bmatrix} \xrightarrow{\text{Row Operations}} \begin{bmatrix} 1 & 2 & 1 & | & 1 \\ 0 & -1 & 2 & | & 1 \\ 0 & 0 & 1 & | & 0 \end{bmatrix}$$
- **해석**: 피벗이 3개이므로 $\text{Rank}(A) = \text{Rank}([A|b]) = 3 = n$. 유일해 존재!
- **거꾸로 대입 (Back-substitution)**: $x_3 = 0 \implies x_2 = -1 \implies x_1 = 3$. **해: $[3, -1, 0]^T$**.

### [Problem 2] Ex 2.2 - Inconsistent System 및 수식적/기하학적 맹점 분석
- **연립방정식**: $x_1 + x_2 = 2$, $2x_1 + 2x_2 = 5$
- **증대행렬 소거**:
  $$[A | b] = \begin{bmatrix} 1 & 1 & | & 2 \\ 2 & 2 & | & 5 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 2R_1} \begin{bmatrix} 1 & 1 & | & 2 \\ 0 & 0 & | & 1 \end{bmatrix}$$
- **비판적 서술 3단계**:
  1. **대수적 모순**: 2행이 $0 \cdot x_1 + 0 \cdot x_2 = 1 \implies \mathbf{0 = 1}$ 이 되어 해 불가능.
  2. **랭크 불일치**: $\text{Rank}(A) = 1 < \text{Rank}([A|b]) = 2$. 결과 $b$가 열공간 $\text{Col}(A)$ 밖으로 튕겨 나감 ($b \notin \text{Col}(A)$).
  3. **기하학적 모순**: 기울기가 같고 절편이 다른 두 평행선이므로 교점이 존재하지 않음.

---

## 🔍 3. 비판적 맹점 & 실전 AI 연결
- **수치적 불안정성 (Numerical Instability)**: 피벗이 0에 매우 가까우면 부동소수점 오차 폭발 ➡️ **Partial Pivoting (부분 피벗팅)** 필수.
- **최소제곱법 (Least Squares)**: 해가 없을 때 정사영을 내려 $w = (X^TX)^{-1}X^Ty$ 최적 근사해 추정.
