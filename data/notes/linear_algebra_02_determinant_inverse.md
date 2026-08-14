# 📐 02. 행렬식, 역행렬, 그리고 수치적 연산 복잡도 (Determinant & Inverse)

## 1. ⚔️ 근본 개념 정의 & 존재 이유
- **행렬식 ($\det(A)$)**: 선형변환 후 공간의 부피(면적) 팽창율. 2D에서 $(ad - bc)$ 면적 그 자체.
- **역행렬 ($A^{-1}$)**: 찌그러진 공간을 원래 공간으로 원상복구하는 역변환 ($A A^{-1} = I$).
- **가역성 파탄 조건**: $\det(A) = 0 \implies$ 면적이 0으로 납작하게 눌려(Dimensionality Collapse) 역변환 불가능.

---

## 📝 2. MML 교재 연습문제 풀이 (MML Ch 2.3)

### [Problem 3] Ex 2.3 - 2x2 역행렬 수식 유도 및 검증
- **문제**: $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ 의 역행렬 유도 및 $AA^{-1} = I$ 검증.
- **수식 유도**:
  - $\det(A) = (1 \cdot 4) - (2 \cdot 3) = -2 \ne 0 \implies$ 역행렬 존재!
  - $A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}$
- **$A A^{-1} = I$ 곱셈 증명**:
  $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix} = \begin{bmatrix} -2+3 & 1-1 \\ -6+6 & 3-2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I_2$$

---

## 🔍 3. 비판적 맹점 & 실전 AI 연결

### 1) $\frac{1}{\det(A)}$ 값의 치명적 수치 폭발 (Ill-conditioned)
- $\det(A) \approx 0$ 이면 $\frac{1}{\det(A)}$ 이 거대해져 역행렬 원소들이 폭발함 ➡️ 부동소수점 오버플로우 발생.

### 2) 역행렬 직접 연산의 $O(n^3)$ 계산 복잡도 증명
- $[A \mid I_n]$ 증대행렬 소거 시, 1개 열 소거에 $O(n^2)$ 필요 ➡️ $n$개 열 소거에 **총 $O(n^3)$ 연산 소요**.
- **AI 대안**: 직접 역행렬 계산을 피하고 **LU 분해**나 **Gradient Descent (경사하강법)** 적용.

### 3) 딥러닝 실전 적용 (Normalizing Flow & VAE)
- **Normalizing Flow**: 변수변환 시 부피 보정용 **Jacobian Determinant $|\det(\mathbf{J})|$** 필수 사용.
- **가우시안 VAE**: 확률밀도 정규화 분모 $|\det(\Sigma)|^{1/2}$ 로 적용.
