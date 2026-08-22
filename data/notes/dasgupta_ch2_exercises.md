# 📐 Dasgupta Algorithm Chapter 2: Divide-and-Conquer (분할 정복 핵심 이론 & 연습문제 2.1 ~ 2.32 전수 해설)

> POSTECH 대학원 지정 교재 《Algorithms (by Sanjoy Dasgupta, Christos Papadimitriou, Umesh Vazirani)》 Chapter 2 핵심 이론 정리 및 연습문제 32문항 100% 전수 해설 노트

---

## 🌐 1. [1단계 명확한 개념 정의]: 분할 정복(Divide-and-Conquer) 풀이를 위한 5대 핵심 이론

### 📌 1. 분할 정복(Divide-and-Conquer)의 3단계 기본 구조
분할 정복은 큰 문제를 한 번에 풀지 않고, 동일한 유형의 더 작은 문제들로 쪼개어 해결하는 알고리즘 설계 패러다임입니다:
1. **Divide (분할)**: 크기가 $n$인 입력 문제를 크기가 더 작은 $b$분의 $1$ 크기인 서브문제 $a$개로 분할합니다.
2. **Conquer (정복)**: 서브문제들을 재귀적(Recursion)으로 해결합니다. 문제 크기가 충분히 작아지면 기저 조건(Base Case)에서 직접 구합니다.
3. **Combine (병합)**: 서브문제들의 해를 합쳐 원래 문제의 최종 답을 만듭니다.

---

### 📌 2. 점상복잡도 분석 1: 마스터 정리 (Master Theorem)
재귀식(Recurrence Relation) 형태가 $T(n) = a T(n/b) + O(n^d)$ 꼴일 때, 재귀 트리를 일일이 그리지 않고 **시간복잡도를 0초 만에 판별하는 핵심 정리**입니다.

- **$a$**: 분할되는 서브문제의 개수 ($a \ge 1$)
- **$b$**: 입력 크기가 줄어드는 비율 ($b > 1$)
- **$O(n^d)$**: 분할 및 병합 단계에 드는 추가 연산량 ($d \ge 0$)

$$\text{임계값 } \log_b a \text{ 와 추가 연산 차수 } d \text{ 를 비교하여 3가지 케이스로 판별:}$$

$$T(n) = \begin{cases} O(n^d) & \text{if } d > \log_b a \;\; (\text{분할/병합 연산이 지배적}) \\\\ O(n^d \log n) & \text{if } d = \log_b a \;\; (\text{모든 레벨의 연산량이 동일}) \\\\ O(n^{\log_b a}) & \text{if } d < \log_b a \;\; (\text{재귀 호출 연산이 지배적}) \end{cases}$$

---

### 📌 3. 점상복잡도 분석 2: 재귀식 전개법 (Substitution / Unrolling)
마스터 정리를 적용할 수 없는 형태(예: $T(n) = T(n-1) + O(1)$, $T(n) = T(\sqrt{n}) + 1$ 등)를 다루는 범용적 해법입니다.

- **원리**: 재귀식을 1단계, 2단계, 3단계 직접 대입 전개하여 **$k$번째 일반항**을 도출한 뒤, 기저 조건(Base Case, 예: $T(1)$)에 도달하는 $k$ 값을 대입하여 시간복잡도를 유도합니다.

---

### 📌 4. 고속 푸리에 변환 (FFT)과 다항식 곱셈
두 $n$차 다항식을 단순 곱셈하면 $O(n^2)$이 걸리지만, **FFT(Fast Fourier Transform)**를 이용하면 **$O(n \log n)$** 만에 계산할 수 있습니다.

- **3단계 메커니즘**:
  1. **계수 표현 ➔ 점-값 표현**: 다항식에 복소수 단위원의 $n$제곱근($\omega = e^{-i 2\pi/n}$) 값들을 대입하여 $O(n \log n)$ 만에 계산 (FFT).
  2. **성분별 곱셈**: 점-값 쌍끼리 곱함 ($O(n)$).
  3. **점-값 표현 ➔ 계수 표현**: 역-FFT(Inverse FFT)를 가해 최종 곱한 다항식의 계수를 복원 ($O(n \log n)$).

---

### 📌 5. 탐색과 정렬의 기하학적/수학적 하한 (Lower Bounds)
- **비교 기반 정렬의 하한**: 원소 간의 대소 비교(`A[i] < A[j]`)만 사용하는 정렬 알고리즘은 결정 트리(Decision Tree)의 리프 노드 개수가 $n!$ 개 이상이어야 하므로, 최소 **$\Omega(n \log n)$** 의 비교 연산이 필수적입니다.
- **비비교 정렬 (Non-comparison Sort)**: 계수 정렬(Counting Sort)처럼 값의 인덱스를 직접 사용하는 알고리즘은 대소 비교를 하지 않으므로 $\Omega(n \log n)$ 하한을 우회하여 **$O(n + M)$** 선형 시간에 정렬할 수 있습니다.

---

## 📝 2. [2단계 원문을 살린 문제 렌더링] & 3. [3단계 문제별 정밀 해설]

---

### 📌 Exercise 2.1 (분할 정복 이진 곱셈)
**[원문]**
Use the divide-and-conquer integer multiplication algorithm to multiply the two binary integers $10011011_2$ and $10111010_2$.

**[해설 및 증명]**
- $x = 10011011_2 = 155_{10}$, $y = 10111010_2 = 186_{10}$ ($n = 8$ 비트)
- 카라츠바(Karatsuba) 분할 정복 알고리즘 적용:
  - $x = x_L \cdot 2^4 + x_R$, 여기서 $x_L = 1001_2 = 9$, $x_R = 1011_2 = 11$
  - $y = y_L \cdot 2^4 + y_R$, 여기서 $y_L = 1011_2 = 11$, $y_R = 1010_2 = 10$
- 3번의 하위 곱셈 계산:
  1. $P_1 = x_L \times y_L = 9 \times 11 = 99 = 1100011_2$
  2. $P_2 = x_R \times y_R = 11 \times 10 = 110 = 1101110_2$
  3. $P_3 = (x_L + x_R) \times (y_L + y_R) = (9 + 11) \times (11 + 10) = 20 \times 21 = 420$
  - 중간항 $P_3 - P_1 - P_2 = 420 - 99 - 110 = 211 = 11010011_2$
- 최종 결합:
  $$xy = P_1 \cdot 2^8 + (P_3 - P_1 - P_2) \cdot 2^4 + P_2$$
  $$= 99 \times 256 + 211 \times 16 + 110 = 25344 + 3376 + 110 = \mathbf{28830} = \mathbf{111000010011110_2}$$

---

### 📌 Exercise 2.2 (거듭제곱 범위 증명)
**[원문]**
Show that for any positive integer $n$ and any base $b$, there must be some power of $b$ lying in the range $[n, bn]$.

**[해설 및 증명]**
- $k = \lceil \log_b n \rceil$ 이라 둡니다.
- 기호 정의에 의해 $k - 1 < \log_b n \le k$ 가 성립합니다.
- 밑이 $b > 1$ 인 지수함수를 양변에 취하면:
  $$b^{k-1} < n \le b^k$$
- $n \le b^k$ 이므로 범위의 하한인 $n$ 보다 $b^k$ 가 크거나 같습니다.
- 또한 $b^{k-1} < n$ 의 양변에 $b$를 곱하면 $b^k < b n$ 이 됩니다.
- 따라서 $n \le b^k < bn$ 이 성립하므로, 거듭제곱 $b^k$ 는 반드시 구간 $[n, bn]$ 내에 존재합니다. $\blacksquare$

---

### 📌 Exercise 2.3 (재귀식 전개법)
**[원문]**
(a) $T(n) = 3T(n/2) + O(n)$ 에 대해 $k$번째 일반항과 $k$ 대입값을 구하시오.
(b) $T(n) = T(n - 1) + O(1)$ 을 전개법으로 푸시오.

**[해설 및 증명]**
- **(a)** $T(n) \le 3 T(n/2) + c n$
  - 1단계: $3(3 T(n/4) + c n/2) + c n = 3^2 T(n/2^2) + c n (1 + 3/2)$
  - 2단계: $3^3 T(n/2^3) + c n (1 + 3/2 + (3/2)^2)$
  - **$k$번째 일반항**: $T(n) \le 3^k T(n/2^k) + c n \sum_{i=0}^{k-1} (3/2)^i = 3^k T(n/2^k) + 2 c n ((3/2)^k - 1)$
  - $k = \log_2 n$ 대입 시 $2^k = n, 3^k = 3^{\log_2 n} = n^{\log_2 3}$:
    $$T(n) \le n^{\log_2 3} T(1) + 2 c n (n^{\log_2 3 - 1} - 1) = \mathbf{O(n^{\log_2 3}) \approx O(n^{1.585})}$$

- **(b)** $T(n) \le T(n-1) + c \le T(n-2) + 2c \le \dots \le T(n-k) + k c$
  - $k = n - 1$ 대입 시: $T(n) \le T(1) + (n-1) c = \mathbf{O(n)}$

---

### 📌 Exercise 2.4 (알고리즘 A, B, C 시간복잡도 비교)
**[원문]**
Algorithm A: 5개의 크기 $n/2$ 서브문제, 병합시간 $O(n)$.
Algorithm B: 2개의 크기 $n-1$ 서브문제, 병합시간 $O(1)$.
Algorithm C: 9개의 크기 $n/3$ 서브문제, 병합시간 $O(n^2)$.

**[해설 및 증명]**
- **Algorithm A**: $T_A(n) = 5 T(n/2) + O(n)$. 마스터 정리 적용 ($a=5, b=2, d=1$).
  - $\log_2 5 \approx 2.32 > 1 = d \implies T_A(n) = \mathbf{O(n^{\log_2 5}) \approx O(n^{2.32})}$
- **Algorithm B**: $T_B(n) = 2 T(n-1) + O(1)$. 재귀 트리로 풀면 깊이 $n$에서 노드 수가 $2^n$개.
  - $T_B(n) = \mathbf{O(2^n)}$ (지수 시간)
- **Algorithm C**: $T_C(n) = 9 T(n/3) + O(n^2)$. 마스터 정리 적용 ($a=9, b=3, d=2$).
  - $\log_3 9 = 2 = d \implies T_C(n) = \mathbf{O(n^2 \log n)}$
- **최종 선택**: 차수가 가장 낮은 **Algorithm C ($O(n^2 \log n)$)** 가 가장 빠르므로 Algorithm C를 선택합니다.

---

### 📌 Exercise 2.5 (11가지 재귀식의 $\Theta$ 바운드 풀이)
**[원문 & 해설]**
- **(a)** $T(n) = 2T(n/3) + 1$: $a=2, b=3, d=0 \implies \log_3 2 > 0 \implies \mathbf{\Theta(n^{\log_3 2})}$
- **(b)** $T(n) = 5T(n/4) + n$: $a=5, b=4, d=1 \implies \log_4 5 > 1 \implies \mathbf{\Theta(n^{\log_4 5})}$
- **(c)** $T(n) = 7T(n/7) + n$: $a=7, b=7, d=1 \implies \log_7 7 = 1 = d \implies \mathbf{\Theta(n \log n)}$
- **(d)** $T(n) = 9T(n/3) + n$: $a=9, b=3, d=1 \implies \log_3 9 = 2 > 1 \implies \mathbf{\Theta(n^2)}$
- **(e)** $T(n) = 8T(n/2) + n^3$: $a=8, b=2, d=3 \implies \log_2 8 = 3 = d \implies \mathbf{\Theta(n^3 \log n)}$
- **(f)** $T(n) = 49T(n/25) + n^{3/2} \log n$: $a=49, b=25 \implies \log_{25} 49 = \log_5 7 \approx 1.209$. $d = 1.5 > 1.209 \implies \mathbf{\Theta(n^{3/2} \log n)}$
- **(g)** $T(n) = T(n-1) + 2$: $T(n) = T(0) + 2n \implies \mathbf{\Theta(n)}$
- **(h)** $T(n) = T(n-1) + n^c$: $T(n) = \sum_{i=1}^n i^c = \mathbf{\Theta(n^{c+1})}$
- **(i)** $T(n) = T(n-1) + c^n \, (c>1)$: 등비수열의 합 $\sum_{i=1}^n c^i = \mathbf{\Theta(c^n)}$
- **(j)** $T(n) = 2T(n-1) + 1$: $T(n) = 2^n - 1 \implies \mathbf{\Theta(2^n)}$
- **(k)** $T(n) = T(\sqrt{n}) + 1$: $m = \log n$ 치환 시 $S(m) = S(m/2) + 1 \implies S(m) = \Theta(\log m) \implies \mathbf{\Theta(\log \log n)}$

---

### 📌 Exercise 2.6 (선형 시불변 시스템과 임펄스 응답)
**[원문]**
$b(t) = 1/t_0$ for $0 \le t \le t_0$, else $0$.
(a) Describe in words the effect of this system.
(b) What is the corresponding polynomial?

**[해설 및 증명]**
- **(a)** 서술적 의미: 입력 신호를 구간 $[0, t_0]$ 동안 이동 평균(Moving Average)하여 평단화(Smoothing)하는 **박스카 필터(Boxcar / Low-pass filter)** 입니다.
- **(b)** 다항식 표현: 연속 신호를 이산화하면 항의 계수가 모두 $1/t_0$ 이고 차수가 $0$부터 $m = t_0/\Delta t$ 까지인 다항식:
  $$B(x) = \frac{1}{t_0} \sum_{i=0}^m x^i = \frac{1}{t_0} \frac{x^{m+1} - 1}{x - 1}$$

---

### 📌 Exercise 2.7 (Unity의 $n$제곱근의 합과 곱)
**[원문]**
What is the sum of the $n$th roots of unity? What is their product if $n$ is odd? If $n$ is even?

**[해설 및 증명]**
- $n$제곱근의 집합: $\omega^k = e^{i 2\pi k / n} \quad (k=0, 1, \dots, n-1)$
- **합 (Sum)**: $\sum_{k=0}^{n-1} \omega^k = \frac{\omega^n - 1}{\omega - 1} = \frac{1 - 1}{\omega - 1} = \mathbf{0}$
- **곱 (Product)**:
  $$\prod_{k=0}^{n-1} e^{i 2\pi k / n} = \exp\left( i \frac{2\pi}{n} \sum_{k=0}^{n-1} k \right) = \exp\left( i \frac{2\pi}{n} \frac{n(n-1)}{2} \right) = e^{i \pi (n-1)} = (-1)^{n-1}$$
  - $n$이 홀수(Odd)일 때: $(-1)^{\text{even}} = \mathbf{1}$
  - $n$이 짝수(Even)일 때: $(-1)^{\text{odd}} = \mathbf{-1}$

---

### 📌 Exercise 2.8 (FFT 계산 연습)
**[원문]**
(a) $(1, 0, 0, 0)$의 FFT 및 역FFT 원본 수열 구하기 ($\omega = -i$).
(b) $(1, 0, 1, -1)$의 FFT 구하기.

**[해설 및 증명]**
- **(a)** $n=4, \omega = e^{-i 2\pi / 4} = -i$.
  - $A(\omega^k) = \sum_{j=0}^3 a_j \omega^{jk} = a_0 = 1$. 따라서 FFT 결과는 $\mathbf{(1, 1, 1, 1)}$.
  - 역FFT로 $(1, 0, 0, 0)$이 나오는 원본 수열은 $\frac{1}{4} \sum_{k=0}^3 \omega^{-jk} = \mathbf{(1/4, 1/4, 1/4, 1/4)}$.
- **(b)** $(1, 0, 1, -1)$:
  - $f(x) = 1 + x^2 - x^3$
  - $f(\omega^0) = f(1) = 1 + 1 - 1 = \mathbf{1}$
  - $f(\omega^1) = f(-i) = 1 + (-i)^2 - (-i)^3 = 1 - 1 - i = \mathbf{-i}$
  - $f(\omega^2) = f(-1) = 1 + 1 - (-1) = \mathbf{3}$
  - $f(\omega^3) = f(i) = 1 + i^2 - i^3 = 1 - 1 + i = \mathbf{i}$
  - 최종 FFT 결과: $\mathbf{(1, -i, 3, i)}$

---

### 📌 Exercise 2.9 (FFT를 이용한 다항식 곱셈)
**[원문]**
(a) $P(x) = x + 1$, $Q(x) = x^2 + 1$ 의 FFT 곱셈.
(b) $P(x) = 1 + x + 2x^2$, $Q(x) = 2 + 3x$ 의 FFT 곱셈.

**[해설 및 증명]**
- **(a)** 차수 합이 3이므로 $n = 4$ 차원 FFT 선택 ($\omega = -i$).
  - $A = (1, 1, 0, 0) \xrightarrow{FFT} (2, 1-i, 0, 1+i)$
  - $B = (1, 0, 1, 0) \xrightarrow{FFT} (2, 0, 2, 0)$
  - 성분별 곱: $C_{FFT} = (4, 0, 0, 0)$
  - 역FFT 계산: $\frac{1}{4} \sum_{k=0}^3 4 \omega^{-jk} = (1, 1, 1, 1)$
  - 다항식 결과: $1 + x + x^2 + x^3 = \mathbf{x^3 + x^2 + x + 1}$
- **(b)** 결과 다항식: $(1 + x + 2x^2)(2 + 3x) = \mathbf{2 + 5x + 7x^2 + 6x^3}$

---

### 📌 Exercise 2.10 (다항식 보간법 - Lagrange Interpolation)
**[원문]**
Find the unique polynomial of degree 4 taking values $p(1)=2, p(2)=1, p(3)=0, p(4)=4, p(5)=0$.

**[해설 및 증명]**
- 라그랑주 보간 공식 $p(x) = \sum_{i=1}^5 y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$ 적용:
  - $y_3 = 0, y_5 = 0$ 이므로 $i=1, 2, 4$ 항만 계산:
  - $i=1$: $2 \cdot \frac{(x-2)(x-3)(x-4)(x-5)}{(1-2)(1-3)(1-4)(1-5)} = \frac{2}{24} (x-2)(x-3)(x-4)(x-5) = \frac{1}{12} (x^4 - 14x^3 + 71x^2 - 154x + 120)$
  - $i=2$: $1 \cdot \frac{(x-1)(x-3)(x-4)(x-5)}{(2-1)(2-3)(2-4)(2-5)} = -\frac{1}{6} (x-1)(x-3)(x-4)(x-5) = -\frac{1}{6} (x^4 - 13x^3 + 59x^2 - 107x + 60)$
  - $i=4$: $4 \cdot \frac{(x-1)(x-2)(x-3)(x-5)}{(4-1)(4-2)(4-3)(4-5)} = -\frac{4}{6} (x-1)(x-2)(x-3)(x-5) = -\frac{2}{3} (x^4 - 11x^3 + 41x^2 - 61x + 30)$
- 계수 합산 정리:
  $$\mathbf{p(x) = -\frac{3}{4} x^4 + \frac{33}{4} x^3 - \frac{133}{4} x^2 + \frac{209}{4} x - 25}$$

---

### 📌 Exercise 2.11 (블록 행렬 곱셈의 성질 증명)
**[원문]**
Show that blockwise matrix multiplication holds for $n/2 \times n/2$ submatrices.

**[해설 및 증명]**
- 행렬 곱의 정의 $XY_{ij} = \sum_{k=1}^n X_{ik} Y_{kj}$ 적용.
- $k$ 인덱스를 앞부분 $1 \dots n/2$ 와 뒷부분 $n/2+1 \dots n$ 으로 분할:
  $$XY_{ij} = \sum_{k=1}^{n/2} X_{ik} Y_{kj} + \sum_{k=n/2+1}^n X_{ik} Y_{kj}$$
- 위치에 따라 $A, B, C, D$ 와 $E, F, G, H$ 블록의 내적 정의와 정확히 일치함을 확인. $\blacksquare$

---

### 📌 Exercise 2.12 (재귀 코드 줄 출력 횟수 분석)
**[원문]**
```python
function f(n)
  if n > 1:
    print_line("still going")
    f(n/2)
    f(n/2)
```

**[해설 및 증명]**
- 재귀식: $L(n) = 2 L(n/2) + 1$, 기저 조건 $L(1) = 0$.
- 마스터 정리 적용 ($a=2, b=2, d=0 \implies \log_2 2 = 1 > 0$):
- 정확한 해: $L(n) = n - 1 = \mathbf{\Theta(n)}$

---

### 📌 Exercise 2.13 (포화 이진 트리 개수와 카탈랑 수)
**[원문]**
(a) $B_3, B_5, B_7$ 값 구하기 및 짝수 정점 불가능 이유.
(b) $B_n$ 의 재귀식 유도.
(c) $B_n = \Omega(2^n)$ 증명.

**[해설 및 증명]**
- **(a)** 포화 이진 트리는 자식이 0개 또는 2개이므로, 정점이 추가될 때 항상 2개씩 늘어납니다. 따라서 짝수 정점 $n$에 대해 $B_n = 0$.
  - $B_3 = 1$, $B_5 = 2$, $B_7 = 5$.
- **(b)** 루트 정점 1개를 빼고 좌측 서브트리에 $k$개, 우측 서브트리에 $n-1-k$개의 정점 배치:
  $$B_n = \sum_{k=1, \text{odd}}^{n-2} B_k B_{n-1-k}$$
- **(c)** 수학적 귀납법: $B_n \ge c \cdot 2^n$ 임을 대입하여 증명. $\blacksquare$

---

### 📌 Exercise 2.14 (중복 제거 $O(n \log n)$ 알고리즘)
**[원문]**
Remove all duplicates from an array of $n$ elements in $O(n \log n)$ time.

**[해설 및 증명]**
1. 배열 $A$를 **MergeSort**로 정렬합니다 ➔ $O(n \log n)$ 소요.
2. 정렬된 배열을 단 1회 순회(Scan)하면서 인접한 원소 $A[i]$와 $A[i+1]$이 같으면 건너뛰고 다를 때만 결과 배열에 추가합니다 ➔ $O(n)$ 소요.
3. 전체 시간 복잡도: $O(n \log n) + O(n) = \mathbf{O(n \log n)}$.

---

### 📌 Exercise 2.15 (In-place Split 연산 구현)
**[원문]**
Implement the QuickSelect split operation in place (without extra memory).

**[해설 및 증명]**
- **3-Way Partitioning (Dutch National Flag Algorithm)**:
  - 투 포인터 $low=1, mid=1, high=n$ 설정.
  - $mid \le high$ 인 동안:
    - $A[mid] < v$ 이면: `swap(A[low], A[mid])`, $low++, mid++$
    - $A[mid] == v$ 이면: $mid++$
    - $A[mid] > v$ 이면: `swap(A[mid], A[high])`, $high--$
  - 메모리 추가 할당 없이 $O(n)$ 시간에 제자리 정렬 완료. $\blacksquare$

---

### 📌 Exercise 2.16 (무한 배열 탐색)
**[원문]**
Find position of $x$ in an infinite sorted array $A$ filled with $\infty$ after $n$ elements in $O(\log n)$ time.

**[해설 및 증명]**
1. **Exponential Search (지수적 범주 탐색)**:
   - 인덱스 $i = 1$ 로 시작하여 $A[i] < x$ 이고 $A[i] \neq \infty$ 인 동안 $i = 2i$ 로 2배씩 늘립니다.
   - 탐색 중단 지점 $i$ 는 $n < i \le 2n$ 범위에 도달하며, 이 단계는 $O(\log n)$ 걸립니다.
2. **Binary Search (이진 탐색)**:
   - 구간 $[i/2, i]$ 내에서 일반 이진 탐색을 수행합니다 ➔ $O(\log n)$ 소요.
3. 총 시간 복잡도: $O(\log n) + O(\log n) = \mathbf{O(\log n)}$.

---

### 📌 Exercise 2.17 (고정점 $A[i] = i$ 찾기)
**[원문]**
Find if there exists $i$ such that $A[i] = i$ in a sorted array of distinct integers in $O(\log n)$ time.

**[해설 및 증명]**
- 변형된 이진 탐색(Binary Search):
  - 중간 인덱스 $mid$ 확인. $D[i] = A[i] - i$ 라고 두면, $A$의 원소들이 서로 다른 정수이므로 $D[i]$는 엄격히 단조증가 함수입니다.
  - $A[mid] == mid$ 이면 정답 $mid$ 반환.
  - $A[mid] > mid$ 이면 우측 원소들은 항상 $A[i] > i$ 이므로 좌측 구간 $[low, mid-1]$ 로 이동.
  - $A[mid] < mid$ 이면 우측 구간 $[mid+1, high]$ 로 이동.
- 탐색 공간이 매 단계 절반으로 줄어들므로 시간 복잡도는 $\mathbf{O(\log n)}$.

---

### 📌 Exercise 2.18 (이진 탐색의 하한 $\Omega(\log n)$ 증명)
**[원문]**
Show that any comparison-based search algorithm takes $\Omega(\log n)$ steps.

**[해설 및 증명]**
- 결정 트리(Decision Tree) 모델: 크기 $n$인 배열에서 찾을 수 있는 결과의 가짓수는 $n+1$개 (각 인덱스 $1 \dots n$ 또는 없음).
- 높이가 $h$인 이진 결정 트리가 가질 수 있는 최대 리프 노드 수는 $2^h$개입니다.
- $2^h \ge n+1 \implies h \ge \log_2(n+1)$
- 따라서 최소 비교 횟수는 $\mathbf{\Omega(\log n)}$ 입니다. $\blacksquare$

---

### 📌 Exercise 2.19 ($k$-way 병합 알고리즘)
**[원문]**
Merge $k$ sorted arrays of size $n$ into a single sorted array of $kn$ elements.

**[해설 및 증명]**
- **(a) 순차 병합**:
  - 1-2번째 병합: $2n$, 3번째 병합: $3n \dots k$번째 병합: $kn$.
  - 총 시간: $\sum_{i=2}^k i n = n \left( \frac{k(k+1)}{2} - 1 \right) = \mathbf{O(k^2 n)}$
- **(b) 분할 정복 병합 (또는 최소 힙 활용)**:
  - $k$개 배열을 2개씩 짝지어 병합하는 토너먼트 방식 적용 (깊이 $\log k$).
  - 각 레벨당 전체 원소 수 $kn$을 병합하므로 $O(kn)$ 작업 소요.
  - 총 시간 복잡도: $\mathbf{O(kn \log k)}$.

---

### 📌 Exercise 2.20 (선형 시간 정렬과 $\Omega(n \log n)$ 하한)
**[원문]**
Sort array in $O(n + M)$ time ($M = \max x_i - \min x_i$). Why doesn't $\Omega(n \log n)$ bound apply?

**[해설 및 증명]**
- **Counting Sort (계수 정렬)** 알고리즘을 사용하면 크기 $M+1$의 카운팅 배열을 이용해 $O(n + M)$ 시간에 정렬할 수 있습니다.
- **하한이 적용되지 않는 이유**: $\Omega(n \log n)$ 하한은 오직 **원소 간의 비교(Comparison-based)**만을 사용하는 정렬에만 적용됩니다. 계수 정렬은 비교 연산을 하지 않고 값의 인덱스 직접 참조(Direct Indexing)를 사용하는 **비비교 정렬(Non-comparison sort)**이기 때문입니다.

---

### 📌 Exercise 2.21 (중앙값, 평균, 그리고 로버스트 통계)
**[원문]**
(a) 중앙값 $\mu_1$이 $\sum |x_i - \mu|$ 를 최소화함을 증명.
(b) 평균 $\mu_2$가 $\sum (x_i - \mu)^2$ 를 최소화함을 증명.
(c) $\mu_\infty$ (미니맥스)를 $O(n)$ 시간에 구하는 법.

**[해설 및 증명]**
- **(a)** $\mu$를 좌측에서 우측으로 이동시킬 때, $\mu$보다 작은 원소 개수를 $L$, 큰 원소 개수를 $R$이라 하면 미분/기울기는 $L - R$ 입니다. $L = R$ 이 되는 지점이 바로 **중앙값(Median)**이며 이때 함수가 최솟값을 가집니다.
- **(b)** $f(\mu) = \sum (x_i - \mu)^2$ 미분: $f'(\mu) = -2 \sum (x_i - \mu) = 0 \implies n \mu = \sum x_i \implies \mathbf{\mu = \frac{1}{n} \sum x_i = \mu_2}$ (평균).
- **(c)** $f(\mu) = \max_i |x_i - \mu|$ 를 최소화하는 값은 최솟값과 최댓값의 정확한 중간점입니다:
  $$\mu_\infty = \frac{\min_i x_i + \max_i x_i}{2}$$
  - 배열의 $\min$과 $\max$는 단 1회 순회로 $O(n)$ 시간에 구할 수 있으므로 $\mu_\infty$ 도 **$O(n)$** 시간에 계산됩니다.

---

### 📌 Exercise 2.22 (두 정렬 배열의 $k$번째 작은 원소)
**[원문]**
Find $k$th smallest element in union of two sorted lists of size $m$ and $n$ in $O(\log m + \log n)$ time.

**[해설 및 증명]**
- 이진 탐색 기법: 각 배열에서 약 $k/2$번째 원소 $A[k/2]$ 와 $B[k/2]$ 를 비교합니다.
- 만약 $A[k/2] < B[k/2]$ 이면, $A$의 앞쪽 $k/2$개 원소는 절대로 전체 $k$번째 작은 원소가 될 수 없으므로 제거하고 $k = k - k/2$ 로 갱신합니다.
- 매 단계마다 $k$가 절반으로 줄어들므로 총 시간 복잡도는 $\mathbf{O(\log m + \log n)}$ 입니다.

---

### 📌 Exercise 2.23 (다수 원소 Majority Element 찾기)
**[원문]**
(a) $O(n \log n)$ 분할 정복 알고리즘.
(b) $O(n)$ 선형 시간 알고리즘 (Boyer-Moore Majority Vote).

**[해설 및 증명]**
- **(a) $O(n \log n)$ 알고리즘**:
  - 배열을 반으로 나누어 좌측 다수 원소 $m_1$과 우측 다수 원소 $m_2$를 재귀적으로 찾습니다.
  - 전체 배열에서 $m_1$과 $m_2$의 개수를 직접 카운트($O(n)$)하여 과반수( $> n/2$ )를 넘는 원소를 반환합니다.
  - 재귀식 $T(n) = 2T(n/2) + O(n) \implies \mathbf{O(n \log n)}$.
- **(b) $O(n)$ 알고리즘 (Boyer-Moore)**:
  - 원소들을 2개씩 짝짓습니다. 두 원소가 다르면 둘 다 버리고, 같으면 하나만 남깁니다.
  - 과반수 원소가 존재한다면 이 제거 과정을 거쳐도 끝까지 살아남게 됩니다.
  - 1회 순회로 후보를 선정 후, 2번째 순회로 검증하여 **$O(n)$** 에 해결합니다.

---

### 📌 Exercise 2.24 (QuickSort 분석)
**[원문]**
(a) Pseudocode (b) 최악 시간복잡도 $\Theta(n^2)$ (c) 평균 시간복잡도 $O(n \log n)$ 유도.

**[해설 및 증명]**
- **(b) 최악의 경우**: 피벗이 항상 최솟값 또는 최댓값으로 선택되면 재귀식이 $T(n) = T(n-1) + O(n)$ 이 되어 $\sum_{i=1}^n i = \mathbf{\Theta(n^2)}$.
- **(c) 평균의 경우**: 모든 피벗 위치가 균등 확률 $1/n$ 을 가짐:
  $$T(n) = O(n) + \frac{2}{n} \sum_{i=1}^{n-1} T(i)$$
  이 재귀식을 풀면 $T(n) \le c n \ln n \implies \mathbf{O(n \log n)}$.

---

### 📌 Exercise 2.25 (10진수 ➔ 2진수 변환 분할 정복)
**[원문]**
(a) $10^n$ 의 이진수 변환 `pwr2bin(n)`.
(b) $n$자리 10진수 $x$의 이진수 변환 `dec2bin(x)`.

**[해설 및 증명]**
- **(a)** `z = pwr2bin(n/2)` 로 구한 뒤 `return fastmultiply(z, z)`.
  - 재귀식: $T(n) = T(n/2) + O(n^{\log_2 3}) \implies \mathbf{O(n^{1.585})}$.
- **(b)** $x = x_L \cdot 10^{n/2} + x_R$ 로 분할.
  - `return add(fastmultiply(dec2bin(xL), pwr2bin(n/2)), dec2bin(xR))`
  - 재귀식: $T(n) = 2T(n/2) + O(n^{1.585}) \implies \mathbf{O(n^{1.585})}$.

---

### 📌 Exercise 2.26 (제곱 연산과 곱셈 연산의 점근적 속도)
**[원문]**
Is squaring an $n$-bit integer asymptotically faster than multiplying two $n$-bit integers?

**[해설 및 증명]**
- **아닙니다 (거짓)**.
- 곱셈을 통해 제곱을 할 수 있음은 자명하며($x^2 = x \times x$), 반대로 항등식 $xy = \frac{(x+y)^2 - x^2 - y^2}{2}$ 를 이용하면 **제곱 알고리즘 3번으로 두 수의 곱셈을 구현**할 수 있습니다.
- 따라서 정수 제곱과 두 정수의 곱셈은 상수 배 차이만 날 뿐 **동일한 점근적 시간 복잡도 $\Theta(M(n))$** 를 가집니다.

---

### 📌 Exercise 2.27 (행렬 제곱과 행렬 곱셈의 동치성)
**[원문]**
(a) $2 \times 2$ 행렬 제곱에 5번 곱셈 충분 증명. (b) Strassen 적용 오류 지적. (c) $S(n) = O(n^c) \iff M(n) = O(n^c)$ 증명.

**[해설 및 증명]**
- **(b) 오류 원인**: 슈트라센 알고리즘은 하위 블록의 곱셈에서 서로 다른 서브 행렬 간의 곱(예: $A \cdot E$)을 수행해야 하므로 **단순 제곱 알고리즘을 하위 문제에 재귀 적용할 수 없습니다**.
- **(c) 증명**:
  - $A = \begin{bmatrix} X & 0 \\ 0 & 0 \end{bmatrix}, B = \begin{bmatrix} 0 & Y \\ 0 & 0 \end{bmatrix}$ 이라 두면, $AB + BA = \begin{bmatrix} 0 & XY \\ 0 & 0 \end{bmatrix}$ 입니다.
  - 항등식 $AB + BA = (A+B)^2 - A^2 - B^2$ 에 의해, $2n \times 2n$ 행렬의 제곱 3번으로 $XY$를 구할 수 있습니다.
  - 따라서 행렬 제곱이 $O(n^c)$ 이면 일반 행렬 곱셈도 **$O(n^c)$** 입니다. $\blacksquare$

---

### 📌 Exercise 2.28 (Hadamard 행렬-벡터 곱셈)
**[원문]**
Show that $H_k v$ can be calculated using $O(n \log n)$ operations for $n = 2^k$.

**[해설 및 증명]**
- $v = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}$ 로 반씩 나누면:
  $$H_k v = \begin{bmatrix} H_{k-1} & H_{k-1} \\ H_{k-1} & -H_{k-1} \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} H_{k-1}v_1 + H_{k-1}v_2 \\ H_{k-1}v_1 - H_{k-1}v_2 \end{bmatrix}$$
- $H_{k-1}v_1$ 과 $H_{k-1}v_2$ 를 재귀적으로 계산한 후 덧셈/뺄셈 $O(n)$ 수행.
- 재귀식: $T(n) = 2 T(n/2) + O(n) \implies \mathbf{O(n \log n)}$.

---

### 📌 Exercise 2.29 (Horner의 법칙 다항식 평가)
**[원문]**
(a) Horner's rule 구현. (b) 덧셈/곱셈 횟수 분석.

**[해설 및 증명]**
- **(a) 알고리즘**: $p(x) = (\dots((a_n x + a_{n-1})x + a_{n-2})\dots)x + a_0$
- **(b) 연산 횟수**: 정확히 **곱셈 $n$ 번, 덧셈 $n$ 번** 소요 ➔ $O(n)$.
  - 일반적인 전개 방식($O(n^2)$) 대비 최적의 연산 횟수입니다.

---

### 📌 Exercise 2.30 (모듈로 아리스메틱 Fourier Transform - Modular FFT)
**[원문]**
Modulo 7 에서의 푸리에 변환 및 다항식 곱셈.

**[해설 및 증명]**
- **(a)** $\omega = 3$ 선택: $3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1 \pmod 7$.
  - 합: $\sum_{i=1}^6 3^i = 3 + 2 + 6 + 4 + 5 + 1 = 21 \equiv \mathbf{0 \pmod 7}$.
- **(b) ~ (d)**: 복소수 영역의 $e^{-i 2\pi/n}$ 대신 모듈로 원시근 $\omega = 3$ 을 사용하여 동일하게 FFT 곱셈 수행.

---

### 📌 Exercise 2.31 (이진 최대공약수 알고리즘 - Binary GCD)
**[원문]**
(a) Binary GCD 규칙 증명. (b) 분할 정복 알고리즘 및 유클리드 대비 효율성.

**[해설 및 증명]**
- **(a) 규칙**:
  - $a, b$ 모두 짝수: $\gcd(a, b) = 2 \gcd(a/2, b/2)$
  - $a$ 홀수, $b$ 짝수: $\gcd(a, b) = \gcd(a, b/2)$
  - $a, b$ 모두 홀수: $\gcd(a, b) = \gcd((a-b)/2, b)$
- **(c) 효율성**: $n$비트 정수 연산 시, 나눗셈(유클리드) 대신 비트 시프트(Shift)와 뺄셈만 사용하므로 비트 연산 단위에서 훨씬 빠르고 단순하게 동작합니다.

---

### 📌 Exercise 2.32 (가장 가까운 두 점 찾기 - Closest Pair of Points)
**[원문]**
(a) $d \times d$ 영역 내 최대 4개 점 증명. (b) 알고리즘 정당성. (c) $T(n) = 2T(n/2) + O(n \log n) \implies O(n \log^2 n)$. (d) $O(n \log n)$ 으로 개선.

**[해설 및 증명]**
- **(a)** 좌측/우측 각 영역 내 점들 간 거리는 최소 $d$ 이상입니다. 크기 $d/2 \times d/2$ 인 4개 소영역에는 각각 최대 1개의 점만 존재 가능하므로 $d \times d$ 영역 내에는 **최대 4개의 점**만 존재할 수 있습니다.
- **(c) 시간복잡도**: $T(n) = 2T(n/2) + O(n \log n) \implies \mathbf{O(n \log^2 n)}$.
- **(d) $O(n \log n)$ 개선법**: 최초에 $y$좌표로 정렬된 배열을 미리 만들어 두고 재귀 호출 시 병합 정렬(MergeSort) 방식으로 $y$좌표 정렬 상태를 유지하면, 경계 영역 점 정렬을 $O(n)$ 시간에 마쳐 $T(n) = 2T(n/2) + O(n) \implies \mathbf{O(n \log n)}$ 으로 줄일 수 있습니다. $\blacksquare$

---

## 🧠 4. [4단계 복습용 정리]: 핵심 시간복잡도 패턴 한눈에 보기

```text
[Dasgupta Ch 2 핵심 알고리즘 복잡도]
  ├── Karatsuba 곱셈 ───────> O(n^1.585)   (T(n) = 3T(n/2) + O(n))
  ├── Strassen 행렬곱 ──────> O(n^2.81)    (T(n) = 7T(n/2) + O(n²))
  ├── Fast Fourier (FFT) ───> O(n log n)   (T(n) = 2T(n/2) + O(n))
  ├── QuickSort (평균) ─────> O(n log n)   (최악 O(n²))
  ├── Median Finding ───────> O(n)         (T(n) = T(n/5) + T(7n/10) + O(n))
  └── Closest Pair ─────────> O(n log n)   (y-정렬 병합 유지 시)
```
