---
title: 고유값 분해 & SVD 심화 문제
dek: PCA의 수치 안정성부터 LoRA, Nuclear Norm, Randomized SVD까지 — 대학원 수준 서술형 문제 5선.
desc: 고유값 분해 & SVD 포스트의 심화 버전. PCA 구현 비교, LoRA와 저계수 근사, Nuclear Norm, Randomized SVD, Attention 행렬 해석을 다룬다.
tags: [Math]
date: Mar 2026
readtime: 20 min read
slug: eigendecomposition-svd-problems
katex: true
---

이 포스트는 [고유값 분해와 SVD](eigendecomposition-svd.html)의 심화 문제집이다. 현대 딥러닝에서 SVD와 저계수 행렬 이론이 어떻게 실용적으로 활용되는지 — LoRA, 추천 시스템, Transformer 해석가능성 — 까지 다룬다.

<style>
.prob-block{background:#fff;border:1px solid #d8d8d2;border-radius:10px;padding:1.5rem 1.7rem;margin-bottom:1.3rem}
.prob-meta{display:flex;align-items:center;gap:.6rem;margin-bottom:.65rem;flex-wrap:wrap}
.prob-num{font-family:-apple-system,sans-serif;font-size:.72rem;font-weight:700;color:#fff;background:#1a56c4;padding:.18em .75em;border-radius:20px;letter-spacing:.04em}
.prob-tag{font-family:-apple-system,sans-serif;font-size:.72rem;color:#666;border:1px solid #ccc;padding:.15em .65em;border-radius:20px}
.prob-q{font-size:1.04rem;font-weight:600;line-height:1.65;margin-bottom:1rem}
.prob-toggle{border:1px solid #bbb;background:none;padding:.38rem 1.1rem;border-radius:5px;font-family:-apple-system,sans-serif;font-size:.82rem;cursor:pointer;color:#555}
.prob-toggle:hover{background:#f0eeea}
.prob-ans{margin-top:0;padding:0 1.4rem;background:#f4f2ec;border-radius:7px;font-size:.96rem;line-height:1.8;visibility:hidden;max-height:0;overflow:hidden}
.prob-ans.open{visibility:visible;max-height:9999px;margin-top:1rem;padding:1.2rem 1.4rem}.prob-lv{font-family:-apple-system,sans-serif;font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#888;margin:2rem 0 1rem;padding-bottom:.4rem;border-bottom:1px solid #d8d8d2}
.prob-ans p{margin-bottom:.8rem}
.prob-ans p:last-child{margin-bottom:0}
.kw{background:#fef3c7;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw2{background:#d4f7d4;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw3{background:#ffd6d6;padding:.05em .35em;border-radius:3px;font-weight:600}
.prob-formula{background:#e8e5de;padding:.65rem 1rem;border-radius:5px;margin:.6rem 0;font-size:.94rem;overflow-x:auto}
</style>
<script>
function tp(btn){var a=btn.nextElementSibling;var o=a.classList.toggle('open');btn.textContent=o?'답안 닫기 ▲':'모범 답안 보기 ▾';if(o&&window.renderMathInElement){renderMathInElement(a,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});}}
</script>

<div class="prob-lv">학부 기초 문제 — 개념과 직관</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 1</span><span class="prob-tag">고유값 · 고유벡터 · 기하학</span></div>
<div class="prob-q">행렬 $A = \begin{pmatrix}3 & 1 \\ 1 & 3\end{pmatrix}$의 고유값과 고유벡터를 손으로 구하라. 구한 결과를 이용하여 이 행렬이 나타내는 선형 변환이 2차원 벡터를 어떻게 변형시키는지 기하학적으로 설명하라. 특히 고유벡터 방향의 벡터는 어떻게 변환되는가?</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>고유값 계산.</strong> 특성 방정식 $\det(A - \lambda I) = 0$:</p>
<div class="prob-formula">$$(3-\lambda)^2 - 1 = 0 \implies \lambda^2 - 6\lambda + 8 = 0 \implies \lambda_1 = 4, \quad \lambda_2 = 2$$</div>

<p><strong>고유벡터.</strong> $\lambda_1 = 4$: $(A-4I)v = 0$ → $\begin{pmatrix}-1&1\\1&-1\end{pmatrix}v = 0$ → $v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}$. $\lambda_2 = 2$: → $v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}$.</p>

<p><strong>기하학적 의미.</strong> 고유벡터는 이 변환이 "늘리거나 줄이기만 하는" 방향이다. 대각선 방향 $v_1 = [1,1]/\sqrt{2}$은 4배로 늘어나고, 반대 대각선 방향 $v_2 = [1,-1]/\sqrt{2}$는 2배로 늘어난다. 일반적인 방향의 벡터는 이 두 성분으로 분해되어 각각 다른 비율로 늘어나므로 <strong>방향이 바뀐다</strong>. 하지만 고유벡터 방향으로 입력하면 방향은 유지되고 크기만 $\lambda$배 변한다. 이 행렬은 두 주축 방향으로 서로 다른 비율로 늘리는 타원형 확장 변환이다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 2</span><span class="prob-tag">SVD · 이미지 압축 · 에너지</span></div>
<div class="prob-q">흑백 이미지를 $512 \times 512$ 행렬 $A$로 표현할 때, SVD $A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$에서 상위 $k$개의 항만 유지하면 압축된 이미지 $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$를 얻는다. (1) $k=1, 10, 50$일 때 보존되는 "에너지" 비율 $\|\!A_k\!\|_F^2 / \|\!A\!\|_F^2$의 의미를 설명하고, (2) $k$를 크게 할수록 어떤 trade-off가 생기는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>(1) 에너지 비율의 의미.</strong> Frobenius 노름의 제곱은 $\|A\|_F^2 = \sum_{i,j} a_{ij}^2 = \sum_i \sigma_i^2$. 따라서:</p>
<div class="prob-formula">$$\frac{\|A_k\|_F^2}{\|A\|_F^2} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$</div>
<p>이 비율이 0.95면 "원본 이미지 에너지의 95%를 상위 $k$개 성분이 담고 있다"는 의미다. 자연 이미지는 특이값이 급격히 감소하는 경향이 있어 (저차원 구조), 작은 $k$로도 높은 에너지 보존이 가능하다. $k=10$이면 저주파 성분(전체적 형태)만, $k=50$이면 중주파 성분(세부 텍스처 일부)까지 포함된다.</p>

<p><strong>(2) Trade-off.</strong> 저장 비용: 원본 $512 \times 512 = 262,144$ 값. $A_k$ 표현: $k$개의 $\sigma_i$ + $k \times 512$개의 $u_i$ + $k \times 512$개의 $v_i$ = $k(1 + 1024)$ ≈ $1025k$ 값. $k < 255$이면 압축 효과. $k$가 커질수록 (a) 화질↑, (b) 압축률↓. 자연 이미지에서는 대체로 $k \approx 50$이면 육안으로 구분 어려울 정도의 품질을 유지하면서 상당한 압축이 가능하다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 3</span><span class="prob-tag">PCA · 분산 최대화 · 공분산</span></div>
<div class="prob-q">PCA의 첫 번째 주성분이 "데이터의 분산을 최대화하는 방향"이라고 한다. 단위 벡터 $w$로 데이터를 투영했을 때 분산이 $w^T C w$ (단, $C$는 공분산 행렬)임을 보이고, 이것을 최대화하는 $w$가 $C$의 최대 고유벡터인 이유를 라그랑주 승수법으로 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>투영 분산 유도.</strong> 중심화된 데이터 $x_i$의 $w$ 방향 투영: $z_i = w^T x_i$. 투영의 분산:</p>
<div class="prob-formula">$$\text{Var}(z) = \frac{1}{n}\sum_i (w^T x_i)^2 = w^T \left(\frac{1}{n}\sum_i x_i x_i^T\right) w = w^T C w$$</div>
<p>$C = \frac{1}{n}X^TX$가 공분산 행렬이므로 투영 분산 = $w^T C w$. ∎</p>

<p><strong>최대화 문제 (라그랑주 승수법).</strong> $\max_w w^T C w \quad \text{s.t.} \quad \|w\|^2 = 1$. 라그랑지안 $\mathcal{L} = w^T C w - \lambda(w^T w - 1)$. $\partial \mathcal{L}/\partial w = 2Cw - 2\lambda w = 0$, 즉 $Cw = \lambda w$. 이것은 고유값 방정식이다. 목적함수 값 = $w^T C w = w^T(\lambda w) = \lambda \|w\|^2 = \lambda$. 따라서 <strong>분산을 최대화하는 $w$는 $C$의 최대 고유값에 대응하는 고유벡터</strong>다. ∎</p>

<p><strong>직관.</strong> 공분산 행렬의 고유벡터는 데이터 구름이 가장 "늘어난" 방향이다. 고유값이 클수록 그 방향으로 데이터가 많이 퍼져 있다. PCA는 이 "가장 많이 퍼진" 방향들을 순서대로 찾는 알고리즘이다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 4</span><span class="prob-tag">행렬식 · 고유값 · 가역성</span></div>
<div class="prob-q">$\det(A) = \prod_i \lambda_i$임을 이용하여, 행렬이 가역(invertible)인 것과 고유값 중 0이 없는 것이 동치임을 설명하라. 신경망의 레이어 가중치 행렬 $W$가 rank-deficient(행렬식 = 0에 가까운)가 되면 어떤 문제가 발생하는지, 순전파(forward pass)와 역전파(backpropagation) 관점에서 각각 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>동치 증명.</strong> $\det(A) = \prod_i \lambda_i$이므로 $\det(A) = 0 \iff$ 어떤 $\lambda_i = 0$. 행렬이 가역 $\iff \det(A) \neq 0 \iff$ 모든 $\lambda_i \neq 0$. ∎ (고유값이 0이면 해당 고유벡터 방향으로의 정보가 완전히 손실되어 역변환이 불가능하다.)</p>

<p><strong>순전파 문제.</strong> $W$가 rank-deficient면 선형 변환이 저차원 공간으로 붕괴된다. 예를 들어 $W \in \mathbb{R}^{10 \times 10}$이 rank 3이면 10차원 입력이 사실상 3차원으로 압축되어 나머지 7차원의 정보는 소멸한다. 이것이 의도된 병목이 아니라면 표현력 손실이다.</p>

<p><strong>역전파 문제.</strong> $W$의 특이값이 0에 가까우면 역행렬의 특이값이 $1/\sigma_{\min} \to \infty$가 되어 수치적으로 불안정해진다(조건수 $\kappa = \sigma_\max/\sigma_\min \to \infty$). 그래디언트가 역전파될 때 이 레이어를 통과하면서 폭발적으로 증폭될 수 있다. 이것이 딥러닝에서 가중치 초기화, Batch Normalization, Spectral Normalization 등으로 가중치의 특이값 분포를 통제하는 이유다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 5</span><span class="prob-tag">Rank-1 행렬 · SVD · 외적</span></div>
<div class="prob-q">행렬 $A = uv^T$ ($u \in \mathbb{R}^m$, $v \in \mathbb{R}^n$, 단 $\|u\| = \|v\| = 1$)의 SVD를 쓰고 rank가 1임을 설명하라. 이 행렬은 어떤 입력 벡터에 대해서도 출력이 단일 방향 $u$만 가능하다는 것의 의미를 설명하고, attention head의 $W_{OV} = W_O W_V$가 rank-1에 가까울 때 이것이 어떤 단순한 기능을 수행하는지 연결하여 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>SVD 형태.</strong> $A = uv^T = \sigma_1 u_1 v_1^T$ where $\sigma_1 = \|u\|\|v\| = 1$, $u_1 = u$, $v_1 = v$. 이것이 단 하나의 특이삼중쌍 $(\sigma_1, u_1, v_1)$으로 이루어진 SVD다. 모든 다른 특이값은 0이므로 $\text{rank}(A) = 1$. ∎</p>

<p><strong>단일 출력 방향의 의미.</strong> 임의의 입력 $x$에 대해 $Ax = (uv^T)x = u(v^T x) = (v \cdot x) \cdot u$. 출력은 항상 $u$ 방향의 스칼라 배다. $v^T x$는 입력이 $v$ 방향으로 얼마나 정렬됐는지를 측정하고, 그 크기만큼 $u$ 방향으로 출력한다. 즉 <strong>$v$ 방향을 읽어서 $u$ 방향으로 쓰는 단순한 정보 전달</strong>이다.</p>

<p><strong>Attention head와의 연결.</strong> $W_{OV}$가 rank-1에 가까우면 그 head는 입력 잔차 스트림에서 $v$ 방향의 성분을 읽어 $u$ 방향으로 출력에 더한다는 단순한 역할을 한다. Mechanistic Interpretability 연구에서 이런 head들이 특정 토큰을 "복사"하거나 특정 특성(품사, 의미역 등)을 다음 위치로 전달하는 기능을 담당함이 발견되었다. Rank-1 구조는 해석가능성 분석을 크게 단순화한다.</p>

</div>
</div>

<div class="prob-lv">대학원 심화 문제 — 엄밀한 논증</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 1</span><span class="prob-tag">PCA · 수치적 안정성 · 조건수</span></div>
<div class="prob-q">PCA를 구현하는 두 가지 방법 — (a) 공분산 행렬 $C = X^TX/n$의 고유값 분해와 (b) 데이터 행렬 $X$에 직접 SVD — 의 차이를 설명하라. 특히 $d \gg n$ (고차원 소표본) 환경에서 수치적 안정성 차이가 왜 발생하는지 조건수(condition number)와 연결하여 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>두 방법의 수학적 동치성.</strong> $X$의 SVD를 $X = U\Sigma V^T$라 하면 공분산 행렬의 고유값 분해는:</p>
<div class="prob-formula">$$C = \frac{X^TX}{n} = \frac{V\Sigma^T U^T U \Sigma V^T}{n} = V\frac{\Sigma^2}{n}V^T$$</div>
<p>따라서 $C$의 고유벡터 = $V$, 고유값 = $\sigma_i^2/n$. 두 방법은 수학적으로 동치이다. 그러나 <span class="kw">수치 계산 과정에서 차이</span>가 생긴다.</p>

<p><strong>수치적 안정성 차이 — 조건수.</strong> 행렬의 조건수(condition number)는 $\kappa(A) = \sigma_\max/\sigma_\min$으로, 값이 클수록 수치 계산이 불안정하다. 방법 (a)에서 공분산 행렬 $C = X^TX/n$을 직접 형성하면 조건수가 $\kappa(C) = \kappa(X)^2$가 된다 — <span class="kw3">제곱으로 증가</span>한다. 예를 들어 $X$의 조건수가 $10^8$이면 $C$의 조건수는 $10^{16}$으로, 64비트 부동소수점의 머신 정밀도($\approx 10^{-16}$)와 비슷한 수준이 되어 수치 오차가 심각해진다. 방법 (b)에서 $X$에 직접 SVD를 적용하면 조건수가 $\kappa(X)$ 그대로 유지된다.</p>

<p><strong>$d \gg n$ 환경의 추가 이점.</strong> $d \gg n$이면 공분산 행렬 $C$는 $d \times d$ 행렬이어서 저장 비용이 $O(d^2)$에 달하며, 고유값 분해 비용도 $O(d^3)$이다. 반면 $X$는 $n \times d$ 행렬이므로 경제형 SVD(thin SVD)를 이용하면 상위 $k$개의 성분만 $O(ndk)$ 비용으로 계산할 수 있다. <span class="kw2">$d \gg n$ 환경에서는 $n \times n$ 크기의 $XX^T$에 대한 고유값 분해로 동일한 결과를 $O(n^3)$ 비용에 얻을 수도 있다</span>. 이것이 scikit-learn 등 실용 라이브러리에서 $n < d$일 때 자동으로 SVD 기반 구현을 선택하는 이유다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 2</span><span class="prob-tag">LoRA · Eckart-Young-Mirsky 정리</span></div>
<div class="prob-q">LoRA(Low-Rank Adaptation)는 사전 학습된 가중치 $W_0 \in \mathbb{R}^{d \times k}$를 고정하고, 업데이트 $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$)만 학습한다. 이것이 Eckart-Young-Mirsky 정리와 어떻게 연결되는지 설명하고, rank $r$의 선택이 만드는 trade-off를 수식으로 정량화하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Eckart-Young-Mirsky 정리.</strong> 행렬 $M$의 SVD를 $M = \sum_{i=1}^{\min(d,k)} \sigma_i u_i v_i^T$라 할 때, Frobenius 노름 기준으로 rank $r$ 행렬 중 최적 근사는:</p>
<div class="prob-formula">$$M_r = \sum_{i=1}^r \sigma_i u_i v_i^T, \quad \|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(d,k)} \sigma_i^2}$$</div>
<p>즉 상위 $r$개의 특이값/벡터를 유지하고 나머지를 버리는 것이 최적 저계수 근사다.</p>

<p><strong>LoRA와의 연결.</strong> LoRA의 전제는 "<span class="kw">사전 학습 모델의 업데이트 $\Delta W$는 본질적으로 낮은 내재 차원(intrinsic rank)을 가진다</span>"는 가설이다. 태스크에 맞는 업데이트가 전체 파라미터 공간이 아닌 저차원 부분공간에 집중된다면, Eckart-Young-Mirsky에 의해 이 업데이트를 rank $r$ 행렬 $BA$로 표현하는 것이 Frobenius 노름 기준 최적 근사다. 즉 LoRA는 암묵적으로 $\Delta W$의 SVD에서 상위 $r$개 성분만 학습하겠다는 가정에 기반한다.</p>

<p><strong>파라미터 효율성의 정량화.</strong> 전체 파인튜닝: $d \times k$ 파라미터. LoRA: $(d+k) \times r$ 파라미터. 압축비:</p>
<div class="prob-formula">$$\text{Compression ratio} = \frac{dk}{(d+k)r} \approx \frac{d}{2r} \quad (d \approx k \text{ 일 때})$$</div>
<p>GPT-3(175B)에서 $r=4$, $d=k=12288$이면 압축비 $\approx 1536$배. <strong>Trade-off</strong>: $r$이 작을수록 파라미터 수가 줄고 훈련이 빠르지만, 업데이트의 표현 용량이 제한된다. 태스크가 단순할수록(예: 특정 도메인 스타일 적용) 작은 $r$로 충분하고, 복잡한 능력(예: 새로운 추론 방식)을 획득하려면 큰 $r$이 필요하다. <span class="kw2">AdaLoRA</span>는 이 문제를 레이어별로 $r$을 적응적으로 조절하여 해결한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 3</span><span class="prob-tag">Nuclear Norm · Convex Relaxation · Matrix Completion</span></div>
<div class="prob-q">행렬 완성(Matrix Completion) 문제에서 rank 최소화는 NP-hard다. Nuclear norm $\|A\|_* = \sum_i \sigma_i(A)$이 rank 최소화의 볼록 완화(convex relaxation)임을 설명하라. 이것이 벡터의 L0 → L1 완화(Compressed Sensing)와 어떻게 대칭적인지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Rank 최소화의 어려움.</strong> $\text{rank}(A)$는 비零 특이값의 수: $\text{rank}(A) = |\{i : \sigma_i > 0\}| = \|\sigma\|_0$. 이것은 특이값 벡터의 L0 노름이다. L0 최소화는 조합론적 문제로 NP-hard이므로 직접 최적화가 불가능하다.</p>

<p><strong>L0 → L1, Rank → Nuclear Norm의 유사성.</strong> 벡터의 sparsity 유도에서 $\|\theta\|_0$을 $\|\theta\|_1$로 완화하듯, 행렬의 rank를 특이값의 L1 노름인 <span class="kw">Nuclear Norm $\|A\|_* = \sum_i \sigma_i$</span>으로 완화한다. Nuclear Norm은 특이값 공간에서의 L1 노름이며, 행렬 공간에서 rank의 <span class="kw2">볼록 포(convex hull)</span>에 해당한다 — 즉 rank 함수의 가장 타이트한 볼록 하한이다. 이 완화 덕분에 행렬 완성 문제를 반정치 프로그래밍(SDP, Semi-Definite Programming)으로 풀 수 있다:</p>
<div class="prob-formula">$$\min_A \|A\|_* \quad \text{s.t.} \quad A_{ij} = M_{ij} \text{ for observed entries } (i,j) \in \Omega$$</div>

<p><strong>회복 조건.</strong> Compressed Sensing에서 벡터 회복을 위해 측정 행렬이 RIP(Restricted Isometry Property)를 만족해야 하듯, 행렬 완성에서는 관측 패턴이 <span class="kw">비간섭(incoherence) 조건</span>을 만족해야 한다. 비간섭 조건은 대략 "행렬의 에너지가 특정 행/열에 집중되지 않아야 한다"는 것으로, 관측이 랜덤하게 이루어지면 높은 확률로 만족된다. Candès & Recht(2009)는 $m \times n$ rank-$r$ 행렬을 정확히 회복하기 위해 $O(rn\log n)$개의 무작위 관측만으로도 충분함을 보였다 — 전체 원소 수 $mn$에 비해 현저히 적다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 4</span><span class="prob-tag">Randomized SVD · Johnson-Lindenstrauss</span></div>
<div class="prob-q">전통적 SVD는 $m \times n$ 행렬에 $O(mn\min(m,n))$ 시간이 걸린다. Halko et al.(2011)의 Randomized SVD가 상위 $k$개의 성분을 어떻게 효율적으로 근사하는지 핵심 알고리즘을 설명하고, Johnson-Lindenstrauss 보조정리가 왜 이 근사가 정확한지를 보장하는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>핵심 아이디어: 무작위 범위 탐색.</strong> 행렬 $A$($m \times n$)의 상위 $k$개 성분을 찾으려면 $A$의 컬럼 공간(range) 중 "중요한" $k$차원 부분공간을 먼저 찾는 것이 핵심이다.</p>

<p><strong>알고리즘 (3단계).</strong></p>
<p>1단계 — 무작위 스케치: 가우시안 랜덤 행렬 $\Omega \in \mathbb{R}^{n \times (k+p)}$ ($p$는 약간의 oversampling)를 생성하고 $Y = A\Omega$를 계산한다. $Y$의 컬럼들은 $A$의 컬럼 공간을 무작위로 탐색한 결과다.</p>
<p>2단계 — 직교 기저: $Y$의 QR 분해 $Y = QR$을 통해 $A$의 approximate range를 나타내는 정규직교 기저 $Q$($m \times (k+p)$)를 얻는다.</p>
<p>3단계 — 저차원 SVD: $B = Q^T A$ ($k \times n$)를 계산하고 이 작은 행렬에 정확한 SVD $B = \hat{U}\Sigma V^T$를 적용한다. 최종 좌 특이벡터는 $U = Q\hat{U}$.</p>
<div class="prob-formula">$$\text{총 비용: } O(mn(k+p)) \approx O(mnk) \quad (k \ll \min(m,n))$$</div>

<p><strong>Johnson-Lindenstrauss 보조정리와의 연결.</strong> JL 보조정리는 고차원 벡터를 무작위 가우시안 행렬로 저차원에 투영해도 <span class="kw">내적(거리) 구조가 높은 확률로 보존된다</span>는 것을 보장한다. Randomized SVD에서 $\Omega$에 의한 투영 $Y = A\Omega$도 마찬가지로, $A$의 컬럼 공간 중 큰 특이값을 가진 방향(에너지가 집중된 방향)은 무작위 투영 후에도 $Y$의 컬럼 공간에서 잘 포착된다. 특이값이 급격히 감소하는(low-rank structure가 있는) 행렬에서 특히 이 근사가 정확하며, <span class="kw2">오차는 $(k+1)$번째 특이값 $\sigma_{k+1}$에 비례</span>한다. 실전에서 자연어 임베딩 행렬, 추천 시스템 행렬 등은 전형적으로 이런 구조를 가진다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 5</span><span class="prob-tag">Attention 행렬 · Mechanistic Interpretability</span></div>
<div class="prob-q">Transformer의 attention head에서 가중치 행렬 $W_{OV} = W_O W_V \in \mathbb{R}^{d \times d}$를 SVD로 분해하면 어떤 정보를 얻을 수 있는가? Mechanistic Interpretability 관점에서 rank-1 head가 어떤 특정 기능과 연결되는지 설명하고, 저계수 구조가 "회로(circuit)" 분석에 어떻게 활용되는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>$W_{OV}$의 의미와 SVD 분해.</strong> Self-attention에서 $W_{OV} = W_O W_V$는 "어떤 토큰에 주목했을 때 그 정보를 어떻게 변환하여 출력에 더할지"를 결정하는 행렬이다. SVD: $W_{OV} = U\Sigma V^T$. 상위 특이값 $\sigma_1$이 압도적으로 크다면 — 즉 <span class="kw">effective rank가 낮다면</span> — 이 head는 소수의 방향에만 집중된 단순한 선형 변환을 수행한다는 의미다.</p>

<p><strong>Rank-1 Head와 기능 해석.</strong> $W_{OV} \approx \sigma_1 u_1 v_1^T$인 경우, 이 head는 입력에서 $v_1$ 방향의 성분을 읽어 $u_1$ 방향으로 쓴다. Elhage et al.(2021)의 연구에서 <span class="kw2">induction head</span>가 이런 구조를 보인다: 이전 컨텍스트에서 패턴 $[A][B] \cdots [A]$를 인식하면 $[B]$를 예측하는 head로, 그 $W_{OV}$가 거의 rank-1에 가깝고 입력 토큰의 임베딩을 그대로 출력 공간에 "복사"하는 방향을 가진다. 마찬가지로 <span class="kw2">copying head</span>는 특정 위치의 토큰을 그대로 다음 위치에 복사하는 기능을 담당하며, $W_{OV}$가 identity에 가까운 저계수 구조를 보인다.</p>

<p><strong>회로(Circuit) 분석에서의 활용.</strong> 복잡한 언어 모델의 동작을 이해하기 위해 Mechanistic Interpretability는 여러 head와 MLP 레이어 사이의 <span class="kw">잔차 스트림(residual stream)</span> 상에서의 정보 흐름을 분석한다. SVD 분해를 통해 각 head가 잔차 스트림의 어느 부분공간을 읽고 쓰는지 파악하면, head들 사이의 구성(composition) — 한 head의 출력이 다음 head의 입력으로 사용되는 패턴 — 을 발견할 수 있다. 저계수 구조는 이 분석을 크게 단순화한다: 관련 없는 차원을 무시하고 핵심 부분공간만 추적하면 된다. 이 방법론은 간접 객체 식별(IOI), 수치 추론, 사실 검색(factual recall) 등 다양한 기능의 회로를 성공적으로 역설계하는 데 활용되었다.</p>

</div>
</div>

<div class="footnote">
  이전 포스트: <a href="eigendecomposition-svd.html">고유값 분해와 SVD</a> · 관련: <a href="eigenvalue-properties-problems.html">고유값의 성질 심화 문제</a>
</div>
