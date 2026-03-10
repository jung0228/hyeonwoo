---
title: 고유값의 성질 심화 문제
dek: Hessian 스펙트럼, RNN 그래디언트 흐름, Spectral Normalization, PSD 행렬 — 대학원 수준 서술형 문제 5선.
desc: 고유값의 성질 포스트의 심화 버전. SAM, RNN 그래디언트 소실/폭발, Spectral Normalization, Power Iteration, PSD와 커널 방법을 다룬다.
tags: [Math]
date: Mar 2026
readtime: 20 min read
slug: eigenvalue-properties-problems
katex: true
---

이 포스트는 [고유값의 성질](eigenvalue-properties.html)의 심화 문제집이다. 딥러닝 최적화, RNN, GAN 안정화, 커널 방법 — 고유값이 현대 딥러닝 이론 전반에 걸쳐 어떻게 등장하는지 깊이 탐구한다.

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
<div class="prob-meta"><span class="prob-num">기초 1</span><span class="prob-tag">대칭 행렬 · 실수 고유값</span></div>
<div class="prob-q">실수 대칭 행렬 $A = A^T$의 고유값이 항상 실수임을 증명하라. 이 성질이 공분산 행렬과 Hessian 행렬에서 왜 필수적인지 각각 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>용어 정리부터.</strong></p>

<p><span class="kw">고유쌍 $(\lambda, v)$</span>이란 행렬 $A$에 대해 $Av = \lambda v$를 만족하는 스칼라 $\lambda$(고유값)와 벡터 $v$(고유벡터)의 쌍이다. 예를 들어 $A = \begin{pmatrix}3&1\\1&3\end{pmatrix}$이면 $\lambda=4$, $v=\begin{pmatrix}1\\1\end{pmatrix}$이 하나의 고유쌍이다. $Av = \begin{pmatrix}4\\4\end{pmatrix} = 4v$ ✓</p>

<p><span class="kw">켤레(conjugate)</span>란 복소수에서 허수부의 부호를 바꾸는 것이다. $z = a+bi$이면 $\bar{z} = a-bi$. 실수는 켤레가 자기 자신: $\bar{3} = 3$. 복소 벡터 $v$의 켤레는 각 성분에 켤레를 취한다.</p>

<p><span class="kw">켤레 전치 $v^*$</span>란 벡터(또는 행렬)를 전치한 뒤 각 원소에 켤레를 취하는 것이다. 실수 벡터면 그냥 전치와 같다. 예를 들어 $v = \begin{pmatrix}1+i \\ 2\end{pmatrix}$이면 $v^* = \begin{pmatrix}1-i & 2\end{pmatrix}$. 실수 벡터 $v = \begin{pmatrix}3 \\ 4\end{pmatrix}$이면 $v^* = \begin{pmatrix}3 & 4\end{pmatrix}$ — 그냥 행벡터.</p>

<p><span class="kw">$v^* A v$</span>는 실수 벡터일 때 $v^T A v$와 같다. 이것은 스칼라(숫자 하나)를 만든다. 예를 들어 $v = \begin{pmatrix}1\\1\end{pmatrix}$, $A = \begin{pmatrix}3&1\\1&3\end{pmatrix}$이면:</p>
<div class="prob-formula">$$v^T A v = \begin{pmatrix}1&1\end{pmatrix}\begin{pmatrix}3&1\\1&3\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}1&1\end{pmatrix}\begin{pmatrix}4\\4\end{pmatrix} = 8$$</div>

<p><strong>증명 본론: 왜 대칭 행렬의 고유값은 항상 실수인가?</strong></p>

<p>고유값 $\lambda$가 혹시 복소수일 수도 있다고 가정하자. $Av = \lambda v$라 할 때, $v^* Av$를 두 가지 방법으로 계산해서 모순을 이끌어낸다.</p>

<p><strong>방식 1:</strong> $Av = \lambda v$를 대입한다:</p>
<div class="prob-formula">$$v^* A v = v^*(\lambda v) = \lambda(v^* v) = \lambda \|v\|^2$$</div>
<p>여기서 $\|v\|^2 = v^* v = |v_1|^2 + |v_2|^2 + \cdots > 0$ (항상 양의 실수).</p>

<p><strong>방식 2:</strong> $A$가 실수 대칭($A = A^T$)이면 켤레 전치도 $A^* = A$. 이것을 이용하면:</p>
<div class="prob-formula">$$(v^* A v)^* = v^* A^* (v^*)^* = v^* A v$$</div>
<p>즉 $v^* Av$라는 숫자는 자기 자신의 켤레와 같다 → <span class="kw2">$v^* Av$는 반드시 실수</span>다. (복소수 $z = \bar{z}$이면 허수부가 0이므로 실수.)</p>

<p><strong>결론:</strong> 방식 1에서 $\lambda \|v\|^2$가 실수이고, $\|v\|^2 > 0$이므로 $\lambda$는 실수. ∎</p>

<p><strong>Hessian 행렬이 뭔가?</strong> Hessian은 다변수 함수의 "이차 미분 행렬"이다. 손실 함수 $\mathcal{L}(\theta_1, \theta_2, \ldots)$에서 각 파라미터 쌍에 대한 이차 편미분을 모은 것: $H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$. 1변수 함수에서 이차 미분이 "볼록한지 오목한지"를 알려주듯, Hessian의 고유값이 그 역할을 한다. 모든 고유값 > 0 → 극소점(loss가 주변보다 낮음), 음수 고유값 존재 → 안장점(saddle point). 고유값이 복소수면 이런 해석 자체가 불가능하다.</p>

<p><strong>공분산 행렬에서의 중요성.</strong> 공분산 행렬은 대칭이므로 고유값이 실수이고, 추가로 모두 $\geq 0$ (PSD)다. 이 덕분에 PCA의 주성분(고유벡터)이 서로 직교하고 각 방향의 분산(고유값)이 명확한 실수값으로 정의된다. 고유값이 복소수면 "이 방향으로 데이터가 3.2만큼 퍼져있다"는 해석 자체가 불가능해진다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 2</span><span class="prob-tag">멱등 행렬 · 사영</span></div>
<div class="prob-q">$A^2 = A$를 만족하는 멱등 행렬(idempotent matrix)의 고유값이 0 또는 1만 가능함을 증명하라. 이것이 사영(projection) 연산의 직관과 어떻게 연결되는지 설명하고, 딥러닝에서 자기 자신에게 두 번 곱해도 같은 연산(예: 특정 마스킹 레이어)이 이 성질을 어떻게 활용하는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>증명.</strong> $Av = \lambda v$이면 $A^2 v = A(Av) = A(\lambda v) = \lambda(Av) = \lambda^2 v$. 멱등 조건 $A^2 = A$이므로 $A^2 v = Av$, 즉 $\lambda^2 v = \lambda v$, $(\lambda^2 - \lambda)v = 0$. $v \neq 0$이므로 $\lambda(\lambda-1) = 0$, 따라서 $\lambda = 0$ 또는 $\lambda = 1$. ∎</p>

<p><strong>사영의 직관.</strong> 사영 연산은 "이미 사영된 벡터를 다시 사영하면 변화 없음"이라는 성질을 가진다: $P(Pv) = Pv$. 고유벡터 관점에서: $\lambda=1$인 고유벡터는 사영 후 그대로 남는 벡터(사영 부분공간 내의 벡터), $\lambda=0$인 고유벡터는 사영 시 0이 되는 벡터(사영 부분공간에 수직인 벡터)다.</p>

<p><strong>딥러닝 연결.</strong> Attention에서 causal masking: 미래 토큰을 마스킹하는 연산은 멱등적이다(두 번 마스킹해도 한 번과 같다). Dropout: 학습 시 특정 뉴런을 끄는 마스킹도 $\{0,1\}$ 값의 대각 행렬이므로 멱등적. 또한 LayerNorm의 투영 성분도 멱등성에 가까운 구조를 가진다. 멱등성은 연산의 안정성을 보장한다 — 반복 적용해도 결과가 폭발하거나 소멸하지 않는다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 3</span><span class="prob-tag">직교 행렬 · 단위원 · 거리 보존</span></div>
<div class="prob-q">직교 행렬 $Q$ ($Q^T Q = I$)의 고유값의 절댓값이 모두 1임을 증명하라. 이것이 직교 변환이 벡터의 길이와 각도를 보존한다는 성질과 어떻게 연결되는지 설명하라. 딥러닝에서 직교 초기화(orthogonal initialization)가 학습 초기의 그래디언트 흐름을 안정화하는 이유는?</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>증명.</strong> $Qv = \lambda v$이면 $\|Qv\|^2 = (Qv)^T(Qv) = v^T Q^T Q v = v^T v = \|v\|^2$. 한편 $\|Qv\|^2 = \|\lambda v\|^2 = |\lambda|^2 \|v\|^2$. 따라서 $|\lambda|^2 = 1$, 즉 $|\lambda| = 1$. ∎ (실수 직교 행렬의 고유값은 $\pm 1$ 또는 켤레 복소수 쌍 $e^{\pm i\theta}$.)</p>

<p><strong>거리 보존과의 연결.</strong> 위 증명에서 이미 $\|Qv\| = \|v\|$ — 직교 행렬은 벡터의 길이를 보존한다. 또한 $\langle Qu, Qv \rangle = u^T Q^T Q v = u^T v$ — 내적(= 각도와 길이)도 보존된다. 모든 고유값의 크기가 1이라는 것은 이 변환이 "회전 또는 반사"만 일으키고, 어떤 방향도 늘리거나 줄이지 않는다는 뜻이다.</p>

<p><strong>직교 초기화의 효과.</strong> 딥러닝에서 $L$층짜리 네트워크의 순전파 $h_L = W_L \cdots W_1 x$에서 모든 $W_i$가 직교 행렬이면 $\|h_L\| = \|x\|$ — 신호가 층을 거쳐도 크기가 보존된다. 역전파에서도 마찬가지로 그래디언트 크기가 보존되어 <strong>소실/폭발 없이 안정적으로 흐른다</strong>. 물론 학습 과정에서 직교성이 깨지지만, 좋은 초기화는 학습 초반의 안정적인 그래디언트 흐름을 보장한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 4</span><span class="prob-tag">Trace · 고유값의 합 · PCA 분산</span></div>
<div class="prob-q">$\text{tr}(A) = \sum_i \lambda_i$임을 이용하여, PCA에서 상위 $k$개의 주성분이 설명하는 분산의 비율을 공분산 행렬의 고유값으로 표현하라. 또한 잔차 연결(residual connection)이 있는 네트워크 $F(x) = x + f(x)$에서 $F$의 야코비안(Jacobian)의 고유값이 $f$의 야코비안 고유값으로부터 어떻게 결정되는지 설명하고, 이것이 왜 그래디언트 흐름에 유리한지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>PCA 설명 분산 비율.</strong> 공분산 행렬 $C$의 고유값 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$. 총 분산 = $\text{tr}(C) = \sum_i \lambda_i$ (모든 변수의 분산의 합). 상위 $k$개 주성분이 설명하는 분산 비율:</p>
<div class="prob-formula">$$\text{설명된 분산 비율} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = \frac{\sum_{i=1}^k \lambda_i}{\text{tr}(C)}$$</div>
<p>PCA 스크리 플롯(scree plot)에서 x축이 $k$, y축이 이 누적 비율이며 "꺾임점"에서 $k$를 선택한다.</p>

<p><strong>잔차 연결과 야코비안.</strong> $F = I + J_f$ (야코비안). $F$의 고유값 $= 1 + \lambda_i(J_f)$. $J_f$의 고유값이 $-1$ 근방(예: -0.9)이어도 $F$의 고유값은 $0.1$ — 소멸 위험이 있다. 하지만 대부분의 경우 $J_f$의 고유값이 작은 값이면 $F$의 고유값 $\approx 1$. 즉 <strong>잔차 연결은 야코비안 고유값을 1 근방에 고정하여 그래디언트가 층을 거쳐도 크기가 보존되도록 보장</strong>한다. ResNet의 성공이 이 원리에 기반한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 5</span><span class="prob-tag">PD · PSD · 공분산</span></div>
<div class="prob-q">양의 정부호(Positive Definite, PD) 행렬과 양의 반정부호(Positive Semi-Definite, PSD) 행렬을 고유값 조건과 이차형식(quadratic form) $v^T A v$으로 정의하라. 공분산 행렬이 반드시 PSD이지만 항상 PD는 아닐 수 있는 이유를 설명하고, PD가 아닌 경우(singular 공분산 행렬)가 실전에서 어떤 문제를 일으키는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>정의.</strong></p>
<p>PSD: 모든 $v \neq 0$에 대해 $v^T A v \geq 0$ ↔ 모든 고유값 $\lambda_i \geq 0$.</p>
<p>PD: 모든 $v \neq 0$에 대해 $v^T A v > 0$ ↔ 모든 고유값 $\lambda_i > 0$. (PD ⊂ PSD.)</p>

<p><strong>공분산 행렬이 PSD인 이유.</strong> $C = \frac{1}{n}X^TX$. 임의의 벡터 $v$에 대해:</p>
<div class="prob-formula">$$v^T C v = \frac{1}{n} v^T X^T X v = \frac{1}{n} \|Xv\|^2 \geq 0$$</div>
<p>∴ PSD. ∎ 이 불등식은 $Xv = 0$일 때 등호가 성립하므로 PD가 아닐 수 있다.</p>

<p><strong>Singular 공분산 행렬의 발생 조건.</strong> (a) 데이터 차원 $d >$ 샘플 수 $n$: 데이터 행렬이 rank $\leq n$이므로 $d$차원 공분산 행렬의 rank $\leq n < d$ → 0인 고유값 존재. (b) 완벽한 선형 종속 특성: 예를 들어 "키(cm) + 키(mm)/10 = 상수"인 특성이 있으면 해당 방향 분산 = 0.</p>

<p><strong>실전 문제.</strong> 가우시안 분포의 확률밀도함수 $p(x) = \frac{1}{\sqrt{(2\pi)^d \det(C)}} \exp(-\frac{1}{2}(x-\mu)^T C^{-1}(x-\mu))$에서 $C$가 singular면 $\det(C) = 0$이 되어 PDF가 정의되지 않고 $C^{-1}$도 존재하지 않는다. 해결책: (a) 특성 선택으로 종속 특성 제거, (b) 정규화 $C' = C + \epsilon I$ (모든 고유값에 $\epsilon$ 추가 → PD 보장), (c) PCA로 rank와 같은 수의 주성분만 사용.</p>

</div>
</div>

<div class="prob-lv">대학원 심화 문제 — 엄밀한 논증</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 1</span><span class="prob-tag">Hessian 스펙트럼 · Sharp/Flat Minima · SAM</span></div>
<div class="prob-q">훈련 손실의 Hessian $H = \nabla^2_\theta \mathcal{L}$의 고유값 스펙트럼이 최솟값의 기하학적 성질(sharp vs flat minima)을 어떻게 나타내는지 설명하라. Sharp minima가 일반화를 해친다는 Keskar et al.의 주장과 Dinh et al.의 반론을 정리하고, SAM(Sharpness-Aware Minimization)이 이를 어떻게 다루는지 수식으로 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Hessian 고유값과 minima의 기하학.</strong> 2차 Taylor 전개 $\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \delta^T \nabla \mathcal{L} + \frac{1}{2}\delta^T H \delta$에서, 최솟값 근방($\nabla \mathcal{L} = 0$)에서 손실의 변화는 $\frac{1}{2}\delta^T H \delta$로 결정된다. $H$의 <span class="kw">최대 고유값 $\lambda_\max$</span>이 크면 특정 방향으로 조금만 이동해도 손실이 급격히 증가하는 "날카로운(sharp)" 최솟값이고, $\lambda_\max$가 작으면 넓고 평평한(flat) 최솟값이다.</p>

<p><strong>Keskar et al.(2016)의 주장.</strong> 큰 배치 크기(large-batch) SGD는 sharp minima에 수렴하고, 작은 배치 크기(small-batch)는 flat minima에 수렴한다. Sharp minima는 훈련 데이터에 과도하게 특화되어 있어 테스트 분포와의 미세한 차이에도 민감하게 반응 → <span class="kw3">일반화 성능 저하</span>. Flat minima는 파라미터 공간에서 넓은 영역이 낮은 손실을 유지하므로 분포 이동에 강건하다.</p>

<p><strong>Dinh et al.(2017)의 반론.</strong> ReLU 네트워크에서는 가중치를 스케일링해도 함수가 동일하게 유지된다: $\text{ReLU}(cx) = c\,\text{ReLU}(x)$. 따라서 flat minima를 파라미터를 재스케일링하여 임의로 sharp minima로 변환할 수 있다. 같은 함수를 표현하는 파라미터임에도 Hessian 고유값이 달라지므로, <span class="kw">sharpness가 일반화의 진정한 척도가 아닐 수 있다</span>는 것이다.</p>

<p><strong>SAM(Sharpness-Aware Minimization).</strong> SAM은 sharpness 논쟁에 실용적으로 응답한다. 목적 함수:</p>
<div class="prob-formula">$$\min_\theta \max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}(\theta + \epsilon) \;\approx\; \min_\theta \mathcal{L}\!\left(\theta + \rho\frac{\nabla_\theta \mathcal{L}}{\|\nabla_\theta \mathcal{L}\|}\right)$$</div>
<p>$\rho$-ball 내 최악의 perturbation에서도 손실이 낮은 파라미터를 찾는 minimax 문제다. 내부 최대화의 근사 해는 그래디언트 방향으로 $\rho$만큼 이동한 점이고, 이 점에서 다시 그래디언트를 계산하여 파라미터를 업데이트한다. <span class="kw2">직관적으로 SAM은 주변 $\rho$-ball 내 모든 이웃 파라미터에서도 손실이 낮은, 즉 "flat한" 최솟값을 찾는다</span>. Dinh et al.의 스케일 불변성 문제는 최근 m-SAM 등의 수정 버전이 다루고 있다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 2</span><span class="prob-tag">RNN · 그래디언트 소실/폭발 · 스펙트럼 반경</span></div>
<div class="prob-q">RNN의 BPTT(Backpropagation Through Time)에서 그래디언트가 스펙트럼 반경 $\rho(W) = \max_i |\lambda_i(W)|$에 의해 어떻게 결정되는지 $T$ 스텝 역전파의 관점에서 수식으로 분석하라. LSTM의 구조적 해결책을 Hessian 관점에서 설명하고, Spectral Normalization이 이와 어떻게 연결되는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>BPTT의 그래디언트 분석.</strong> 단순 RNN $h_t = \sigma(W h_{t-1} + U x_t)$에서 $T$ 스텝에 걸친 그래디언트는:</p>
<div class="prob-formula">$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^T \text{diag}(\sigma'(z_t)) \cdot W$$</div>
<p>$\sigma' \approx 1$로 가정하면 이 곱은 $W^T$에 비례한다. $W$의 스펙트럼 분해 $W = Q\Lambda Q^{-1}$를 대입하면 $W^T = Q\Lambda^T Q^{-1}$. 고유값 $|\lambda_i|$에 따라:</p>
<div class="prob-formula">$$|\lambda_i|^T \begin{cases} \to 0 & |\lambda_i| < 1 \quad \text{(그래디언트 소실)} \\ \to \infty & |\lambda_i| > 1 \quad \text{(그래디언트 폭발)} \end{cases}$$</div>
<p><span class="kw">스펙트럼 반경 $\rho(W) = 1$이 경계</span>다. 고유값이 하나라도 1을 넘으면 폭발, 모두 1 미만이면 소실이 일어나 긴 시퀀스에서 long-range dependency를 학습할 수 없다.</p>

<p><strong>LSTM의 구조적 해결.</strong> LSTM은 cell state $c_t$를 통해 <span class="kw2">덧셈 연산으로 그래디언트를 전파</span>한다: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. 그래디언트 흐름은 $\partial c_t/\partial c_{t-1} = f_t$ (forget gate). $f_t \in (0,1)^d$이지만 각 차원이 독립적으로 조절되므로, 필요한 정보 채널은 $f_t \approx 1$로 유지하여 그래디언트를 거의 손실 없이 전달할 수 있다. 이것은 "highway" 구조로, 잔차 연결(residual connection)의 전신이기도 하다.</p>

<p><strong>Spectral Normalization과의 연결.</strong> Spectral Normalization(Miyato et al., 2018)은 각 레이어의 가중치를 최대 특이값 $\sigma_\max(W)$으로 나누어 $\|W\|_2 = 1$을 강제한다. 이는 RNN의 그래디언트 폭발 방지와 정확히 같은 원리다: $\|W^T\|_2 \leq \|W\|_2^T = 1^T = 1$이 되어 순전파/역전파 모두 그래디언트 크기가 통제된다. GAN에서는 이것이 판별자의 <span class="kw">1-Lipschitz 조건</span>을 만족시키는 데 사용된다(다음 문제 참조).</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 3</span><span class="prob-tag">Spectral Normalization · Lipschitz · WGAN</span></div>
<div class="prob-q">Spectral Normalization이 왜 레이어의 함수를 1-Lipschitz로 만드는지 행렬 노름의 관점에서 증명하라. Wasserstein GAN에서 판별자가 1-Lipschitz여야 하는 이유를 Kantorovich-Rubinstein 쌍대성을 통해 설명하고, Gradient Penalty 대비 Spectral Normalization의 장단점을 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Spectral Normalization → 1-Lipschitz 증명.</strong> 선형 레이어 $f(x) = Wx$에 대해 Lipschitz 상수는 $\text{Lip}(f) = \sup_x \frac{\|Wx\|}{\|x\|} = \|W\|_2 = \sigma_\max(W)$ (스펙트럼 노름). Spectral Normalization은 $\hat{W} = W/\sigma_\max(W)$로 정규화하므로 $\|\hat{W}\|_2 = 1$이 되어 이 레이어는 1-Lipschitz다. 비선형 활성화 함수(ReLU, LeakyReLU 등)도 1-Lipschitz이므로, <span class="kw">Lipschitz 함수의 합성은 각 Lipschitz 상수의 곱</span>으로 전파: 모든 레이어가 1-Lipschitz이면 전체 네트워크도 1-Lipschitz다.</p>

<p><strong>WGAN의 Lipschitz 요구 조건.</strong> Wasserstein 거리(Earth Mover's Distance)의 Kantorovich-Rubinstein 쌍대 표현:</p>
<div class="prob-formula">$$W(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{x \sim q}[f(x)]$$</div>
<p>여기서 $\|f\|_L \leq 1$은 $f$가 1-Lipschitz 함수여야 한다는 제약이다. 판별자 $D = f$가 이 클래스에 속할 때 위 식의 최대값이 Wasserstein 거리가 된다. 따라서 WGAN은 이론적으로 판별자가 반드시 1-Lipschitz여야 올바른 거리를 추정할 수 있다.</p>

<p><strong>Gradient Penalty vs Spectral Normalization 비교.</strong> WGAN-GP(Gulrajani et al., 2017)의 Gradient Penalty는 <span class="kw2">보간된 샘플에서 그래디언트 노름이 1이 되도록 소프트 제약</span>을 추가한다 — 유연하지만 계산이 비싸고(추가 역전파 필요), 배치 크기에 민감하다. Spectral Normalization은 파라미터 정규화로 <span class="kw">명시적이고 층별 1-Lipschitz를 하드 보장</span>하며, 추가 역전파 없이 Power Iteration으로 $\sigma_\max$를 효율적으로 추정한다. 단점은 각 레이어의 표현 용량이 제한될 수 있다는 것이다 — 단일 특이값만 정규화하므로 다른 방향의 스케일이 불균형해질 수 있다. 실전에서는 Spectral Normalization이 구현이 단순하고 안정적이어서 더 널리 쓰인다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 4</span><span class="prob-tag">Power Iteration · 수렴 분석 · Deflation</span></div>
<div class="prob-q">Power Iteration이 지배적 고유벡터로 수렴하는 이유를 스펙트럼 분해를 통해 수학적으로 분석하라. 수렴 속도가 고유값 비율 $|\lambda_2/\lambda_1|$에 의존함을 보이고, 이를 극복하기 위한 Deflation 방법과 Lanczos 알고리즘의 핵심 아이디어를 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Power Iteration의 수렴 분석.</strong> 대칭 행렬 $A$의 고유분해 $A = Q\Lambda Q^T$에서 고유값 $|\lambda_1| > |\lambda_2| \geq \cdots$. 초기 벡터 $v_0 = \sum_i \alpha_i q_i$로 표현하면, $k$번 반복 후:</p>
<div class="prob-formula">$$\frac{A^k v_0}{\|A^k v_0\|} = \frac{\sum_i \alpha_i \lambda_i^k q_i}{\|\sum_i \alpha_i \lambda_i^k q_i\|} = \frac{\alpha_1 q_1 + \sum_{i\geq 2} \alpha_i (\lambda_i/\lambda_1)^k q_i}{\|\cdots\|}$$</div>
<p>$|\lambda_i/\lambda_1| < 1$ ($i \geq 2$)이므로 $k \to \infty$에서 $q_1$ 방향 외 모든 항이 지수 감쇠한다. 수렴 속도는 <span class="kw">$|\lambda_2/\lambda_1|^k$</span> — 이 비율이 1에 가까울수록(거의 중복 고유값) 수렴이 느리다.</p>

<p><strong>Deflation.</strong> 지배적 고유쌍 $(\lambda_1, q_1)$을 구한 후, 새 행렬 $A' = A - \lambda_1 q_1 q_1^T$를 구성한다. $A'$에서 $\lambda_1$에 해당하는 성분이 제거되므로 이제 $\lambda_2$가 지배적 고유값이 된다. 같은 과정을 반복하면 고유쌍을 순서대로 찾을 수 있다. 단점은 <span class="kw3">수치 오차가 누적</span>된다는 것 — 초기 단계의 오차가 이후 모든 고유쌍 계산에 전파된다.</p>

<p><strong>Lanczos 알고리즘.</strong> Krylov 부분공간 $\mathcal{K}_k = \text{span}(v_0, Av_0, A^2v_0, \ldots, A^{k-1}v_0)$을 구성하고, 이 부분공간 안에서 $A$를 삼대각(tridiagonal) 행렬로 투영한다. 크기 $k \times k$의 삼대각 행렬의 고유값은 원래 행렬의 극단 고유값(가장 크거나 작은 것들)에 빠르게 수렴한다. Deflation보다 수치적으로 안정적이며, <span class="kw2">상위 $k$개의 고유쌍을 $O(k \cdot \text{matrix-vector product})$ 비용으로 계산</span>할 수 있다. 대규모 희소 행렬(Transformer Hessian 추정, 그래프 라플라시안 등)에서 표준적으로 사용된다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 5</span><span class="prob-tag">PSD 행렬 · Mercer's Theorem · SVM 볼록성</span></div>
<div class="prob-q">커널 함수 $k(x, x')$가 유효한 커널이 되기 위한 필요충분조건이 그램 행렬(Gram matrix) $K_{ij} = k(x_i, x_j)$의 Positive Semi-Definiteness(PSD)임을 Mercer's Theorem의 관점에서 설명하라. PSD 행렬의 고유값이 모두 비음수임을 증명하고, 이것이 왜 SVM의 QP 문제가 볼록(convex)임을 보장하는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Mercer's Theorem과 유효 커널.</strong> Mercer 정리는 커널 함수 $k(x, x')$가 유효한 커널 — 즉 어떤 (무한 차원일 수 있는) 특징 공간 $\mathcal{H}$로의 사상 $\phi$가 존재하여 $k(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$인 것 — 의 필요충분조건은 <span class="kw">모든 유한 데이터 집합에 대해 그램 행렬이 PSD</span>인 것임을 말한다. 직관: 내적 $\langle \phi(x_i), \phi(x_j) \rangle$로 구성된 그램 행렬은 반드시 PSD여야 한다 ($v^T K v = \|\sum_i v_i \phi(x_i)\|^2 \geq 0$). 역으로 그램 행렬이 PSD이면 함수 $\phi$의 존재가 보장된다.</p>

<p><strong>PSD 행렬의 고유값이 비음수임 증명.</strong> $K$가 PSD이면 모든 벡터 $v$에 대해 $v^T K v \geq 0$. $v$를 고유벡터 $q$ ($Kq = \lambda q$)로 택하면:</p>
<div class="prob-formula">$$q^T K q = q^T (\lambda q) = \lambda \|q\|^2 \geq 0$$</div>
<p>$\|q\|^2 > 0$이므로 $\lambda \geq 0$. ∎</p>

<p><strong>SVM QP의 볼록성 보장.</strong> SVM의 쌍대 문제(dual problem)는 다음과 같은 이차 계획법(QP)이다:</p>
<div class="prob-formula">$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$</div>
<p>이것을 최소화로 바꾸면 목적 함수의 이차항 계수 행렬은 $Q_{ij} = y_i y_j K(x_i, x_j)$. $K$가 PSD이고 $y_i \in \{-1, +1\}$이면, 임의의 벡터 $u$에 대해:</p>
<div class="prob-formula">$$u^T Q u = \sum_{ij} u_i Q_{ij} u_j = (u \odot y)^T K (u \odot y) \geq 0$$</div>
<p>따라서 $Q$도 PSD이고, 이차 목적 함수는 <span class="kw2">볼록 함수</span>다. 볼록 QP는 지역 최솟값 = 전역 최솟값이 보장되며, 효율적인 내점법(interior point method) 등으로 다항 시간에 풀 수 있다. <span class="kw">커널이 유효하지 않으면(그램 행렬이 PSD가 아니면) QP가 비볼록해져 최적화가 불안정</span>해지는 것이 커널 유효성 조건의 실용적 의미다.</p>

</div>
</div>

<div class="footnote">
  이전 포스트: <a href="eigenvalue-properties.html">고유값의 성질</a> · 관련: <a href="eigendecomposition-svd-problems.html">고유값 분해 & SVD 심화 문제</a>
</div>
