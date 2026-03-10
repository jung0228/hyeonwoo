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
.prob-ans{margin-top:1rem;padding:1.2rem 1.4rem;background:#f4f2ec;border-radius:7px;font-size:.96rem;line-height:1.8;display:none}
.prob-ans.open{display:block}
.prob-ans p{margin-bottom:.8rem}
.prob-ans p:last-child{margin-bottom:0}
.kw{background:#fef3c7;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw2{background:#d4f7d4;padding:.05em .35em;border-radius:3px;font-weight:600}
.kw3{background:#ffd6d6;padding:.05em .35em;border-radius:3px;font-weight:600}
.prob-formula{background:#e8e5de;padding:.65rem 1rem;border-radius:5px;margin:.6rem 0;font-size:.94rem;overflow-x:auto}
</style>
<script>
function tp(btn){var a=btn.nextElementSibling;var o=a.classList.toggle('open');btn.textContent=o?'답안 닫기 ▲':'모범 답안 보기 ▾';}
</script>

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
