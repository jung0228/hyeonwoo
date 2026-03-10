---
title: MLE & MAP 심화 문제
dek: Fisher Information부터 EM 알고리즘까지 — 대학원 수준의 서술형 문제 5선.
desc: MLE & MAP 포스트의 심화 버전. Fisher Information, L1/L2 기하학, Empirical Bayes, EM 알고리즘, 베이즈 모델 선택을 다룬다.
tags: [Math]
date: Mar 2026
readtime: 20 min read
slug: mle-map-problems
katex: true
---

이 포스트는 [MLE & MAP 추정](mle-map.html)의 심화 문제집이다. 기본 개념을 숙지한 상태에서, 각 문제를 읽고 스스로 논증을 전개해본 뒤 모범 답안과 비교하자. 답안을 미리 열어보는 것보다, 30분 이상 직접 고민한 후 확인하는 것을 권장한다.

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
<div class="prob-meta"><span class="prob-num">기초 1</span><span class="prob-tag">MLE · 베르누이</span></div>
<div class="prob-q">동전을 10번 던져 앞면이 7번 나왔다. (1) MLE로 앞면 확률 $p$를 추정하라. (2) Beta(2, 2) 사전 분포를 사용한 MAP 추정값을 구하라. (3) 두 값이 다른 이유를 설명하고, 시행 횟수가 1000번으로 늘어나도 7:3 비율이 유지된다면 두 추정값이 어떻게 변하는지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>(1) MLE.</strong> 베르누이 우도 $L(p) = p^7(1-p)^3$을 최대화. 로그 우도 미분: $\frac{d}{dp}[7\log p + 3\log(1-p)] = \frac{7}{p} - \frac{3}{1-p} = 0$. 풀면 $\hat{p}_\text{MLE} = 7/10 = 0.7$. 직관적으로 MLE는 "관측 빈도 그 자체"다.</p>

<p><strong>(2) MAP.</strong> Beta(2,2) 사전 분포 $p(\theta) \propto \theta^{2-1}(1-\theta)^{2-1} = \theta(1-\theta)$. 사후 분포: $p(\theta|D) \propto \theta^{7+1}(1-\theta)^{3+1} = \theta^8(1-\theta)^4$ → Beta(9,5). MAP = 모드 = $(9-1)/(9+5-2) = 8/12 = 0.667$. 사전 분포가 0.5 방향으로 "당기는" 효과가 반영된다.</p>

<p><strong>(3) 수렴 분석.</strong> 시행이 늘어도 7:3 비율이면 700앞/300뒤. MAP = $(700+1)/(1000+2) = 701/1002 \approx 0.6996$. MLE = 0.7. 사전 분포의 기여(Beta(2,2)의 가중치 2+2=4)가 데이터(1000)에 비해 무시할 정도로 작아진다. 이것이 <strong>Bernstein-von Mises 정리</strong>의 직관이다 — 데이터가 많을수록 사전 분포의 영향이 희석되어 MAP → MLE로 수렴한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 2</span><span class="prob-tag">MLE · 가우시안 · MSE</span></div>
<div class="prob-q">선형 회귀 $y = \theta^T x + \epsilon$에서 잔차 $\epsilon \sim \mathcal{N}(0, \sigma^2)$로 가정할 때, MLE를 최대화하는 $\theta$가 MSE 손실을 최소화하는 $\theta$와 동일함을 유도하라. 이 결과가 "MSE를 쓰는 것은 암묵적으로 가우시안 노이즈를 가정한다"는 말의 의미를 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>유도.</strong> $\epsilon \sim \mathcal{N}(0,\sigma^2)$이면 $y|x,\theta \sim \mathcal{N}(\theta^T x, \sigma^2)$. $n$개 데이터의 로그 우도:</p>
<div class="prob-formula">$$\log L(\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \theta^T x_i)^2$$</div>
<p>$\theta$에 관해 최대화 = $\sum(y_i - \theta^T x_i)^2$ 최소화. 앞의 상수 항은 $\theta$와 무관하므로 $\hat\theta_\text{MLE} = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n (y_i - \theta^T x_i)^2 = \hat\theta_\text{MSE}$. ∎</p>

<p><strong>함의.</strong> MSE 손실을 아무 생각 없이 쓰는 것은 <strong>"관측값과 예측값의 차이가 정규 분포를 따른다"</strong>는 강한 가정을 암묵적으로 채택하는 것이다. 이 가정이 틀리면(예: 이상치가 많은 데이터) MSE는 최적이 아니다. 라플라스 분포 노이즈 가정 → MAE(L1 손실)가 대응되며, 이상치에 더 강건하다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 3</span><span class="prob-tag">MLE vs MAP · 소표본</span></div>
<div class="prob-q">의사가 희귀 질환을 진단하는 모델을 훈련한다. 훈련 데이터가 단 5명뿐이고 그 중 4명이 양성이다. MLE는 양성 확률을 0.8로 추정하지만, 실제로는 이 질환이 인구의 1%에게서 발생한다고 알려져 있다. 이 상황에서 MAP가 더 합리적인 이유를 설명하고, 적절한 사전 분포를 제안하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>MLE의 문제.</strong> 5개 데이터 기반 MLE = 4/5 = 0.8. 그러나 희귀 질환의 실제 유병률이 1%라는 사전 지식을 무시한 결과다. 소표본에서 MLE는 우연한 표본 편향에 극도로 취약하다 — 만약 진단 양성인 환자들만 데이터에 포함됐다면 더 극단적인 편향이 생긴다.</p>

<p><strong>MAP의 해결.</strong> 사전 지식 "유병률 ≈ 1%"를 Beta 사전 분포로 인코딩한다. Beta($\alpha, \beta$)의 평균이 1%가 되려면 $\alpha/(\alpha+\beta) = 0.01$. 강한 사전 지식을 반영하려면 큰 $\alpha+\beta$를 선택, 예를 들어 Beta(1, 99). 사후 분포: Beta(1+4, 99+1) = Beta(5, 100). MAP = $(5-1)/(5+100-2) = 4/103 \approx 0.039$. 여전히 MLE(0.8)보다 훨씬 낮고, 사전 지식에 부합한다.</p>

<p><strong>결론.</strong> 데이터가 적을수록 사전 분포의 정규화 효과가 크다. 이것이 <strong>의료, 법률, 안전 시스템처럼 소표본+강한 도메인 지식이 있는 분야에서 베이즈 방법론이 필수적</strong>인 이유다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 4</span><span class="prob-tag">사후 분포 · 불확실성</span></div>
<div class="prob-q">MAP 추정은 사후 분포의 "최빈값(mode)"만 반환한다. 가우시안 사후 분포에서 MAP와 사후 평균(posterior mean)이 같은 이유를 설명하라. 반면 비대칭 분포(예: Beta 사후 분포)에서 이 두 값이 왜 달라지는지 논하고, 점 추정만 쓰는 것의 위험성을 불확실성 관점에서 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>가우시안에서 mode = mean.</strong> 가우시안 분포 $\mathcal{N}(\mu, \sigma^2)$는 $\mu$에서 대칭이다. 대칭 분포에서 mode(최빈값), mean(평균), median(중앙값)이 모두 같다. 따라서 가우시안 사후 분포에서 MAP = 사후 평균 = $\mu$.</p>

<p><strong>Beta 사후에서의 차이.</strong> Beta($\alpha, \beta$)의 mode = $(\alpha-1)/(\alpha+\beta-2)$ (단, $\alpha,\beta > 1$), mean = $\alpha/(\alpha+\beta)$. 예: Beta(2, 5): mode = 1/5 = 0.2, mean = 2/7 ≈ 0.286. 분포가 오른쪽으로 치우쳐(positively skewed) 있으면 mean > mode. 이처럼 비대칭 분포에서는 MAP가 사후 분포의 "중심"을 대표하지 못한다.</p>

<p><strong>점 추정의 위험성.</strong> MAP는 사후 분포에서 가장 그럴듯한 파라미터 값 하나만 반환한다. 그러나 <strong>불확실성 정보가 사라진다</strong>: 사후 분포가 날카로운지(확신 높음) 퍼져있는지(불확실) 알 수 없다. 의사 결정에서 이것은 중요하다 — "예측값이 0.7"보다 "예측값이 0.7, 95% 신뢰 구간은 [0.4, 0.9]"가 훨씬 더 많은 정보를 담는다. 완전 베이즈(full Bayesian) 접근은 사후 분포 전체를 유지함으로써 이 불확실성을 다운스트림 의사결정에 전파한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">기초 5</span><span class="prob-tag">교차 엔트로피 · 소프트맥스 · MLE</span></div>
<div class="prob-q">3-클래스 분류 문제에서 정답이 클래스 2일 때 (원핫: [0, 1, 0]), 모델이 [0.1, 0.7, 0.2]를 출력하는 경우와 [0.01, 0.98, 0.01]을 출력하는 경우의 교차 엔트로피 손실을 각각 계산하라. 교차 엔트로피가 카테고리 분포의 음의 로그 우도와 동치임을 보이고, 모델이 확신할수록 손실이 줄어드는 이유를 설명하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>교차 엔트로피 계산.</strong> $\mathcal{L} = -\sum_k y_k \log \hat{y}_k$. 정답 클래스만 $y_k = 1$이므로 $\mathcal{L} = -\log \hat{y}_\text{정답}$.</p>
<div class="prob-formula">$$\mathcal{L}_1 = -\log(0.7) \approx 0.357 \qquad \mathcal{L}_2 = -\log(0.98) \approx 0.020$$</div>
<p>정답 클래스에 더 높은 확률을 부여할수록 손실이 줄어든다.</p>

<p><strong>카테고리 분포 MLE와의 동치.</strong> 카테고리 분포의 우도: $p(y|\hat{y}) = \prod_k \hat{y}_k^{y_k}$. 음의 로그 우도:</p>
<div class="prob-formula">$$-\log p(y|\hat{y}) = -\sum_k y_k \log \hat{y}_k = H(y, \hat{y})$$</div>
<p>이것이 정확히 교차 엔트로피다. 따라서 <strong>분류 모델에서 교차 엔트로피 손실을 최소화하는 것 = 카테고리 분포 MLE</strong>.</p>

<p><strong>확신할수록 손실 감소의 이유.</strong> 로그 함수의 단조성 때문이다. $-\log(\hat{y})$는 $\hat{y} \to 1$일수록 0으로 수렴, $\hat{y} \to 0$일수록 $+\infty$로 발산. 모델이 정답 클래스에 확률 1을 부여하면 손실 = 0, 오답에 확률 1을 부여하면 손실 = $\infty$. 이 비대칭적 페널티가 모델을 정답에 대한 확신을 키우는 방향으로 학습시킨다.</p>

</div>
</div>

<div class="prob-lv">대학원 심화 문제 — 엄밀한 논증</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 1</span><span class="prob-tag">Fisher Information · Cramér-Rao</span></div>
<div class="prob-q">Fisher Information $\mathcal{I}(\theta) = -\mathbb{E}\!\left[\frac{\partial^2}{\partial\theta^2}\log p(x|\theta)\right]$을 정의하고, Cramér-Rao 하한이 MLE의 점근적 분산과 어떻게 연결되는지 설명하라. MLE가 <em>점근적으로 효율적인(asymptotically efficient)</em> 추정량이라는 것의 의미를 논하고, 이것이 대규모 데이터 환경에서 MLE가 선호되는 통계적 근거가 되는 이유를 서술하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>Fisher Information의 정의와 의미.</strong> Fisher Information $\mathcal{I}(\theta)$는 관측값 $x$가 파라미터 $\theta$에 대해 담고 있는 정보의 양을 측정한다. 동치 표현으로 $\mathcal{I}(\theta) = \mathbb{E}\!\left[(\partial_\theta \log p(x|\theta))^2\right]$도 성립한다 — score function의 분산이다. $\mathcal{I}(\theta)$가 크다는 것은 score가 $\theta$에 민감하게 변동한다는 뜻, 즉 데이터로부터 $\theta$를 정밀하게 추정할 수 있음을 의미한다.</p>

<p><strong>Cramér-Rao 하한(CRLB).</strong> 불편추정량 $\hat\theta$의 분산은 Fisher Information의 역수보다 작을 수 없다:</p>
<div class="prob-formula">$$\text{Var}(\hat\theta) \;\geq\; \frac{1}{\mathcal{I}(\theta)}$$</div>
<p>증명의 핵심은 Cauchy-Schwarz 부등식이다. $\mathbb{E}[\hat\theta] = \theta$이므로 양변을 $\theta$로 미분하면 $\mathbb{E}[(\hat\theta - \theta) \cdot \partial_\theta \log p] = 1$이 되고, Cauchy-Schwarz를 적용하면 $\text{Var}(\hat\theta) \cdot \mathcal{I}(\theta) \geq 1$을 얻는다. CRLB를 달성하는 추정량을 <span class="kw">효율적(efficient)</span>이라 한다.</p>

<p><strong>MLE의 점근적 효율성.</strong> $n$개의 i.i.d. 샘플에 대한 MLE $\hat\theta_\text{MLE}$는 $n \to \infty$ 극한에서 다음 정규분포로 수렴한다:</p>
<div class="prob-formula">$$\sqrt{n}(\hat\theta_\text{MLE} - \theta^*) \;\xrightarrow{d}\; \mathcal{N}\!\left(0,\; \mathcal{I}(\theta^*)^{-1}\right)$$</div>
<p>즉 MLE의 점근적 분산이 정확히 CRLB를 달성한다 — <span class="kw2">어떠한 불편추정량도 점근적으로 MLE보다 낮은 분산을 가질 수 없다</span>. 이것이 점근적 효율성(asymptotic efficiency)의 의미다.</p>

<p><strong>대규모 데이터 환경에서의 함의.</strong> 딥러닝처럼 $n$이 매우 큰 환경에서는 MLE가 가장 빠른 수렴 속도($1/\sqrt{n}$)와 최소 분산을 동시에 달성한다. 반면 사전 분포 기반의 MAP는 $n$이 증가함에 따라 사전 분포의 영향이 희석되어 결국 MLE로 수렴한다(Bernstein-von Mises 정리). 따라서 대규모 데이터에서 MLE는 "prior 선택의 부담 없이 통계적으로 최적인" 추정을 제공한다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 2</span><span class="prob-tag">L1 · L2 정규화 · 기하학</span></div>
<div class="prob-q">MAP 추정에서 L2 정규화(Ridge)는 Gaussian 사전 분포, L1 정규화(Lasso)는 Laplace 사전 분포에 대응된다. 등고선(contour)과 제약 집합의 <em>기하학적</em> 관점에서 L1이 sparse한 해를 유도하고 L2는 그렇지 않은 이유를 설명하라. Elastic Net이 두 정규화를 결합하는 동기와 실전적 의미도 서술하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>제약 집합의 기하학.</strong> 정규화 파라미터 $\lambda$를 라그랑주 승수로 보면, L2 정규화는 $\|\theta\|_2^2 \leq t$인 구(球) 안에서 손실을 최소화하는 제약 최적화와 동치이고, L1 정규화는 $\|\theta\|_1 \leq t$인 <span class="kw">다이아몬드(菱形)</span> 안에서 최소화하는 것과 동치다.</p>

<p><strong>왜 L1이 sparse한 해를 유도하는가.</strong> 손실 함수의 등고선(타원)이 제약 집합과 처음 만나는 점이 최적해다. L2 제약 집합인 구는 표면이 매끄러워 등고선과 대부분 매끄러운 곡선 위에서 접한다 — 이 접점이 축 위에 올 이유가 없다. 반면 L1 다이아몬드는 <span class="kw">꼭짓점(corner)</span>이 좌표축 위에 있고 면이 넓다. 등고선이 어떤 방향에서 오든 꼭짓점 근처에서 먼저 만날 확률이 높고, 꼭짓점은 정확히 일부 파라미터가 0인 점이다. 따라서 L1은 구조적으로 sparse 해를 선호한다.</p>

<p><strong>사전 분포 관점에서의 확인.</strong> Laplace 분포 $p(\theta) \propto e^{-\lambda|\theta|}$의 로그를 취하면 $-\log p(\theta) \propto \lambda|\theta|$. Gaussian $p(\theta) \propto e^{-\lambda\theta^2/2}$의 로그는 $-\log p(\theta) \propto \lambda\theta^2/2$. Laplace 분포는 $\theta=0$에서 <span class="kw">뾰족한 첨두(sharp peak)</span>를 가지므로 사전 분포 자체가 정확히 0인 값을 강하게 선호한다. Gaussian은 $\theta=0$ 근방에서 완만하게 집중되어 있어 정확히 0을 유도하지 않는다.</p>

<p><strong>Elastic Net의 동기.</strong> L1의 sparsity 유도 능력과 L2의 안정성(서로 상관된 변수 그룹을 동시에 선택하는 경향)을 결합한 것이 Elastic Net이다: $\lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2$. L1만 쓰면 상관된 변수들 중 하나만 임의로 선택하는 경향이 있고 고차원에서 불안정하다. Elastic Net은 <span class="kw2">그룹 선택(group selection)</span>과 sparsity를 동시에 달성하며, 사전 분포 관점으로는 Laplace-Gaussian 혼합 사전에 대응된다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 3</span><span class="prob-tag">Empirical Bayes · 주변 우도</span></div>
<div class="prob-q">순수 베이즈(Full Bayes) 관점에서 사전 분포의 하이퍼파라미터 $\alpha$는 그 자체로도 사전 분포(hyperprior)를 가져야 한다. Empirical Bayes는 대신 $\alpha$를 주변 우도(marginal likelihood) $p(D|\alpha) = \int p(D|\theta)\,p(\theta|\alpha)\,d\theta$를 최대화하여 추정한다. 이것이 왜 "빈도주의적 베이즈"라고 불리는지, 과적합 위험이 있음에도 실전에서 왜 효과적인지 논하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>"빈도주의적 베이즈"인 이유.</strong> Full Bayesian 접근에서는 모든 미지수($\theta$와 $\alpha$ 모두)에 대해 사전 분포를 부여하고 완전한 사후 분포를 계산한다. 그러나 Empirical Bayes는 $\alpha$를 <span class="kw">데이터의 함수로 추정된 점 추정값</span>으로 처리한다. 이것은 $\alpha$에 대해 사전 분포 대신 데이터로부터 직접 추정하는 빈도주의적 방식을 사용하되, $\theta$에 대해서는 베이즈 추론(사후 분포 계산)을 유지하기 때문에 두 패러다임의 혼합이다. Efron이 "empirical Bayes"라는 용어를 대중화했으며, Good-Turing smoothing이 대표적 사례다.</p>

<p><strong>과적합 위험.</strong> $\alpha$를 같은 데이터 $D$로부터 추정하므로 사전 분포가 데이터에 "오염"된다. 이는 순수 베이즈 관점에서 이중 사용(double-dipping) 문제이며, 데이터 수가 적으면 $\alpha$가 과도하게 데이터에 맞춰져 사전 분포의 정규화 효과가 사라질 수 있다. 공식적으로는 $p(D|\alpha)$의 최대화가 marginal likelihood에 대한 MLE이므로, 복잡한 모델에서는 overfitting이 발생할 수 있다.</p>

<p><strong>그럼에도 효과적인 이유.</strong> 첫째, 주변 우도 $p(D|\alpha) = \int p(D|\theta)p(\theta|\alpha)d\theta$는 $\theta$에 대한 적분이므로 이미 <span class="kw2">파라미터 공간 전체에 대한 평균</span>을 반영한다 — $\alpha$에 대한 과적합이 $\theta$ 수준에서는 자동 정규화되는 효과가 있다. 둘째, hyperprior를 정하는 것 자체가 어렵고 주관적인 반면, 주변 우도 최대화는 데이터에 의해 객관적으로 $\alpha$를 결정한다. 셋째, 가우시안 프로세스, 자동 연관성 결정(ARD) 등 실용적 모델에서 Empirical Bayes는 계산 가능하면서도 좋은 일반화 성능을 보인다. 데이터가 충분히 많으면 이중 사용 문제의 영향이 희석된다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 4</span><span class="prob-tag">EM 알고리즘 · Jensen 부등식</span></div>
<div class="prob-q">잠재 변수(latent variable) $z$가 있는 모델에서 관측 로그 우도 $\log p(x|\theta) = \log \int p(x,z|\theta)\,dz$를 직접 최대화하기 어렵다. EM 알고리즘이 E-step과 M-step을 통해 이 우도를 단조 증가(monotonically non-decreasing)시킴을 증명하라. Jensen 부등식이 어떻게 핵심적 역할을 하는지 명확히 포함하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>ELBO 도출.</strong> 임의의 분포 $q(z)$를 도입하면:</p>
<div class="prob-formula">$$\log p(x|\theta) = \log \int \frac{p(x,z|\theta)}{q(z)} q(z)\,dz = \log\,\mathbb{E}_{q}\!\left[\frac{p(x,z|\theta)}{q(z)}\right]$$</div>
<p>로그 함수의 <span class="kw">오목성(concavity)</span>에 Jensen 부등식을 적용하면:</p>
<div class="prob-formula">$$\log p(x|\theta) \;\geq\; \mathbb{E}_{q}\!\left[\log\frac{p(x,z|\theta)}{q(z)}\right] = \mathbb{E}_q[\log p(x,z|\theta)] + H(q) \;\equiv\; \mathcal{L}(q,\theta)$$</div>
<p>$\mathcal{L}(q,\theta)$를 <span class="kw">ELBO(Evidence Lower BOund)</span>라 한다. 등호는 $q(z) = p(z|x,\theta)$일 때 성립한다.</p>

<p><strong>E-step.</strong> $\theta = \theta^{(t)}$를 고정하고 $q$에 대해 ELBO를 최대화한다. ELBO = $\log p(x|\theta^{(t)}) - D_{KL}(q \| p(z|x,\theta^{(t)}))$이므로, KL을 0으로 만드는 $q^{(t+1)}(z) = p(z|x,\theta^{(t)})$가 최적이다. 이때 ELBO = $\log p(x|\theta^{(t)})$, 즉 하한이 현재 우도와 일치한다.</p>

<p><strong>M-step.</strong> $q = q^{(t+1)}$을 고정하고 $\theta$에 대해 ELBO를 최대화한다:</p>
<div class="prob-formula">$$\theta^{(t+1)} = \arg\max_\theta\; \mathbb{E}_{q^{(t+1)}}[\log p(x,z|\theta)]$$</div>
<p>이것은 완전 데이터 로그 우도의 기댓값(Q-function)을 최대화하는 것으로, 보통 닫힌 해가 존재한다.</p>

<p><strong>단조 증가 증명.</strong> E-step 후 ELBO $= \log p(x|\theta^{(t)})$. M-step은 ELBO를 증가시키므로 $\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(x|\theta^{(t)})$. 그런데 모든 $q, \theta$에 대해 ELBO $\leq \log p(x|\theta)$이고, 다음 E-step에서 다시 하한이 $\log p(x|\theta^{(t+1)})$로 올라간다. 따라서 <span class="kw2">$\log p(x|\theta^{(t+1)}) \geq \log p(x|\theta^{(t)})$</span>가 보장된다. EM은 수렴하지만 지역 최솟값에 갇힐 수 있다는 한계가 있다.</p>

</div>
</div>

<div class="prob-block">
<div class="prob-meta"><span class="prob-num">문제 5</span><span class="prob-tag">베이즈 모델 선택 · Occam's Razor · BIC</span></div>
<div class="prob-q">두 모델 $M_1, M_2$ 중 더 나은 모델을 선택하는 베이즈 모델 비교(Bayes Factor)에서, <em>더 복잡한 모델이 자동으로 패널티를 받는</em> 이유를 주변 우도 $p(D|M)$의 관점에서 설명하라. 이것이 Occam's Razor의 수학적 구현이라 불리는 이유는? BIC(Bayesian Information Criterion)가 주변 우도를 어떻게 근사하는지도 서술하라.</div>
<button class="prob-toggle" onclick="tp(this)">모범 답안 보기 ▾</button>
<div class="prob-ans">

<p><strong>주변 우도의 확률론적 구조.</strong> 모델 $M$의 주변 우도는 $p(D|M) = \int p(D|\theta, M)\,p(\theta|M)\,d\theta$. 이것은 파라미터 공간 전체에 대한 우도의 사전 분포 가중 평균이다. 복잡한 모델은 파라미터 공간이 넓어 사전 분포가 퍼져있고, 최대 우도가 높아도 <span class="kw">적분 전체는 희석(diluted)</span>된다. 반면 단순한 모델은 설명 가능한 데이터 집합이 좁지만, 그 범위 안에 실제 데이터가 들어오면 높은 평균 우도를 달성한다.</p>

<p><strong>Occam's Razor 자동 구현.</strong> 복잡한 모델 $M_2$가 더 많은 데이터 패턴을 설명할 수 있다고 해도, $p(D|M_2) = \int p(D|\theta)p(\theta|M_2)d\theta$를 계산하면 파라미터 공간이 넓어 각 $\theta$에 배당된 확률 질량이 작다. 마치 $M_2$는 "모든 경우를 다 조금씩 예측"하고, $M_1$은 "좁은 범위를 강하게 예측"하는 것과 같다. 데이터가 $M_1$의 예측 범위 안에 있다면 $p(D|M_1) > p(D|M_2)$가 될 수 있다. 이것이 <span class="kw">베이즈적 Occam's Razor</span>: 더 단순한 모델이 설명력이 충분하다면 자동으로 선호된다.</p>

<p><strong>Bayes Factor.</strong> $K = p(D|M_1)/p(D|M_2)$. $K > 1$이면 $M_1$ 선호. 명시적 모델 복잡도 패널티항이 없어도 복잡도가 자동 반영된다는 것이 핵심이다.</p>

<p><strong>BIC 근사.</strong> Laplace 근사를 주변 우도에 적용하면:</p>
<div class="prob-formula">$$\log p(D|M) \approx \log p(D|\hat\theta_\text{MLE}) - \frac{k}{2}\log n + \text{const}$$</div>
<p>여기서 $k$는 파라미터 수, $n$은 데이터 수. <span class="kw2">BIC = $-2\log p(D|\hat\theta) + k\log n$</span>은 이를 최소화 문제로 바꾼 것으로, 파라미터 수에 $\log n$ 비율의 패널티를 부과한다. AIC($k$ 패널티)보다 $n$이 클수록 복잡도에 더 강한 패널티를 주므로, 대규모 데이터에서 더 보수적인 모델을 선택하는 경향이 있다.</p>

</div>
</div>

<div class="footnote">
  이전 포스트: <a href="mle-map.html">MLE & MAP 추정</a> · 관련: <a href="information-theory.html">정보이론</a> · <a href="information-theory-problems.html">정보이론 심화 문제</a>
</div>
