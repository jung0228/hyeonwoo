# MLE / MAP / Bayesian Inference

카테고리: Math & Stats  
자신감: ⭐⭐⭐ (중급)  
마지막 복습: 2026-08-11


## 한 문장 요약

MLE는 데이터만으로 최적 parameter를 추정하고, MAP는 prior도 반영하며, Bayesian은 uncertainty까지 분포로 표현한다.


## MLE (Maximum Likelihood Estimation)

$$\hat{\theta}_{MLE} = \arg\max_\theta p(D|\theta) = \arg\max_\theta \sum_i \log p(x_i|\theta)$$

- Gaussian noise 가정 → log-likelihood 최대화 = MSE 최소화 (OLS)
- Categorical 분포 → log-likelihood 최대화 = Cross-Entropy 최소화

### Consistency

실제 모수 $\theta_0$일 때, 표본 수 증가 → 추정량이 $\theta_0$으로 확률수렴:

$$\hat{\theta}_n \xrightarrow{p} \theta_0$$

> MLE가 항상 consistent하지는 않음 — 식별 가능성(identifiability) + 정규성 조건 필요


## MAP (Maximum A Posteriori)

$$\hat{\theta}_{MAP} = \arg\max_\theta \underbrace{p(D|\theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}$$

로그를 취하면:

$$\hat{\theta}_{MAP} = \arg\max_\theta \left[\log p(D|\theta) + \log p(\theta)\right]$$

→ prior의 log = regularization 항처럼 작용

| Prior | Regularization |
|---|---|
| Gaussian prior $\mathcal{N}(0, \sigma^2)$ | $L_2$ (Ridge) |
| Laplace prior | $L_1$ (Lasso) |


## Bayesian Inference

$$p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$$

- Prior $p(\theta)$: 데이터 전 parameter 믿음
- Likelihood $p(D|\theta)$: parameter 주어졌을 때 data 가능성
- Posterior $p(\theta|D)$: data 반영 후 parameter 분포
- Evidence $p(D) = \int p(D|\theta)p(\theta)d\theta$: 정규화 상수

### Sequential update

$$p(\theta|D_1, D_2) \propto p(D_2|\theta) \cdot p(\theta|D_1)$$

→ 이전 posterior를 다음 prior로 사용 가능

### MLE vs MAP vs Bayesian

| | MLE | MAP | Bayesian |
|---|---|---|---|
| Prior | 무시 | 반영 | 반영 |
| 출력 | Point estimate | Point estimate | 분포 전체 |
| Uncertainty | 표현 불가 | 표현 불가 | 표현 |
| 데이터 적을 때 | 불안정 | Prior로 안정 | 안정 |


## Cross-Entropy와의 연결

Categorical model에서 MLE = Cross-Entropy 최소화:

$$\text{CE}(q, p) = -\sum_k q_k \log p_k = H(q) + D_{KL}(q \| p)$$

$q$가 고정이면 CE 최소화 = KL Divergence 최소화


## 체크리스트

- [x] MLE 수식 및 직관 설명
- [x] MAP = MLE + regularization 연결
- [x] Gaussian prior → L2 regularization 유도
- [x] Bayesian posterior update 과정
- [x] Cross-entropy와 MLE 연결
- [x] Consistency 정의
- [ ] EM 알고리즘과 MLE 연결 (hidden variable)
- [ ] Variational Bayes 개요
