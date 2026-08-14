# 📊 MLE (Maximum Likelihood) vs MAP (Maximum A Posteriori)

## [1단계] 명확한 개념 정의
- **MLE (최대유도추정)**: 관측 데이터 $D$의 우도 $P(D|\theta)$를 최대화하는 파라미터 $\theta$ 추정 방식.
  $$\hat{\theta}_{MLE} = \arg\max_{\theta} P(D | \theta)$$
- **MAP (최대사후확률추정)**: 데이터 $D$와 사전 지식 $P(\theta)$를 결합하여 사후 확률 $P(\theta|D)$를 최대화하는 추정 방식.
  $$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta | D) = \arg\max_{\theta} [P(D | \theta) \cdot P(\theta)]$$

## [2단계] 왜 쓰는가?
- **MLE**: 사전 정보가 없을 때 순수 데이터에 기반한 최적해 추정.
- **MAP**: 데이터 개수가 적을 때 발생할 수 있는 과적합(Overfitting)을 Prior 사전 경험으로 억제하여 모델을 안정화하기 위함.

## [3단계] 상황별 차이점 & 직관 (Trade-off)
- **데이터 부족 ($N \ll \infty$)**: MLE는 3번의 동전 던지기(모두 앞면)로 "앞면 확률 100%"라 단정짓지만, MAP는 $P(\theta)$ Prior가 0.5 근처로 확률을 억제하여 오버피팅을 방지함 (MAP 압승).
- **데이터 풍부 ($N \rightarrow \infty$)**: $P(D|\theta)$ 우도가 Prior를 무력화하여 MAP가 MLE 결과로 자연스럽게 수렴함.

## [4단계] 실전 AI / 딥러닝과의 연결고리
- **딥러닝 Weight Decay (L2 Regularization) = MAP의 가우시안 Prior!**
  - $\theta \sim \mathcal{N}(0, \sigma^2)$ 가우시안 Prior를 적용하고 $-\log$를 취하면:
    $$\min \left[ -\log P(D | \theta) + \frac{1}{2\sigma^2} \|\theta\|^2 \right]$$
  - 딥러닝에서 사용하는 Weight Decay는 파라미터가 0 근처일 것이라는 Prior를 주입한 MAP 추정과 수식적으로 100% 일치함.

## 연결 개념
- Cross-Entropy: Negative Log-Likelihood (MLE) 기반 손실함수
- L2 Regularization: MAP의 Gaussian Prior 주입 효과
