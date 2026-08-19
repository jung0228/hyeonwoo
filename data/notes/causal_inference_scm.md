# 구조적 인과 모델 & $do$-연산 (Structural Causal Models & $do$-Calculus)

## 핵심 아이디어
단순히 데이터의 통계적 상관관계(Correlation, $P(Y \mid X)$)를 관측하는 수준을 넘어, 특정 변수에 인위적인 개입(Intervention, $do(X=x)$)을 가했을 때 다른 변수들에 미치는 인과적 영향과 "만약 과거에 다른 선택을 했다면?(Counterfactuals)"이라는 반사실적 가상 시나리오를 수학적으로 추론하는 주디아 펄(Judea Pearl)의 인과 추론 체계입니다.

---

## 핵심 수식 및 3단계 인과 사다리 (Ladder of Causation)

```
[ 계층 3: 반사실 (Counterfactuals) ] ──▶ P(Y_x | x', y') : "만약 치료를 받지 않았다면 사망했을까?"
[ 계층 2: 개입 (Intervention) ]       ──▶ P(Y | do(X=x))   : "약을 강제로 투여하면 환자가 살까?"
[ 계층 1: 연관 (Association) ]        ──▶ P(Y | X=x)       : "약을 먹은 사람 중 생존율은 얼마인가?"
```

### 1. $do$-연산자 & 백도어 조정 공식 (Backdoor Adjustment Formula)
혼란 변수(Confounder) 집합 $\mathbf{Z}$가 백도어 기준을 만족할 때, 능동적 개입 확률:

$$P(Y \mid do(X=x)) = \sum_{\mathbf{z}} P(Y \mid X=x, \mathbf{Z}=\mathbf{z}) P(\mathbf{Z}=\mathbf{z})$$

### 2. 구조적 인과 방정식 (Structural Causal Equations)
각 변수 $V_i$는 부모 노드 $PA_i$와 외생적 노이즈 $U_i$의 결정론적 함수로 정의됨:

$$V_i := f_i(PA_i, U_i), \quad i = 1, \dots, N$$

---

## 비디오 VLM & 시공간 세그멘테이션(VIRST) 적용 원리
- **왜 필요한가**: 현재의 비디오 모델은 비디오 프레임들을 시간 순서대로 단순히 이어붙인 상관관계 텐서로 학습하기 때문에, "운전자가 2초 전에 핸들을 꺾었다면?"이라는 질문에 시간 인과 방향을 뒤죽박죽 섞어버리는 치명적 환각을 생성함.
- **해결책**: 비디오 내 객체의 시간 궤적을 인과 방향성 비순환 그래프(Causal DAG)로 모델링하고, 멀티모달 RoPE 어텐션 가중치에 $do(X=x)$ 연산 마스크를 주입하여 **반사실적 분기 시공간 픽셀 마스크(Counterfactual Spatiotemporal Mask)**를 오차 없이 추적.

---

## 연결 개념 및 논문
- [[paper_virst]] : VIRST의 3D 시공간 세그멘테이션을 인과 추론으로 승격
- [[paper_mevis]] : 모션 기반 비디오 분할에서의 교란 변수 제거
- [[rq_counterfactual_video_causality]] : 반사실적 비디오 인과 추론 핵심 연구 과제
