# 해밀토니안 역학 & 물리 보존 법칙 (Hamiltonian Mechanics & Energy Conservation)

## 핵심 아이디어
고전역학에서 시스템의 총 에너지(운동 에너지 $T$ + 위치 에너지 $V$)를 나타내는 해밀토니안 함수 $H(\mathbf{q}, \mathbf{p})$를 통해, 상태 공간(위상 공간, Phase Space)에서 에너지 보존 법칙을 만족하는 시간 발전 궤적을 미분 방정식으로 모델링하는 수학적 프레임워크입니다. AI 월드 모델에 주입하여 픽셀의 비물리적 환각(물체 관통, 비정상적 에너지 증폭)을 원천 차단합니다.

---

## 핵심 수식

### 1. 해밀턴 정준 운동 방정식 (Hamilton's Canonical Equations)
일반화 좌표 $\mathbf{q}$와 일반화 운동량 $\mathbf{p}$, 그리고 총 에너지 해밀토니안 $H(\mathbf{q}, \mathbf{p})$에 대해:

$$\frac{d\mathbf{q}}{dt} = \frac{\partial H}{\partial \mathbf{p}}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{q}}$$

### 2. 에너지 보존 법칙 (Conservation of Energy)
시간에 독립적인 고립계($\frac{\partial H}{\partial t} = 0$)에서:

$$\frac{dH}{dt} = \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{d\mathbf{q}}{dt} + \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{d\mathbf{p}}{dt} = \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{\partial H}{\partial \mathbf{p}} - \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{\partial H}{\partial \mathbf{q}} = 0$$

### 3. Hamiltonian Neural Networks (HNN) Loss
신경망 $H_\theta(\mathbf{q}, \mathbf{p})$가 예측한 편미분값과 실제 관측된 시간 미분값 $(\dot{\mathbf{q}}, \dot{\mathbf{p}})$ 사이의 잔차 최소화:

$$\mathcal{L}_{\text{HNN}} = \left\| \frac{\partial H_\theta}{\partial \mathbf{p}} - \dot{\mathbf{q}} \right\|_2^2 + \left\| -\frac{\partial H_\theta}{\partial \mathbf{q}} - \dot{\mathbf{p}} \right\|_2^2$$

---

## 옴니모달 피지컬 AI / 로보틱스 적용 원리
- **왜 필요한가**: 2D 비디오 생성 모델은 "물체가 어떻게 생겼는가"만 모사하여 무거운 쇠공을 들었을 때와 풍선을 들었을 때를 시각적으로 구별하지 못함.
- **해결책**: 비디오 잠재 공간(Latent Space)의 좌표를 위상 공간 $(\mathbf{q}, \mathbf{p})$으로 바인딩하고, $\mathcal{L}_{\text{HNN}}$을 손실 함수로 부과하여 로봇이 실제 물리적 관성과 질량을 온전히 반영하여 50Hz 모터 토크를 출력하도록 제약.

---

## 연결 개념 및 논문
- [[paper_cosmos]] : 물리 법칙이 위배되는 2D 비디오 월드 모델의 한계 극복
- [[paper_flow_matching]] : 위상 공간에서의 최적 수송 직선 궤적 학습
- [[rq_physical_world_model]] : Dynin-Robotics를 위한 Physics-Grounded 4D World Model 핵심 엔진
