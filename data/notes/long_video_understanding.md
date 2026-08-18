# Long Video & Temporal Grounding

## 핵심 아이디어
수십 분에서 수 시간 분량의 장시간 비디오에서 단순한 프레임별 정적 인식을 넘어, 시간의 흐름에 따른 인과관계(Causality), 다중 모달(시각+오디오+자막), 그리고 특정 사건이 발생한 정밀한 시간 구간(Temporal Moment Retrieval)을 정확히 찾아내고 추론하는 멀티모달 기술입니다.

---

## 핵심 수식 및 문제 정의

### 1. Long Video Moment Retrieval (LVMR)
길이 $T$의 비디오 $V = \{f_1, f_2, \dots, f_T\}$와 자연어 쿼리 $Q$가 주어졌을 때, 쿼리에 부합하는 시간 구간 $[t_{\text{start}}, t_{\text{end}}]$을 예측:
$$[\hat{t}_{\text{start}}, \hat{t}_{\text{end}}] = \arg\max_{[s, e]} P([s, e] \mid V, Q)$$

### 2. Temporal IoU 손실 함수
예측 구간과 정답 구간 간의 정렬 오차 최소화:
$$\mathcal{L}_{\text{temporal}} = 1 - \text{tIoU}(\hat{I}, I_{\text{gt}}) + \lambda \mathcal{L}_{\text{smooth-L1}}(\hat{I}, I_{\text{gt}})$$

### 3. Multi-modal Token Fusion
$$\mathbf{Z} = \text{Transformer}(\mathbf{Z}_{\text{visual}} \oplus \mathbf{Z}_{\text{audio}} \oplus \mathbf{Z}_{\text{text}})$$

---

## 직관적 설명
2시간짜리 축구 경기 영상에서 "후반전 30분경 손흥민의 감아차기 골 장면"을 단 몇 초 만에 정확한 타임스탬프([01:15:20 ~ 01:15:45])로 짚어내고, 그 골이 어떤 패스 플레이로부터 시작되었는지 인과관계를 설명하는 능력입니다.

---

## 연결 개념
- [[multimodal]] : 시각, 음향, 텍스트의 상호 모달 정렬
- [[vision_encoder]] : 대규모 비디오 프레임 임베딩 추출
- [[streamkv]] : 초장거리 비디오 토큰을 효율적으로 처리하기 위한 KV Cache 경량화 기법

---

## 참고
- LongVALE: Towards Time-Aware Omni-Modal Perception of Long Videos (CVPR 2025)
- Momentseeker: A Benchmark for Long-Video Moment Retrieval (CVPR 2025)
- TCVP: Time-Centric Video Perception and Grounding (ArXiv 2026)
