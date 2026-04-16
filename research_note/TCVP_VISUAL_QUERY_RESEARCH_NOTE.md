# TCVP Visual Query 방향 리서치 노트

> 업데이트: 2026-03-26

---

## 1. 핵심 태스크 정의

### MomentSeeker IMR (Image-conditioned Moment Retrieval)
- **Input**: q_T (텍스트) + q_I (이미지) + 영상 V
- **Output**: [t_start, t_end]
- **현재 최고 성능**: Gemini-2.5-Pro 29.6 R@1
- **LanguageBind baseline**: 18.2 R@1 (우리 목표: 이걸 넘기)
- **핵심 발견**: "No model was specifically designed or fine-tuned for IMR"
- **논문**: arXiv 2502.12558

### Ego4D VQ2D (Visual Query 2D Localization)
- **Input**: 이미지 crop (시각적 쿼리) + 영상
- **Output**: 해당 객체가 나오는 시간 + 위치 (bounding box)
- MomentSeeker IMR과 거의 동일한 태스크, 차이는 텍스트 쿼리 없음
- VQ2D는 연구 많이 됨 → 이미지+텍스트 IMR은 아직 공백

---

## 2. 관련 논문 (검색 완료: 2026-03-26)

### Ego4D VQ2D / Visual Query Localization 계열

| 논문 | arxiv | 핵심 방법 | 관련성 |
|------|-------|-----------|--------|
| **RELOCATE** (CVPR 2025) | 2412.01826 | SAM2로 객체 추적 후 "reiteration": 다른 시점 crop 자동 추출 → visual query 재사용. **우리 파이프라인과 구조적으로 동일** | ★★★ |
| **ESOM** | 2411.16934 | GroundingDINO + EgoSTARK tracker + object memory (ID별 시공간 좌표 저장). OMP 파이프라인이 자동 (image_query, segment) 생성과 유사 | ★★★ |
| **REN** (NeurIPS 2025) | 2505.18153 | DINOv2 + SigLIP2 조합으로 visual query localization SOTA | ★★★ 어드바이저 추천 논문 후보 |
| **HERO-VQL** | 2509.00385 | QueryAug: 같은 객체의 다른 annotated instance로 query 교체 (temporal variation 문제 인식) | ★★ |
| **VQ-SAM** | 2603.08898 | SAM2 + YouTube 영상으로 visual query segmentation. VQS-4K 데이터셋 (수동 구축). 우리와 동일 방향 | ★★★ |

### SAM2 Long Video 추적 계열

| 논문 | arxiv | 핵심 | 우리 적용 |
|------|-------|------|-----------|
| **SAMURAI** | 2411.11922 | Kalman filter로 재등장 처리, 거의 real-time. **실용적 선택** | ★★★ 추천 |
| **SAM2Long** | 2410.16268 | memory tree로 장기 추적, +5.3 J&F. 느리지만 고품질 | ★★ |
| **DAM4SAM** | 2411.17576 | distractor-aware memory. 유사 객체 많을 때 최적 | ★★ |

### IMR / Moment Retrieval 계열

| 논문 | arxiv | 핵심 | 관련성 |
|------|------|------|--------|
| **MomentSeeker** | 2502.12558 | IMR/TMR/VMR 벤치마크, 우리 평가 기준 | ★★★ |
| **LanguageBind** | — | text/video/image/audio 동일 embedding space, fine-tuning 대상 | ★★★ |
| **CoVR-2** | — | Reference image + text modifier → video retrieval | ★★ |
| **Vid-Morp** | 2412.00811 | GPT-4o로 50K 비디오 pseudo-annotation 자동 생성 (text만) | ★★ |

---

## 3. TCVP Visual Query 파이프라인 설계

### 핵심 아이디어
YouTube 타임스탬프 댓글 → 자동으로 (q_T, q_I, video_segment) IMR 트리플 생성

### 데이터 구성
```
전체 1000개 영상 → 텍스트 쿼리 (기존)
Visual query subset (~200개 영상) → IMR 트리플 추가
  대상 카테고리: Entertainment, News, Cooking, Making
```

### 난이도 스펙트럼 (논문 contribution)
| 케이스 | 예시 | 난이도 |
|--------|------|--------|
| 유명인 | Mr. Beast, 트럼프 | 쉬움 (모델이 알 수 있음) |
| 반유명인 | 중간 규모 유튜버 | 중간 |
| 일반인 | 브이로그 일반인 | 어려움 (순수 시각 re-ID 필요) |

→ fine-tuning 후 "모르는 사람"에서도 성능 향상이 핵심 claim

### q_I 자동 생성 파이프라인
```
Visual 타입 댓글 (텍스트)
    ↓
Grounding DINO (댓글 텍스트 → 해당 객체 탐지 → crop)
    ↓
SigLIP / DINOv2 (crop embedding)
    ↓
같은 영상 내 다른 타임스탬프에서 re-ID 매칭
  조건: cosine_sim > 0.55 AND |t1-t2| > 30초
    ↓
q_I = 다른 시점의 동일 객체 crop
Ground truth = [t2-δ, t2+δ]
```

### Frame sampling 전략
- 전체 영상 0.1fps 샘플링 (10초에 1프레임)
- 20분 영상 → 120 프레임
- 200개 영상 → 24,000 프레임 → YOLO+DINOv2 처리 충분히 가능

---

## 4. LanguageBind Fine-tuning 방식

### 기존 (text-only)
```
query = text_embedding
target = video_segment_embedding
→ contrastive loss
```

### IMR (image+text)
```
query = (image_emb + text_emb) / 2  ← weighted sum (단순하게 시작)
target = video_segment_embedding
→ 같은 contrastive loss
```

LanguageBind는 image/text/video 모두 같은 embedding space → fusion이 자연스러움

---

## 5. 어드바이저 피드백 요약 (2026-03-26)

- Visual query = 별도 논문 수준의 contribution
- 현재 논문에 붙이려면 "fancy한 방법론" 필요 (단순 평균 fusion 부족)
- 추천: REN 논문 (DINO+SigLIP Subject ID) 참고
- LanguageBind fine-tuning 방향은 좋음 (1B 미만이라 가능)
- 2번(LanguageBind fine-tuning) + 4번(데이터 확장)만으로도 arXiv 가능
- Visual query는 arXiv에 별도로 남겨도 좋음

---

## 6. 현재 진행 상황 (2026-03-26)

- [x] 댓글 Visual/Audio 타입 분류 완료
- [x] Mixed Modality 방법론 구현 완료 (돌리기만 하면 됨)
- [x] 영상 다운로드 진행 중 (300 → 1000개, 서버)
- [ ] Grounding DINO + SigLIP re-ID 파이프라인 구현
- [ ] (q_I, q_T, video_segment) 트리플 생성
- [ ] LanguageBind fine-tuning
- [ ] MomentSeeker IMR 평가 (baseline 18.2 R@1 넘기)
- [ ] Human eval 지시문 수정

---

## 7. 일정 (목표: 5월 초 완성, 5월 25일 EMNLP 제출)

```
3월 4주  영상 다운로드 + 파이프라인 설계
4월 1주  Grounding DINO + SigLIP 파이프라인 구현
4월 2주  IMR 트리플 생성 + 데이터 품질 확인
4월 3주  LanguageBind fine-tuning
4월 4주  MomentSeeker IMR 평가 + ablation
5월 1주  논문 작성
5월 초   Draft 완성
```
