---
title: "이미지 쿼리로 영상 장면 찾기: Visual Query VMR 연구 동향"
dek: "텍스트만으론 부족한가? Composed Retrieval과 Image-Conditioned Moment Search 최신 흐름 정리"
tags: ["Video", "Multimodal", "LLM"]
date: "2026-03-16"
readtime: "12"
slug: "visual-query-survey"
katex: false
---

## 왜 이걸 조사하는가

Video Moment Retrieval(VMR)의 전통적인 설정은 단순하다. 텍스트 쿼리 하나를 주면, 영상에서 해당 순간을 찾아준다. 그런데 최근 2~3년 사이 연구 커뮤니티는 조용히 다른 질문을 던지기 시작했다.

> "텍스트만으로는 정말 충분한가?"

사람이 어떤 장면을 기억할 때, 언어만으로 표현하기 어려운 순간들이 있다. 특정 표정, 특정 제스처, 이름 모를 음악이 깔린 장면. 이럴 때 **이미지나 영상 클립을 쿼리로 주면 어떨까**라는 아이디어가 Composed Retrieval이라는 이름으로 빠르게 성장하고 있다.

이 포스트는 해당 흐름을 정리한 리서치 노트다.

---

## 핵심 트렌드: Composed Video Retrieval

**Composed Retrieval**의 기본 형태는 이렇다.

```
입력: 레퍼런스 이미지 + 텍스트 modifier
출력: 수정된 조건을 만족하는 영상 / 모멘트
```

"이 장면인데, 밤 버전으로 찾아줘" 같은 식이다. 레퍼런스 이미지가 시각적 맥락을 제공하고, 텍스트가 그것과의 차이(delta)를 표현한다.

<div class="pullquote">
"Reference image anchors the visual content. Text expresses the user's intent to modify it."
</div>

이 패러다임을 정의한 논문이 **CoVR-2** (TPAMI 2024)다.

---

## 주요 논문 정리

### 1. CoVR-2: Composed Video Retrieval의 기반
**IEEE TPAMI 2024 · arXiv 2308.14746**

Lucas Ventura 외가 WebVid-CoVR이라는 1.6M 트리플렛 데이터셋을 자동으로 구축했다. 각 트리플렛은 `(레퍼런스 영상, 텍스트 수정 지시, 타겟 영상)`으로 구성된다. BLIP-2 기반 retrieval 모델을 학습해 composed image retrieval 벤치마크에서 강력한 zero-shot transfer를 보여줬다.

**핵심 기여:** Composed Video Retrieval을 공식 태스크로 정립하고 대규모 자동 데이터 구축 파이프라인을 제시.

---

### 2. Composed Video Retrieval via Enriched Context
**CVPR 2024 · arXiv 2403.16997**

Omkar Thawakar 외. 레퍼런스 이미지 + 텍스트 modifier 조합에서 vision-only, text-only, composed 임베딩을 각각 분리 인코딩하고, 풍부한 언어 설명으로 시각적 맥락 공백을 채운다. CoVR 벤치마크에서 recall 약 7% 개선.

**TCVP에 주는 시사점:** 이미지와 텍스트를 단순히 concat하는 것보다, 각 모달리티의 역할을 분리 설계하는 것이 중요함을 보여준다.

---

### 3. HUD: Hierarchical Uncertainty-Aware Disambiguation
**ACM MM 2025 · arXiv 2512.02792**

Composed Retrieval의 핵심 실패 모드 두 가지를 정의한다.

1. **Subject referring ambiguity** — 이미지 속 어느 객체를 텍스트가 지칭하는가?
2. **Fine-grained semantic focus 부족** — 텍스트가 이미지의 어느 부분에 집중해야 하는가?

계층적 불확실성 모델링으로 이 두 가지를 해소한다.

**핵심 메시지:** 이미지 쿼리는 종종 under-specified하다. 사용자의 의도를 disambiguate하는 메커니즘이 필수다.

---

### 4. MomentSeeker: Long-Video Moment Retrieval 벤치마크
**arXiv 2502.12558 (2025) · 268개 영상, 평균 1,202초**

Huaying Yuan 외. 긴 영상에서의 moment retrieval을 체계적으로 벤치마킹한다.

**세 가지 쿼리 형태 (5:2:2 비율):**

| 태스크 | 입력 | 쿼리 수 |
|---|---|---|
| TMR (텍스트) | q_T | 900 |
| IMR (이미지+텍스트) | q_T + q_I | 360 |
| VMR (영상+텍스트) | q_T + q_V | 360 |

<div class="callout">
<strong>IMR 정확한 입출력 정의</strong><br><br>
입력: 텍스트 쿼리 q_T <strong>+ 레퍼런스 이미지 q_I</strong> + 긴 영상 V<br>
출력: [[t_start_1, t_end_1], ..., [t_start_n, t_end_n]] (1≤n≤5)<br><br>
이미지 q_I는 같은 영상의 keyframe이거나 다른 영상의 관련 프레임. 어노테이터가 직접 선택.
</div>

**주요 베이스라인 성능 (Overall R@1, IoU=0.3):**

| 모델 | 타입 | R@1 | mAP@5 |
|---|---|---|---|
| Gemini-2.5-Pro | Generation | **29.6** | 31.4 |
| GPT-4o | Generation | 18.2 | 18.9 |
| Qwen2.5VL-72B | Generation | 17.2 | 16.9 |
| InternVideo2 | Retrieval | 19.7 | 26.6 |
| LanguageBind | Retrieval | 18.2 | 25.4 |
| CoVR | Retrieval | 13.0 | 18.5 |

**핵심 발견 — 왜 IMR 성능이 낮은가:**

논문이 명시적으로 밝힌 이유:

> "No model was specifically designed or fine-tuned for IMR. All tested models fail at deeper multi-modal integration and reasoning across modalities."

즉 태스크가 근본적으로 어려운 게 아니라, **IMR로 학습한 모델이 하나도 없기 때문**이다. 현재 모델들은 전부 zero-shot으로 IMR을 수행하고 있다.

논문은 명시적으로 이렇게 제안한다:

> "Better multi-modal query fusion and training data specifically designed for IMR would significantly improve performance."

---

### 5. SketchQL: 스케치로 영상 순간 찾기
**VLDB 2024 · arXiv 2405.18334**

텍스트 대신 **객체의 이동 궤적을 스케치**로 그려서 쿼리로 사용한다. 사전 학습된 궤적 인코더로 zero-shot 유사도 매칭. 훈련 데이터 불필요.

**의의:** 순수 시각적 쿼리 인터페이스의 가능성을 보여준다. 사용자가 텍스트로 표현 못하는 움직임 패턴을 이미지/스케치로 전달할 수 있다.

---

### 6. VIRTUE: Unified Multimodal Video Retrieval
**arXiv 2601.12193 (2026)**

MLLM 기반으로 corpus-level 검색(영상 전체)과 moment-level 검색(클립 단위)을 하나의 모델로 통합. 텍스트 쿼리와 composed 쿼리(이미지+텍스트) 동시 지원. LoRA로 700K 쌍 데이터 학습.

**핵심 기여:** 쿼리 형태와 검색 granularity를 모두 통합하는 방향의 선구적 시도.

---

### 7. GranAlign: 쿼리 granularity 정렬 문제
**AAAI 2026 · arXiv 2601.00584**

Mingyu Jeon 외. 텍스트 쿼리가 다양한 추상화 레벨(coarse vs. fine-grained)로 작성되지만, 영상 설명은 보통 단일 레벨에 머문다는 mismatch를 지적한다. Training-free 프레임워크로 쿼리를 여러 granularity로 재작성하고, 영상도 generic + query-specific 캡션을 동시 생성. QVHighlights zero-shot에서 +3.23% mAP.

**TCVP에 주는 시사점:** 쿼리의 구체성 수준이 retrieval 성능에 직접 영향을 준다. 댓글 기반 쿼리는 granularity가 다양하므로 이를 고려해야 한다.

---

### 8. Object-Centric VMR
**AAAI 2026 · arXiv 2512.18448**

Zongyao Li 외. 프레임 단위 특징 대신, 쿼리 관련 객체에서 scene graph를 구성하고 relational tracklet transformer로 시공간 관계를 모델링. Charades-STA, QVHighlights, TACoS에서 성능 개선.

**의의:** 이미지 쿼리가 "이 객체가 이 행동을 하는 순간 찾기"로 구체화될 때, 객체 중심 접근이 효과적임을 시사.

---

### 9. Moment of Untruth: Negative Query 처리
**WACV 2025 · arXiv 2502.08544**

Kevin Flanagan 외. 쿼리가 가리키는 순간이 영상에 존재하지 않는 경우를 체계적으로 다룬 최초 연구. Negative-Aware VMR(NA-VMR) 제안. UniVTG-NA가 98.4% negative rejection accuracy 달성.

**TCVP에 주는 시사점:** 이미지 쿼리가 코퍼스 내 어떤 영상에도 없는 장면을 가리킬 수 있다. 이를 gracefully reject하는 메커니즘이 real-world 시스템에서 필수다.

---

### 10. MINOTAUR: Multi-task Video Grounding From Multimodal Queries
**arXiv 2302.08063 (Meta AI, 2023)**

Raghav Goyal 외 (Meta). 이미지 crop, 텍스트, 액티비티 레이블 세 가지 쿼리 형태를 단일 통합 모델로 처리한다. Ego4D 에피소딕 메모리 태스크 세 가지를 하나의 아키텍처로 해결하며, 크로스-태스크 학습이 개별 태스크 성능을 개선함을 보였다.

**중요한 한계:** 시각적 쿼리(이미지 crop)가 같은 Ego4D 영상 도메인에서 추출된 것이다. 일반 웹 이미지를 쿼리로 사용하는 시나리오는 다루지 않는다.

---

### 11. EAGLE: Episodic Appearance- and Geometry-aware Memory
**AAAI 2026 · arXiv 2511.08007**

Yifei Cao 외. 1인칭 에고센트릭 영상에서 이미지 쿼리(객체 crop)가 주어지면 2D 세그멘테이션 마스크와 3D 공간 좌표까지 동시에 찾아준다. Dual memory 시스템(외형 + 기하학)으로 시점 변화에도 강인하다.

**중요한 한계:** Ego4D 전용. 카메라가 물체를 직접 본 기록이 있는 1인칭 영상에서만 작동한다.

---

## 연구 지형 요약

```
Composed Video Retrieval (이미지+텍스트 동시 쿼리)
├── 태스크 정의: CoVR-2 (TPAMI'24)
├── 모델 개선: CVPR'24 Thawakar
├── Disambiguation: HUD (MM'25)
└── 시계열 확장: TF-CoVR (2025)

Image-Conditioned Moment Search — 일반 영상
└── 벤치마크만 있음: MomentSeeker (2025)
    → LanguageBind R@1 = 4.8, 데이터셋은 없음 ← 공백

Image Query — Ego4D 한정
├── MINOTAUR (Meta, 2023) — 멀티태스크 통합
└── EAGLE (AAAI'26) — 2D+3D 위치 추정

Visual-only Query
└── Sketch 기반: SketchQL (VLDB'24)

Unified Multimodal Retrieval
└── VIRTUE (2026) — 쿼리 타입 × 검색 granularity 통합

보조 주제
├── Query granularity: GranAlign (AAAI'26)
├── Object-centric: AAAI'26
└── Negative query: WACV'25
```

---

## 핵심 공백: 아직 아무도 안 한 것

두 에이전트가 수십 편의 논문을 조사한 결과, 하나의 명확한 공백이 드러났다.

<div class="callout" style="border-left-color: #c0392b;">
<strong>발견된 공백</strong><br><br>
일반 YouTube 영상(3인칭)에서, 실제 사용자의 의도에서 비롯된 이미지 쿼리로 모멘트를 찾는 데이터셋은 2025년 기준으로 존재하지 않는다.<br><br>
• Ego4D 이미지 쿼리 → MINOTAUR, EAGLE이 이미 다룸<br>
• 일반 영상 이미지 쿼리 벤치마크 → MomentSeeker가 정의했지만 데이터셋은 없음<br>
• Real user intent 기반 이미지+텍스트 쿼리 쌍 → <strong>아무것도 없음</strong>
</div>

TCVP는 타임스탬프 댓글에서 텍스트 쿼리와 keyframe을 동시에 추출할 수 있다. 이 구조가 위 공백을 자연스럽게 채운다.

---

## 전체 논문 정리표

| 논문 | 연도 | 베뉴 | 쿼리 타입 | 영상 도메인 | 핵심 |
|---|---|---|---|---|---|
| CoVR-2 | 2024 | TPAMI | 이미지+텍스트 | 일반 웹 | Composed Retrieval 정립 |
| CVPR'24 Thawakar | 2024 | CVPR | 이미지+텍스트 | 일반 웹 | 모달리티 분리 인코딩 |
| HUD | 2025 | MM | 이미지+텍스트 | 일반 웹 | Disambiguation |
| MomentSeeker | 2025 | arXiv | 텍스트/이미지/영상 | 일반 (영화 등) | IMS 벤치마크 |
| MINOTAUR | 2023 | arXiv | 이미지 crop+텍스트 | **Ego4D만** | 멀티태스크 통합 |
| EAGLE | 2026 | AAAI | 이미지 crop | **Ego4D만** | 2D+3D 위치 추정 |
| SketchQL | 2024 | VLDB | 스케치 | 일반 | 순수 시각 쿼리 |
| VIRTUE | 2026 | arXiv | 텍스트+composed | 일반 | 통합 멀티모달 |
| GranAlign | 2026 | AAAI | 텍스트 | 일반 | 쿼리 granularity |
| Object-Centric VMR | 2026 | AAAI | 텍스트 | 일반 | 객체 중심 접근 |
| Moment of Untruth | 2025 | WACV | 텍스트 | 일반 | Negative query |

---

## TCVP에 주는 시사점

**첫째 — 입출력이 정확히 일치한다.**

MomentSeeker IMR의 입력이 `(q_T, q_I)` 쌍이라는 게 확인됐다. TCVP가 타임스탬프 댓글에서 자연스럽게 만드는 게 바로 `(댓글 텍스트, keyframe)` 쌍이다. 구조가 완벽하게 맞아떨어진다.

```
MomentSeeker IMR 요구: (q_T, q_I) → (t_start, t_end)
TCVP가 제공:          (댓글 쿼리, 타임스탬프 keyframe) → 타임스탬프
```

**둘째 — 성능이 낮은 이유가 "데이터 없음"이다.**

논문이 명확히 밝혔다: 모든 모델이 zero-shot으로 IMR을 수행하고 있다. IMR 전용 학습 데이터가 존재하지 않기 때문이다. TCVP가 real user intent 기반 IMR 학습 데이터를 제공하면, LanguageBind를 fine-tuning해서 MomentSeeker IMR 성능을 직접 끌어올릴 수 있다. 이게 **downstream training 실험**이 된다.

```
현재: LanguageBind zero-shot IMR → 낮은 성능
TCVP fine-tuning 후: MomentSeeker IMR 성능 ↑
→ "real user intent 데이터가 IMR 성능의 핵심이었다"
```

**셋째 — Ego4D 한계가 TCVP의 기회다.**

MINOTAUR, EAGLE 모두 Ego4D(1인칭) 전용이다. 일반 YouTube 영상(3인칭)에서 real user intent 기반 IMR 데이터는 아직 없다.

**넷째 — HUD의 disambiguation 문제를 구조적으로 해결한다.**

이미지 단독 쿼리는 under-specified하다(HUD). TCVP의 댓글 텍스트가 이 모호성을 자연스럽게 해소한다. 모델이 해결하려던 문제를 데이터 구조가 처음부터 막아준다.

---

## TCVP 다음 버전의 그림

지금까지 조사한 내용을 종합하면, TCVP v2의 contribution 구조가 이렇게 잡힌다.

```
[데이터 기여]
YouTube 타임스탬프 댓글
  → (q_T, q_I) IMR 쌍 자동 생성
  → real user intent 기반 최초 IMR 학습 데이터

[태스크 기여]
MomentSeeker IMR을 baseline으로
  → TCVP 데이터로 LanguageBind fine-tuning
  → IMR 성능 향상 증명

[분석 기여]
Mixed modality 본론 편입
  → 댓글의 시각/청각/혼합 분포 분석
  → IMR에서 각 모달리티의 역할 ablation
```
