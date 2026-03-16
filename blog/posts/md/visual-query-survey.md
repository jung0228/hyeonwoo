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
**arXiv 2502.12558 (2025)**

Huaying Yuan 외. 긴 영상(평균 1200초)에서의 moment retrieval을 체계적으로 벤치마킹한다. 세 가지 레벨을 포함한다.

| 레벨 | 설명 | 예시 태스크 |
|---|---|---|
| Global | 영상 전체 맥락 | 이벤트 인식 |
| Event | 특정 사건 구간 | 행동 로컬라이제이션 |
| Object | 특정 객체 추적 | 객체 위치 확인 |

그리고 세 가지 쿼리 형태를 지원한다: **텍스트 쿼리**, **이미지 조건 쿼리(IMS)**, **영상 조건 쿼리**.

IMS(Image-conditioned Moment Search)에서 LanguageBind의 R@1이 **4.8**에 불과하다는 것이 확인됐다. 태스크가 그만큼 어렵다는 뜻이다.

<div class="callout">
<strong>IMS 태스크 정의 (MomentSeeker)</strong><br>
입력: 이미지 q_v + 긴 영상 V<br>
출력: V 내에서 q_v와 가장 관련 있는 순간 (t_start, t_end)<br>
현재 SOTA LanguageBind R@1 = 4.8 — 아직 매우 어려운 태스크
</div>

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

## 연구 지형 요약

```
Composed Video Retrieval (이미지+텍스트 동시 쿼리)
├── 태스크 정의: CoVR-2 (TPAMI'24)
├── 모델 개선: CVPR'24 Thawakar
├── Disambiguation: HUD (MM'25)
└── 시계열 확장: TF-CoVR (2025)

Image-Conditioned Moment Search (IMS)
└── 벤치마크: MomentSeeker (2025)
    → LanguageBind R@1 = 4.8 (여전히 매우 어려운 태스크)

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

## TCVP에 주는 시사점

세 가지를 정리할 수 있다.

**첫째**, Composed Retrieval 트렌드에서 데이터가 핵심 병목이다. CoVR-2조차 자동 생성 데이터를 쓴다. TCVP는 실제 사용자가 남긴 타임스탬프 댓글에서 쿼리를 추출하므로, **real user intent 기반 데이터**라는 차별점이 살아있다.

**둘째**, MomentSeeker IMS에서 LanguageBind R@1이 4.8이라는 숫자는 기회다. 태스크는 정의됐지만 데이터가 없어서 성능이 낮다. TCVP가 real-world IMS 데이터를 제공하면 의미 있는 기여가 된다.

**셋째**, HUD가 지적한 것처럼 이미지 쿼리는 under-specified하다. 이 모호성을 텍스트 쿼리(댓글)가 해소하는 구조 — 즉 TCVP의 `(댓글 텍스트, 타임스탬프 keyframe)` 쌍 — 가 자연스럽게 이 문제를 해결한다. 이것이 단순한 기능 추가가 아닌 구조적 기여가 될 수 있는 이유다.
