---
title: 딥리서치 보고서에 이미지를 넣는다
dek: 텍스트 중심이던 Deep Research 보고서에 차트와 이미지를 어떻게 넣는지, 그리고 독자마다 다른 시각화를 주는 personalization까지 7편의 논문으로 따라간다.
desc: Deep Research 보고서가 텍스트 일색에서 벗어나 차트·이미지를 본문에 끼워 넣는 multimodal report generation 연구를, 이미지 생성·검색·검증·평가·개인화 다섯 축으로 7편의 논문을 통해 정리한다.
tags: [Agent, LLM, Multimodal]
date: Jun 2026
readtime: 24 min read
slug: multimodal-deep-research-reports
sans: true
---

ChatGPT, Gemini, Perplexity의 "Deep Research"가 내놓는 보고서는 잘 쓰였지만 대개 **글뿐이다**. 그런데 사람이 쓰는 진짜 리서치 보고서를 떠올려 보자. 재무 분석에는 추세 차트가, 시장 보고서에는 점유율 파이가, 기술 문서에는 아키텍처 다이어그램이 박혀 있다. 그림은 장식이 아니라 **근거이자 설명 수단**이다.

이 글은 두 가지 질문을 따라간다. 첫째, **딥리서치가 보고서를 쓸 때 이미지를 어떻게 넣는가** — 직접 그리는가(차트 생성), 웹에서 가져오는가(이미지 검색), 그리고 그게 본문 주장과 진짜 맞물리는지 어떻게 검증·평가하는가. 둘째, 한 걸음 더 나아가 **사람마다 잘 이해하는 시각화가 다른데, 독자에 맞춰 그림을 다르게 줄 수 있는가** — 시각화의 personalization 문제다.

<div class="callout">
<strong>다루는 소스 (7편)</strong>

<ul style="margin:.4em 0 0;padding-left:1.2em;">
<li>Multimodal DeepResearcher <span style="color:#888;">(arXiv:2506.02454)</span> — 차트를 직접 생성하는 text-chart interleaved 보고서</li>
<li>Deep-Reporter <span style="color:#888;">(arXiv:2604.10741)</span> — 실제 이미지를 검색해 넣는 multimodal RAG</li>
<li>TVIR <span style="color:#888;">(arXiv:2606.02320)</span> — 검색+생성 통합, 텍스트·시각 dual-path 평가</li>
<li>Ptah <span style="color:#888;">(arXiv:2605.29861)</span> — Visual Working Memory와 verifier로 신뢰성 검증</li>
<li>MMDeepResearch-Bench <span style="color:#888;">(arXiv:2601.12346)</span> — 멀티모달 딥리서치 평가 벤치마크</li>
<li>Drillboards <span style="color:#888;">(arXiv:2410.12744)</span> — 독자 expertise에 맞춰 적응하는 시각화 대시보드</li>
<li>StoryLensEdu <span style="color:#888;">(arXiv:2602.17067)</span> — 학습자 개인화 narrative 리포트</li>
</ul>
</div>

## Part 0 — Multimodal Deep Research란 무엇인가

Deep Research(DR)는 질문 하나를 받아 여러 step에 걸쳐 웹을 검색·추론하고, 인용이 달린 long-form 보고서를 생성하는 패러다임이다. 그런데 지금까지의 DR은 출력이 거의 **글뿐**이었다. **Multimodal Deep Research**(또는 multimodal report generation)는 이 보고서 생성 단계를 확장해, 본문에 **차트·다이어그램·이미지를 함께 짜 넣는(text-visual interleaved)** 것을 목표로 하는 갈래다. TVIR은 이 task를 아예 이름으로 못 박는다.

> "rethinking deep research not as a purely textual task, but as a multimodal synthesis problem in which text and visuals must be jointly generated, and evaluated."

핵심은 시각 요소가 **사후 장식이 아니라 추론·근거의 일부**여야 한다는 것이다. 그래서 이 분야의 질문은 하나가 아니라 다섯 축으로 갈라진다 — 이미지를 **(1) 만들 것인가, (2) 가져올 것인가**, 둘을 **(3) 어떻게 통합·검증할 것인가**, 결과를 **(4) 어떻게 평가할 것인가**, 그리고 한 걸음 더 — **(5) 독자마다 다른 시각화를 줄 수 있는가**.

이 글이 다루는 7편을 이 다섯 축에 올려놓으면 분야의 지형이 한눈에 보인다.

| 축 | 무엇을 푸는가 | 논문 |
|---|---|---|
| ① 이미지 **생성** | 데이터→차트를 직접 그려 넣기 | Multimodal DeepResearcher |
| ② 이미지 **검색** | 웹에서 실제 시각 근거 가져오기 | Deep-Reporter |
| ③ **통합·검증** | 생성+검색을 합치고 정합성 검증 | TVIR · Ptah |
| ④ **평가** | 보고서의 시각 근거를 객관적으로 채점 | MMDeepResearch-Bench |
| ⑤ **개인화** | 독자 expertise·배경에 맞춘 시각화 | Drillboards · StoryLensEdu |

아래에서는 먼저 "왜 이미지인가"(Part 1)로 분야의 공통 출발점을 잡고, ①–⑤ 축을 Part 2–6에서 논문별로 따라간 뒤, Part 7에서 아직 비어 있는 자리를 짚는다.

## Part 1 — 왜 보고서에 이미지인가

거의 모든 논문이 같은 출발점을 공유한다. 딥리서치는 multi-step 검색·추론·long-form 생성에서 강해졌지만, **벤치마크도 시스템도 여전히 text-centric**이라는 것. TVIR은 이를 한 문장으로 요약한다.

> "existing benchmarks and systems remain predominantly text-centric, with limited evaluation of whether visual elements are factually reliable and well aligned with the surrounding analysis."

실제 전문 보고서는 글만으로 굴러가지 않는다. <span class="q">"interleave narrative analysis with charts, diagrams, and images that serve as evidential artifacts"</span> — 서사와 시각 근거가 번갈아 가며 주장(claim)을 떠받친다. 그런데 현재 시스템에서 시각 요소는 <span class="q">"treated as decorative supplements rather than first-class reasoning components"</span>, 즉 사후에 끼워 넣는 장식 취급이다.

그렇다면 보고서에 들어가는 이미지는 **두 종류**다. Ptah가 깔끔하게 나눈다.

<figure>
<img src="img/multimodal-deep-research-reports/ptah_intro.png" alt="Interleaved image-text reports: benefits (illustration images, evidence images) vs challenges (incorrect references, invalid references, uninformative images, improper placement)">
<figcaption><strong>Figure 1</strong> — 보고서 속 이미지의 두 역할과 함정. <strong>Illustration images</strong>는 개념을 구체화해 인지 부하를 줄이고, <strong>Evidence images</strong>는 측정 가능한 결과로 주장을 뒷받침한다. 오른쪽은 흔한 실패: 잘못된/무효한 image reference, 정보 없는 이미지, 엉뚱한 위치 배치. 출처: Ptah (arXiv:2605.29861), Fig 1.</figcaption>
</figure>

- **Illustration(설명용)** — 개념을 그림으로 구체화해 이해를 돕는다. "Transformer 구조"를 글로만 설명하는 대신 도식을 보여주는 식.
- **Evidence(근거용)** — 수치·결과를 차트로 제시해 주장을 입증한다. "성능이 떨어진다"가 아니라 떨어지는 곡선 그 자체.

그리고 이 이미지를 보고서에 넣는 방법도 **두 갈래**로 갈린다. Deep-Reporter의 그림이 이 분기를 정확히 보여준다.

<figure>
<img src="img/multimodal-deep-research-reports/dreporter_paradigm.png" alt="Three paradigms: (a) text-only report, (b) text-to-image generation with hallucinations, (c) retrieving and integrating real-world visual evidence">
<figcaption><strong>Figure 2</strong> — long-form 보고서 생성의 세 패러다임. (a) 전통적 text-only는 시각 정보가 없고, (b) 순수 T2I 생성은 <span class="q">"factual hallucinations and narrative fragmentation"</span>에 시달리며, (c) 실제 시각 근거를 검색·통합하면 coherence와 factuality를 함께 얻는다. 출처: Deep-Reporter (arXiv:2604.10741), Fig 1.</figcaption>
</figure>

이미지를 **만들 것인가(generate)**, **가져올 것인가(retrieve)**. 데이터로부터 차트를 그려내는 쪽은 정확하지만 표현이 차트에 한정되고, 웹에서 실제 이미지를 가져오는 쪽은 풍부하지만 환각·정합성 위험이 따른다. 아래 Part 2·3은 각각의 갈래를, Part 4는 둘을 합치고 검증까지 붙인 시스템을 본다.

## Part 2 — 이미지를 만든다: 차트 생성

**문제의식.** 딥리서치 프레임워크는 학계·산업 모두 <span class="q">"predominantly focus on generating textual content"</span>다. Multimodal DeepResearcher는 정면으로 묻는다 — LLM이 처음부터 **글과 차트가 번갈아 짜인(text-chart interleaved) 보고서**를 통째로 생성하게 할 수 있는가? 개별 차트는 코딩으로 그릴 수 있지만, 그 차트를 텍스트 문맥에 자연스럽게 엮을 **표현**(representation)이 없다는 게 진짜 병목이다.

**핵심 제안.** **Formal Description of Visualization**(FDV) — 차트를 구조화된 텍스트로 기술하는 표현이다. grammar of graphics 이론에서 영감을 받아 시각화를 **네 관점**으로 적는다.

- **Overall layout** — 어떤 subplot들로 구성되고 공간적으로 어떻게 배치되는지.
- **Plotting scale** — "데이터 → 시각 채널(위치·색 등)" 매핑의 스케일링 논리와 주석.
- **Data** — 차트 생성에 쓰인 수치 데이터와 텍스트 요소.
- **Marks** — 각 시각 요소(점·선·막대 등)의 디자인 명세.

핵심은 **양방향**이라는 점이다. MLLM이 사람 전문가의 차트 이미지에서 FDV를 추출(textualize)하면 그 보고서가 통째로 텍스트가 되어 in-context 예시가 되고, 거꾸로 FDV를 코드로 구현(reconstruct)하면 다시 차트가 된다. 차트를 "글처럼" 다룰 수 있게 만든 표현이다.

**작동 흐름.** FDV 위에서 네 단계로 돈다.

<figure>
<img src="img/multimodal-deep-research-reports/mmdr_overview.png" alt="Multimodal DeepResearcher four stages: researching, exemplar textualization with FDV, planning, multimodal report generation with drafting/coding/refining">
<figcaption><strong>Figure 3</strong> — Multimodal DeepResearcher의 4단계. (A) Researching: 주제를 반복 조사, (B) Exemplar Textualization: 전문가 보고서를 FDV(layout·scale·data·marks)로 텍스트화, (C) Planning: outline + 시각화 스타일 가이드, (D) Report Generation: drafting → coding → 반복 refining으로 최종 보고서. 출처: Multimodal DeepResearcher (arXiv:2506.02454), Fig 1.</figcaption>
</figure>

1. **Researching** — 주제 `t`에서 키워드를 뽑아 웹 검색 → 결과를 추론·종합해 learnings `L`을 만들고, 다음 라운드의 research question을 세워 반복한다(`n_R`회).
2. **Exemplar textualization** — 사람 전문가의 멀티모달 보고서에서 차트마다 MLLM으로 FDV를 추출해 이미지를 FDV 텍스트로 치환, in-context 예시 `R̃`을 만든다.
3. **Planning** — learnings·exemplar를 바탕으로 hierarchical outline `O`와, 색 팔레트·폰트 위계 같은 **시각화 스타일 가이드** `G`(보고서 전체 차트 스타일을 일관되게)를 정한다.
4. **Multimodal generation** — 먼저 차트 자리에 FDV를 placeholder로 둔 텍스트 보고서를 쓰고, 각 FDV를 **D3.js 코드**로 구현한다. 여기에 **actor-critic 루프**가 붙는다 — actor LLM이 코드를 짜면 브라우저로 렌더링해 console 에러와 스크린샷을 얻고, critic MLLM이 시각 품질을 보고 피드백을 준다. critic이 만족하거나 **최대 3회** 재시도까지 반복한 뒤 마지막 두 후보 중 더 나은 것을 고른다.

D3.js를 쓰는 이유도 분명하다 — matplotlib 같은 declarative 라이브러리로는 FDV가 표현하는 자유로운 디자인을 다 못 그리기 때문에, imperative한 D3.js로 떨어뜨린다.

**결과.** 동일한 Claude 3.7 Sonnet으로 baseline(DataNarrative) 대비 <span style="background:#fef3c7;padding:0 .2em;">**82% overall win rate**</span>. report-level·chart-level 각각 5개 지표로 자동·사람 평가했고, 생성 차트도 막대·선에 그치지 않고 stacked area, Sankey, infographic, 타임라인 막대, 파이 등으로 다양하다.

<figure>
<img src="img/multimodal-deep-research-reports/mmdr_cases.png" alt="Variety of generated charts: stacked area, Sankey diagram, infographic, horizontal bar, dashboard, pie chart">
<figcaption><strong>Figure 4</strong> — 생성된 차트 예시. stacked area, Sankey diagram, infographic, horizontal bar, dashboard, pie 등 표현 형식이 다양하다. 출처: Multimodal DeepResearcher (arXiv:2506.02454).</figcaption>
</figure>

**한계.** 모든 시각 요소를 **코드로 그릴 수 있는 차트**로 환원한다. 데이터 기반 차트에는 강하지만, 실제 사진·다이어그램·지도처럼 "검색해 와야 하는" 이미지는 다루지 못한다.

<div class="callout"><strong>이 논문의 위치 —</strong> "이미지를 만든다" 갈래의 대표. 데이터→차트 생성은 정확하지만 표현이 차트로 한정된다는 점이, 다음 "가져온다" 갈래의 동기가 된다.</div>

## Part 3 — 이미지를 가져온다: 멀티모달 검색

**문제의식.** 차트 생성만으로는 부족하다. 진짜 보고서에는 사진·실측 도표·실제 figure가 들어간다. Deep-Reporter는 text-only 생성과 순수 T2I 생성(환각 문제)을 모두 비판하며, **웹에서 실제 시각 근거를 검색·통합**하는 길을 택한다. 다만 텍스트 기반 agentic search를 멀티모달로 확장하는 데 세 가지 난제가 있다 — agentic multimodal retrieval, coherent multimodal long-form generation, 그리고 evaluation.

**핵심 제안.** Planner·Searcher-Filter·Reporter 세 에이전트 + <span class="q">"Checklist-Guided Incremental Synthesis"</span> 합성 메커니즘. 보고서 구조를 "Semantic Anchors"로 형식화하고, 문맥을 누적 갱신하며 글과 이미지가 일관되게 interleave되도록 한다.

**작동 흐름.** 세 단계가 맞물린다.

- **Sectional Planning(이중 입도 체크리스트)** — Planner가 질의를 섹션들 `{S₁…S_N}`로 분해하되, 각 섹션을 `(D_k, C_k)`로 적는다. `D_k`는 거친 입도의 내용 범위, `C_k`는 그 섹션이 다뤄야 할 **fact·argument를 콕 집은 semantic anchor**들. 이 이중 입도가 일관성과 정밀도를 동시에 잡는다.
- **Agentic Multimodal Search & Filtering** — 섹션마다 **두 갈래 쿼리**를 만든다. Narrative Query(`Q_txt`)는 사실·통계 passage를, Visual Query(`Q_img`)는 차트·다이어그램·infographic을 노린다. 검색된 후보 풀은 필터가 거른다 — 텍스트는 LLM이 `(D_k, C_k)`와의 entailment를 보고, 이미지는 **VLM이 informativeness를 판정**해 장식용 이미지를 버리고 정보 밀도 높은 그림만 남긴다.
- **Incremental Synthesis(순환 문맥 관리)** — 매 섹션을 plan·evidence·memory·position 조건으로 생성한다. 이미지가 토큰을 많이 먹어 전체를 한 프롬프트에 넣으면 context overflow가 나므로, 과거 문맥을 `m_global`(서사 전체의 재귀 요약)과 `m_local`(직전 섹션의 원문 꼬리)로 압축해 들고 다닌다. 이미지는 caption으로 텍스트화해 두고, 모델이 `![](cite:img1)` 같은 인용을 **적절한 위치에 삽입**하도록 학습된다.

<figure>
<img src="img/multimodal-deep-research-reports/dreporter_arch.png" alt="Deep-Reporter framework: multi-agent planning, multimodal information seeking, incremental writing; plus agentic trace construction pipeline">
<figcaption><strong>Figure 5</strong> — Deep-Reporter 구조. (a) Inference: 멀티 에이전트가 planning·멀티모달 검색·점진적 작성을 오케스트레이션. (b) Training: 8K개 expert trajectory를 합성해 open-weight 모델에 멀티모달 능력을 주입. 출처: Deep-Reporter (arXiv:2604.10741), Fig 2.</figcaption>
</figure>

**결과.** open-weight 모델은 멀티모달 agentic 능력이 없으므로, 3단계 trace 합성 파이프라인으로 학습 데이터를 만든다 — ① 전문가가 9개 도메인에 걸쳐 outline·checklist를 다듬어 **1K개 질의**를 만들고, ② frontier 모델로 프레임워크를 돌려 전체 상호작용 trace **17K개**를 증류한 뒤, ③ 시각 환각·잘못된 이미지 인용·위치 결함 등을 공격적으로 걸러 **8K개 고품질 trace**만 남긴다(전문가 500개 샘플 검증에서 자동 필터와 92.4% 일치). 평가용으로는 정적 멀티모달 sandbox(95K 이미지, 108M text chunk; 보고서당 평균 102 이미지·168 chunk가 ground-truth 근거)를 구축해 검색 알고리즘만 격리 비교할 수 있게 했다. RAG baseline을 크게 앞선다.

**한계.** 검색 기반이라 코퍼스 품질·검색 정밀도에 성능이 묶이고, 검색해 온 이미지가 본문 주장과 실제로 정합한지(image-text consistency)는 여전히 까다로운 검증 대상이다 — 바로 다음 Part의 주제다.

<div class="callout"><strong>이 논문의 위치 —</strong> "이미지를 가져온다" 갈래의 대표. 실제 시각 근거를 넣어 풍부해지지만, 정합성 검증이라는 새 숙제를 남긴다.</div>

## Part 4 — 둘 다, 그리고 검증

생성과 검색은 양자택일이 아니다. 두 연구가 둘을 합치고 거기에 **검증**을 더한다.

### TVIR — 검색 이미지 + 생성 차트, 그리고 dual-path 평가

**문제의식.** 기존 벤치마크는 text-only이거나 시각 요소가 약하다. TVIR은 딥리서치를 아예 <span class="q">"Text--Visual Interleaved Report Generation"</span> 문제로 정의하고, 시각 요소가 **특정 분석 sub-goal에 의미적으로 결속**되도록 강제한다(사후에 갖다 붙이는 게 아니라).

<figure>
<img src="img/multimodal-deep-research-reports/tvir_intro.png" alt="Benchmark comparison: DeepResearch Bench (text-only), MultimodalReportBench (charts), LiveResearchBench (text-only), TVIR-Bench (charts and images)">
<figcaption><strong>Figure 6</strong> — 벤치마크 비교. 기존 벤치마크는 text-only이거나 차트만 다루지만, TVIR-Bench는 <strong>검색한 이미지 + 코드 생성 차트</strong>가 모두 들어간 interleaved 보고서를 요구한다. 출처: TVIR (arXiv:2606.02320), Fig 1.</figcaption>
</figure>

**핵심 제안.** ① **TVIR-Bench** — 100개 태스크(중국어 50 + 영어 50), 10개 도메인·3단계 난이도, 8개 기능 유형(trend prediction·mechanism explanation·comparative analysis 등). 전문가 topic 제안 → Grok-4.1-Thinking 초안 → 3인 전문가 검토(design/factual/logical/multimodal validity) → 태스크별 검증 checklist 컴파일까지 거친다. ② 4단계 multi-agent **baseline**. ③ **dual-path 평가**.

**baseline 4단계.** ⓐ **Research-Grounded Planning** — Planner가 검색으로 정보를 모아 outline을 짜는데, 각 섹션 unit `σ_i`에 제목·요약뿐 아니라 **planned visual requirement** `V_i^req`와, 인용·URL·핵심 발견이 담긴 research note `N_i`를 붙인다. ⓑ **Visual Asset Instantiation** — 두 전문 에이전트가 나눠 맡는다. **Image Searcher**는 인물·장면·아키텍처 도식을 Google 이미지로 검색→휴리스틱 필터→**VQA로 관련성 검증** 후 선택하고, **Chart Generator**는 데이터를 검색→출처 간 일관성 확인→Python 코드 생성→sandbox 실행으로 차트를 만든다. 검색 이미지는 출처 webpage URL을, 생성 차트는 데이터 출처 URL을 보존한다. ⓒ **Context-Aware Sequential Writing** — Writer가 직전 섹션들의 제목·요약을 담은 global context로 중복을 줄이며 섹션별로 작성하고, 설명에 맞춰 시각 자산의 **삽입 위치를 직접 결정**해 Markdown으로 interleave. ⓓ **Global Index Polishing** — Polisher가 미인용 reference 제거, URL·내용 기준 전역 dedup, figure/citation 재번호를 매겨 보고서 단위로 정리한다.

**dual-path 평가.** Textual Assessment는 Citation Support·Instruction Alignment·Writing Quality·Analytical Depth&Breadth·Factual&Logical Consistency 5개, Visual Assessment는 Multimodal Composition·Figure Quality·Figure Caption Quality·Figure-Context Integration·Chart-Source Consistency 5개. 대부분 LLM-as-a-Judge(0–100)지만, Figure Quality는 해상도·선명도 같은 **CV 기반 측정**까지 섞는다.

<figure>
<img src="img/multimodal-deep-research-reports/tvir_pipeline.png" alt="TVIR data construction pipeline (topic proposal, drafting, multi-expert review) and dual-path evaluation framework (textual assessment, visual assessment)">
<figcaption><strong>Figure 7</strong> — (a) 데이터 구축: 도메인 전문가의 topic proposal → drafting → 다중 전문가 검토(design/factual/logical/multimodal). (b) 평가 프레임워크: Judge LLM이 Textual Assessment와 Visual Assessment를 함께 채점. 출처: TVIR (arXiv:2606.02320).</figcaption>
</figure>

**결과.** 9개 딥리서치 시스템 실험에서 핵심 통찰 하나 — 현재 LLM은 텍스트 유창성엔 강하지만 <span style="background:#ffd6d6;padding:0 .2em;">**"decorative" 시각을 "evidential" 시각보다 우선**</span>하는 경향이 있다. 그럴듯하지만 근거가 안 되는 그림을 넣는다는 것.

### Ptah — Visual Working Memory와 verifier로 신뢰성을 건다

**문제의식.** 딥리서치는 (1) open-endedness(정답이 없어 검증이 어려움)와 (2) multimodal interleaving 두 난제를 동시에 안는다. 기존 파이프라인은 단계별 검증이 없어 초반 노이즈가 누적되고, 이미지 통합을 <span class="q">"post-hoc decorative step"</span>로 취급한다.

**핵심 제안.** Planning·Research·Writing 3단계 harness. 검색한 이미지를 **Visual Working Memory**에 source-aligned 상태로 유지하다가 보고서에 배치하고, 각 단계 사이에 **verifier agent**가 acceptance function으로서 factual grounding·citation fidelity·cross-modal consistency를 통과시켜야 다음으로 넘어간다.

<figure>
<img src="img/multimodal-deep-research-reports/ptah_arch.png" alt="Ptah three-stage harness: Planning (planner, visual requirements), Research (researcher, visual working memory, verifier), Writing (writer, image search/generation, refine/render)">
<figcaption><strong>Figure 8</strong> — Ptah의 3단계 harness. Planning은 visual-aware 계획을, Research는 Visual Working Memory에 source-aligned 이미지 후보를 유지하며, 각 단계 사이 <strong>Verifier</strong>가 rubric/rule 기반 검증으로 통과를 결정한다. Writing은 image search·generation을 declarative tool로 호출해 최종 렌더링. 출처: Ptah (arXiv:2605.29861), Fig 2.</figcaption>
</figure>

**작동 흐름.** ⓐ **Planning** — Planner가 검색으로 도메인 지식을 훑어, 섹션별 research goal·expected evidence type과 함께 "시각 요소를 어디에, 어떤 역할로, 어떤 형태(차트/도식/스크린샷/삽화)로 넣을지"의 **visual specification**을 명시한 plan을 만든다. Verifier가 rule 기반(프로토콜·tool 제약·JSON 포맷) + LLM rubric(쿼리 coverage·섹션 일관성·시각-논증 관련성)으로 통과를 판정. ⓑ **Research** — 섹션마다 Researcher가 **병렬로** 조사해 findings·evidence·수치·표·인용·writing instruction이 담긴 research package를 만들고, 동시에 방문 webpage에서 이미지를 추출해 **Visual Working Memory**를 구축한다(rule 필터 → VLM 선별, 각 이미지에 출처 URL·문맥·섹션·의도된 역할을 함께 저장). 역시 Verifier가 claim support·수치 일관성·시각 관련성을 검사. ⓒ **Writing** — Writer가 텍스트와 image directive를 **함께** 생성(declarative)하고, harness가 세 연산을 중재한다 — **Image Reference**(VWM의 source-aligned 이미지 재사용, 우선), **Image Search**(부족하면 추가 검색), **Image Generation**(차트는 코드 렌더링, 삽화는 생성 모델).

**작동 흐름 — Test-Time Scaling.** raw 보고서를 바로 내지 않고 6단계 refinement hook을 건다 — ① Section Refine, ② **Image Refine**(각 이미지를 Keep/Delete/Edit 판정하고 Edit는 실행), ③ Overall Refine(전역 일관성·image-text alignment), ④ HTML Generate, ⑤ HTML Refine(간격·가독성), ⑥ Render(브라우저 렌더링된 사용자 대면 보고서).

**결과.** **PtahEval**은 두 축 — Image Content Quality(Visual Clarity·Cross-Modal Alignment·Information Complementarity·Evidentiary Support)와 Multimodal Presentation Quality(렌더링된 페이지의 1000×2000 viewport를 캡처해 Density-Legibility·Saliency·Encoding Diversity·Ergonomics 평가), 각 5점 Likert. DeepConsult에서 평균 <span style="background:#fef3c7;padding:0 .2em;">**16.18 vs WebThinker 7.35**</span>로 멀티모달 baseline을 크게 앞선다.

**한계(TVIR·Ptah 공통).** 검증·평가 모두 결국 LLM/VLM judge에 의존하고, "어떤 그림이 정말 evidential한가"의 판단 자체가 모델 능력에 묶인다.

<div class="callout"><strong>이 논문들의 위치 —</strong> 생성·검색을 통합하고 cross-modal 정합성 검증을 1급 단계로 끌어올렸다. 남는 질문은 "그래서 이걸 어떻게 객관적으로 측정하나" — Part 5.</div>

## Part 5 — 어떻게 평가하나

**문제의식.** open-ended 보고서는 gold answer가 없어 평가가 어렵다. MMDeepResearch-Bench는 멀티모달 딥리서치를 **integrated(통합) 능력과 atomic(기초) 능력 양쪽에서** 재는 통일 벤치마크를 만든다.

<figure>
<img src="img/multimodal-deep-research-reports/mmb_intro.png" alt="MMDR-Bench evaluates two levels: integrated (multimodal task understanding, visually-grounded planning, citation-grounded reasoning, long-form synthesis) and atomic (visual perception, web search tools, long-context understanding, instruction following)">
<figcaption><strong>Figure 9</strong> — MMDR-Bench의 두 레벨. <strong>Integrated</strong>: 멀티모달 task 이해·visually-grounded planning·citation-grounded reasoning·long-form 합성. <strong>Atomic</strong>: visual perception·web search·long-context·instruction following. 출처: MMDeepResearch-Bench (arXiv:2601.12346), Fig 2.</figcaption>
</figure>

**핵심 제안.** 140개 expert-crafted 태스크(Daily/Research 두 regime, 21개 도메인). 각 태스크는 image-text bundle로 패키징된다. 평가 프레임워크는 세 모듈이 **순차적으로** 돈다 — FLAE·TRACE를 병렬 계산하고, 둘 다 gating threshold를 넘을 때만 MOSAIC를 켠다(아니면 0점). 텍스트가 부실하면 멀티모달 채점 자체를 안 하는, 일종의 자격 게이트다.

- **FLAE**(Formula-LLM Adaptive Evaluation) — 보고서 품질을 Readability·Insightfulness·Structural Completeness 3축으로 본다. 재현 가능한 **공식 채널**(lexical diversity·섹션 구조·문장 길이 분포 등 통계를 고정 변환)과 **LLM judge 채널**을 합치되, 가중치는 task별로 적응시킨다. 공식 채널 덕에 judge 없이도 안정적·감사 가능한 점수가 나온다.
- **TRACE**(Trustworthy Retrieval-Aligned Citation Evaluation) — claim-URL 쌍을 만들어 인용된 페이지를 실제로 가져와 지지 여부를 판정(Consistency·Coverage·Fidelity). 여기에 **Visual Evidence Fidelity**(VEF)를 더한다 — task별 textualized visual ground truth에 대해 judge가 0–10점과 PASS/FAIL을 내고, **6점 미만이면 강제 FAIL**인 hard 제약. 시각 데이터를 오독하거나 환각하면 그 자체로 떨어뜨린다.
- **MOSAIC**(Multimodal Support-Aligned Integrity Check) — 이미지를 인용한 문장(MM-item)을 추출해, 차트/도식/사진 **유형별로 라우팅**한 뒤 유형에 맞는 검사를 한다(차트는 수치 타당성, 도식은 구조 대응, 사진은 의미 grounding). Visual-Semantic Alignment·Data Interpretation Accuracy·Complex VQA Quality 3축으로 채점.

**결과.** 25개 SOTA 시스템 평가에서 writing quality·citation discipline·multimodal grounding 사이의 **persistent trade-off**가 드러난다 — 셋을 동시에 잘하는 시스템이 드물다.

<figure>
<img src="img/multimodal-deep-research-reports/mmb_results.png" alt="MMDR-Bench leaderboard bar plot of representative tool-using LMMs and Deep Research systems ranked by final score">
<figcaption><strong>Figure 10</strong> — MMDR-Bench 종합 점수(0–100) 리더보드. 대표 tool-using LMM과 딥리서치 시스템을 점수순으로 정렬. 출처: MMDeepResearch-Bench (arXiv:2601.12346), Fig 1.</figcaption>
</figure>

<div class="callout"><strong>이 논문의 위치 —</strong> Part 2–4가 "더 잘 만드는" 시스템이었다면, 여기는 "제대로 만들었는지 재는" 자. VEF처럼 시각 근거를 hard 제약으로 채점하는 게 핵심 기여다.</div>

<div class="pullquote">
여기까지는 모두 <strong>모든 독자에게 동일한</strong> 멀티모달 보고서를 더 잘 만들고 검증하는 이야기다. 그런데 사람마다 잘 이해하는 시각화가 다르다면?
</div>

## Part 6 — 사람마다 다른 시각화: 개인화

같은 데이터라도 초심자에게는 요약된 한 장이, 전문가에게는 세부 차트 여럿이 맞다. 같은 정보라도 누군가는 표를, 누군가는 그래프를 잘 읽는다. 이 **시각화의 개인화**를 다루는 두 연구를 본다 — 다만 둘 다 딥리서치가 아니라 인접한 HCI·교육 영역이라는 점이 중요하다.

### Drillboards — expertise에 맞춰 펼쳐지는 대시보드

**문제의식.** 대시보드는 <span class="q">"designed for a specific audience and purpose, making them essentially immutable"</span> — 특정 독자용으로 고정돼 있다. 같은 데이터라도 독자의 expertise·관심·들일 시간이 다른데, 모두에게 동일한 뷰를 던진다. 이 논문은 네 가지 설계 목표를 세운다.

<div class="callout">
<strong>설계 목표 (DG1–DG4)</strong>
<ul style="margin:.5em 0 0;padding-left:1.2em;">
<li><strong>DG1</strong> — 하나의 대시보드가 다양한 expertise 수준의 독자에게 동일한 콘텐츠를 (다른 추상도로) 제공할 것</li>
<li><strong>DG2</strong> — 같은 대시보드로 다양한 task를 지원할 것 (detail 수준·시간·집중도가 다른 task)</li>
<li><strong>DG3</strong> — 독자가 시간이 지나며 익숙해지거나 잊어버려도, 같은 대시보드를 계속 쓸 수 있을 것</li>
<li><strong>DG4</strong> — 이런 적응형 대시보드를 만드는 저작이 비교적 쉬울 것</li>
</ul>
</div>

<figure>
<img src="img/multimodal-deep-research-reports/drill_fig1.png" alt="Drillboards teaser: authoring mode (A) shows hierarchy browser and merge operations; reading mode (B) shows drill-down expanding a chart into 3 children, and roll-up collapsing them back">
<figcaption><strong>Figure 11</strong> — Drillboards의 두 모드. <strong>(A) 저작 모드</strong>: ①차트를 선택하고, ④병합 연산(Summarize·Archetype 등)을 고른 뒤 제목·설명을 추가하면 ⑤차트가 병합된다. 왼쪽 treeview(②③)가 계층 변화를 실시간 반영. <strong>(B) 독자 모드</strong>: ①원하는 차트를 좌클릭하면 ⑥원본이 사라지고 3개의 자식 차트가 강조되어 등장. 자식 차트를 우클릭하면 ⑤다시 부모 pile로 roll-up. treeview(①→③)가 탐색 경로를 동기화한다. 출처: Drillboards (arXiv:2410.12744), Fig 1.</figcaption>
</figure>

**핵심 제안 — 공식 모델.** **drillboards** — 차트들을 계층(aggregation hierarchy)으로 쌓아, 독자가 drill-down / roll-up으로 원하는 detail 수준을 탐색하는 적응형 대시보드.

구조가 명확하다. 대시보드를 데이터 + chart atom들의 집합으로 보고, drillboard는 여기에 **aggregation hierarchy**를 추가한다. 계층의 기본 단위는 두 가지다.

- **Chart atom** — 더 쪼갤 수 없는 단일 차트. 잎 노드.
- **Pile** — 하나 이상의 atom 또는 하위 pile을 담는 재귀 컨테이너. 자신의 시각 표현을 가진다.

계층의 루트는 단일 pile(전체 데이터를 한 장에 요약), 바닥은 모든 chart atom이 펼쳐진 full-detail view다. 사용자가 보는 **현재 뷰**는 현재 펼쳐진 pile·atom의 시퀀스이고, drill-down(pile → 자식들)과 roll-up(자식 그룹 → 부모 pile)으로 이동한다. 저자는 "novice / intermediate / expert" 같은 **pre-defined view**를 계층 중간에 표시해둘 수 있다.

**6가지 병합 연산 (Aggregation Operations).** 계층 구축은 임의 그룹핑이 아니라, 형식 어휘에서 정의된 6가지 Merge 연산을 반복 적용하는 것이다. 각 연산은 둘 이상의 차트를 단일 pile로 합치되 새 pile의 시각 표현을 결정한다.

| 연산 | 기호 | 무슨 일 | 정보 손실 |
|---|---|---|---|
| **Label** | 🏁 | 자식들을 단일 텍스트 레이블로 대표 (평균·최솟값 등 scalar) | 높음 (시각화 없음) |
| **Summarization** | 🔢 | 같은 유형 차트를 데이터 추상(평균·합·차)으로 하나로 합침 | 중간 |
| **Archetype** | ⭐ | 자식 중 하나를 대표자로 선택해 나머지를 숨김 | 높음 (대표 1개만 남음) |
| **Projection** | ⬇ | 두 차트의 데이터 차원을 scatterplot 축으로 투영 | 중간 |
| **Juxtaposition** | ➕ | 자식 차트들을 small multiples로 pile 안에 나란히 배치 | 낮음 (정보 보존, 해상도↓) |
| **Overlay** | 🗂 | 같은 축을 공유하는 차트들을 겹쳐 그림 | 낮음 (같은 유형 차트만 가능) |

연산마다 제약이 있다. Summarization은 두 차트의 y축 단위가 같아야 하고, Overlay는 같은 차트 유형만 합칠 수 있다. DrillVis는 적용 불가한 연산을 자동으로 grayed-out해 저자가 헷갈리지 않게 한다.

**저작 모드 (Author Mode).** DrillVis에서 저자는 두 단계로 drillboard를 만든다.

1. **차트 생성** — CSV/TSV를 올리면 연속형 데이터는 line/histogram, 이산형은 bar chart로 자동 배치된다. 멀티레벨 드롭다운으로 원하는 feature·조건을 골라 뷰에 추가한다.
2. **계층 구축** — 합치려는 차트들을 선택하고, 연산 패널에서 가능한 Merge 연산을 골라 적용한다. 왼쪽 treeview가 계층 구조를 실시간으로 반영한다. 만족하면 현재 상태를 named view(예: "novice")로 저장한다. 계층 최상단(루트 pile)과 최하단(모든 atom)은 자동으로 pre-defined view가 된다.

<figure>
<img src="img/multimodal-deep-research-reports/drill_novice.png" alt="Drillboard novice view: aggregated, grouped charts at a higher level of abstraction">
<figcaption><strong>Figure 12</strong> — 같은 데이터의 novice view. 차트가 상위 카테고리로 묶여(aggregated) 적은 수의 추상화된 뷰로 제시된다. 출처: Drillboards (arXiv:2410.12744).</figcaption>
</figure>

**독자 모드 (Reader Mode).** 독자는 자신의 expertise level에 맞는 pre-defined view에서 시작한다. 이후 두 가지 인터랙션으로 계층을 탐색한다.

- **Drill-down** — pile(카드가 겹쳐 보이는 시각 단서)을 좌클릭하면 해당 pile이 사라지고 자식 차트들이 강조(highlighted)되어 등장한다. 각 depth마다 색·불투명도로 구분되고, treeview가 동기화된다.
- **Roll-up** — 탐색 중인 차트를 우클릭하면 자식 그룹 전체가 부모 pile로 다시 접힌다. treeview도 함께 수축한다.

<figure>
<img src="img/multimodal-deep-research-reports/drill_expert.png" alt="Drillboard expert view: many detailed unit charts at full granularity">
<figcaption><strong>Figure 13</strong> — 같은 데이터의 expert view. 모든 unit chart가 세부 단위까지 펼쳐진다. 출처: Drillboards (arXiv:2410.12744).</figcaption>
</figure>

**결과.** 전문가 3명이 농업 데이터셋에 대해 novice용 drillboard를 저작하고 일반 사용자 10명이 평가했다. 데이터의 맥락과 출처를 전달하는 communication tool로 효과적이었고, novice가 전문가 의도를 빠르게 파악했다. 특히 데이터의 provenance — "왜 이 차트들이 묶였는가"를 전달하는 데 강점을 보였다.

**한계.** 저작이 일반 대시보드보다 훨씬 복잡하다. 개인화 단위가 "expertise 레벨" 하나에 머물고(직종·관심사별 figure 선택까지는 아님), 현재 구현은 단순 tabular 데이터와 line/bar/scatter에 한정된다. 딥리서치 파이프라인과는 별개 도구다.

### StoryLensEdu — 학습자에 맞춘 narrative 리포트

**문제의식.** 학습 분석 대시보드와 텍스트 리포트는 해석이 어렵고(<span class="q">"poor interpretability"</span>), 단조롭고, 일방향이다. 학생마다 데이터가 다른데 같은 형식으로 던져진다.

**핵심 제안.** 두 모듈로 나뉜다 — **report generation engine**과 **interaction module**. 엔진은 세 에이전트가 릴레이한다. **Data Analyst**가 learning-objective 중심 구조로 학생 데이터에서 인사이트를 뽑고(데이터 검색→전처리→insight 추출→시각화), **Teacher**가 그 인사이트의 교육적 가치를 평가하며 맞춤 제안을 더하고, **Storyteller**가 **Hero's Journey** 서사 틀에 얹어 학습자의 여정을 따라가는 이야기로 조직한다. 생성 후 학생이 리포트의 시각·텍스트 요소를 직접 골라 후속 질문하는 interaction module이 일방향성을 깬다. 즉 단순 차트 나열이 아니라, **데이터 인사이트 ↔ 서사**를 동적으로 결합한 context-aware 시각화 추천이다.

<figure>
<img src="img/multimodal-deep-research-reports/story_workflow.png" alt="StoryLensEdu workflow: Data Analyst, Teacher, Storyteller agents produce a personalized learning report plus interaction module for follow-up QA">
<figcaption><strong>Figure 13</strong> — StoryLensEdu 워크플로. 학습자 데이터 + learning-objective graph를 Data Analyst·Teacher·Storyteller가 가공해 개인화 리포트를 만들고, Interaction Module로 요소를 골라 후속 질의응답까지. 출처: StoryLensEdu (arXiv:2602.17067), Fig 1.</figcaption>
</figure>

**결과.** 실제 고교 현장 formative study로 설계하고 실사용자 평가에서 engagement와 학습 과정 이해가 향상됐다. 핵심 기여는 데이터 인사이트와 narrative storytelling을 동적으로 결합해 **non-expert를 위한 context-aware 시각화 추천**을 자동화했다는 점.

**한계.** 교육 도메인 특화이고, 개인화가 narrative·설명 수준에 집중된다. 역시 딥리서치 보고서 생성은 아니다.

<div class="callout"><strong>이 두 논문의 위치 —</strong> "독자마다 다른 시각"이 가능함을 보여주지만, 둘 다 <strong>주어진 데이터 한 벌</strong>을 다루는 HCI/교육 도구다. 웹을 뒤져 보고서를 만드는 딥리서치 안으로는 아직 들어오지 않았다.</div>

## Part 7 — 열린 질문: 멀티모달 × 개인화의 빈칸

정리하면 두 흐름이 평행선을 달린다.

| 흐름 | 한 일 | 빠진 것 |
|---|---|---|
| **멀티모달 딥리서치**<br>(MMDR · Deep-Reporter · TVIR · Ptah · MMDR-Bench) | 보고서에 이미지를 생성·검색·검증·평가 | 이미지가 **모든 독자에게 동일** |
| **시각화 개인화**<br>(Drillboards · StoryLensEdu) | 독자 expertise·배경에 맞춰 시각을 적응 | **주어진 데이터 한 벌**만, 딥리서치 아님 |

두 원이 겹치는 가운데 — **"독자의 배경·수준·취향에 맞춰, 딥리서치 보고서의 figure 자체를 다르게 검색·생성·배치하는"** 연구는 아직 비어 있다. 의대생과 환자가 같은 "당뇨 치료 동향"을 물어도, 텍스트뿐 아니라 **그림도 달라야** 한다. 의대생에게는 기전 다이어그램과 임상시험 forest plot을, 환자에게는 생활 습관 인포그래픽을.

### 질문을 세 겹으로 쪼개면

시각화 개인화를 딥리서치 안에서 실현하려면 세 층위의 선택이 동시에 달라져야 한다.

1. **어떤 형식으로** — 같은 사실이라도 표, 막대차트, 도식, 사진 중 무엇이 이 독자에게 더 잘 통하는가.
2. **어떤 figure를** — 웹·데이터베이스에서 어떤 specific 이미지를 골라 넣을 것인가.
3. **어떻게 재구성할지** — 선택한 이미지나 차트를 독자 수준에 맞게 스타일·추상도·설명 밀도를 바꿀 수 있는가.

현재 시스템은 이 세 질문을 독자와 무관하게 고정 처리한다. Ptah의 Visual Working Memory도, TVIR의 Image Searcher도 주어진 보고서 맥락에는 최적화되어 있지만 "이 독자가 어떤 시각화를 더 잘 이해하는가"는 묻지 않는다.

### Figure별 처리 정책 — baseline과 개인화 레이어

지금 논문들이 그어놓은 핵심 경계선은 **illustration(설명용) vs evidence(근거용)** 구분이다. 이 구분이 개인화 설계의 출발점이 된다.

<div class="callout">
<strong>Figure 유형별 처리 원칙</strong>

<ul style="margin:.5em 0 0;padding-left:1.2em;">
<li><strong>Evidence(근거용)</strong> — 수치·측정·결과를 담는 차트·그래프. 데이터는 손대면 안 된다. 코드 렌더링(차트 재생성)이나 실제 이미지 검색(출처 URL 보존)이 baseline. 스타일(색·레이블·폰트)만 독자별로 바꿀 수 있고, 수치 자체를 바꾸는 순간 evidence가 깨진다.</li>
<li><strong>Illustration(설명용)</strong> — 개념·구조·흐름을 보여주는 도식·다이어그램·삽화. 정보를 정확히 인코딩하면 스타일·복잡도·추상도를 독자에 맞춰 바꿔도 된다. 이미지 생성 모델을 쓸 수 있는 자리다.</li>
</ul>
</div>

둘을 뒤섞으면 위험하다. Evidence에 생성 모델을 쓰면 수치를 hallucinate할 가능성이 있고, Illustration에 지나치게 사실 충실성을 요구하면 개인화 여지가 없어진다. TVIR의 Image Searcher(출처 URL 보존)와 Chart Generator(코드 렌더링 + 데이터 출처)의 역할 분리가 바로 이 경계를 지키는 방식이다.

### 시각화 재구성의 스펙트럼

개인화를 얼마나 깊이 파고드느냐에 따라 가능한 개입이 스펙트럼으로 늘어선다.

| 개입 수준 | 무엇을 바꾸나 | evidence 가능 | illustration 가능 |
|---|---|---|---|
| **Restyle** | 색·폰트·여백·레이블 언어 | O | O |
| **Re-encode** | 같은 데이터를 다른 차트 유형으로 | O (코드 재생성) | O |
| **Re-aggregate** | granularity 조정 (Drillboards의 drill-down) | O (원본 데이터 있을 때) | — |
| **Re-author** | 개념을 다른 맥락·비유·시각 스타일로 새로 그리기 | X (수치 변형 위험) | O |

Re-author가 개인화의 가장 강력한 형태지만, evidence에는 쓸 수 없다. illustration에서만, 그리고 원본 개념을 정확히 보존한다는 조건 하에만 가능하다. FDV(Formal Description of Visualization)의 Data 필드가 "변경 불가 핵심", Marks/Overall layout 필드가 "재구성 가능 외피"를 가르는 기준이 될 수 있다.

### 독자 시각 프로파일 — 무엇을 파악해야 하나

텍스트 개인화(PersonaPlex의 explicit/implicit 선호, O-Mem의 주제별 기억)와 다르게, 시각 개인화에서 파악해야 할 독자 변수는 별도로 정의되어야 한다.

- **Modality ratio** — 이 독자가 글로 된 설명과 그림 중 어느 쪽을 먼저 이해하는가.
- **Form preference** — 표 vs 그래프 vs 도식 등 선호하는 시각화 형식.
- **Register** — 전문적 figure(임상 논문 수준) vs 대중적 figure(인포그래픽 수준).
- **Domain literacy** — 이 분야 특화 도식(회로도, 악보, 해부도)을 읽을 수 있는가.

이 네 변수의 조합이 "같은 evidence figure라도 어떤 형식으로 re-encode할지"와 "illustration을 어떤 스타일로 re-author할지"를 결정한다.

### 왜 개인화된 이미지가 실제로 도움이 되는가 — 근거

직관적으로 맞아 보이지만 근거를 따져보면 층위가 나뉜다.

**강한 근거 — 인지 부하와 expertise reversal effect.** Mayer의 multimedia principle은 잘 설계된 그림+글 조합이 글 단독보다 이해를 높인다는 것을 보여준다. 더 중요한 것은 **expertise reversal effect** — 초심자에게는 자세한 설명 도식이 인지 부하를 줄여 이해를 돕지만, 전문가에게는 오히려 방해가 된다(이미 아는 내용이 추가 처리 부담이 된다). 같은 이미지로 모든 독자를 만족시키는 것은 이 효과가 방향이 반대이기 때문에 구조적으로 불가능하다. Drillboards의 novice/expert view 분리가 정확히 이것을 겨냥한다.

**중간 근거 — 스타일 개인화와 engagement.** 귀여운 스타일, 친숙한 비유, narrative 포장이 이해 자체를 높인다는 증거는 약하다. 하지만 **engagement와 만족도**를 높인다는 증거는 꽤 있다. StoryLensEdu가 평가 지표로 comprehension보다 engagement를 앞세운 것이 이 층위다. 즉 스타일 개인화는 "더 잘 이해시킨다"가 아니라 "더 오래 들여다보게 만든다"로 정당화된다.

**주의할 근거 — 학습 스타일 이론.** "시각형/청각형/운동형 학습자"라는 학습 스타일 이론은 **반복 실험에서 재현되지 않아 현재 심리학계에서 부정된 가설**이다. 시각 개인화의 정당화로 학습 스타일을 끌어들이면 근거가 허물어진다.

### Pedagogical agent — 개인화의 극단

illustration re-author의 가장 강한 형태로 **pedagogical agent** — 캐릭터가 설명을 주도하는 방식을 상상해볼 수 있다. 독자가 애니 스타일의 마스코트를 좋아한다면, 딥리서치 보고서의 개념 도식이 그 캐릭터와 함께 표현되는 식이다.

이것이 동작하려면 두 조건이 필요하다. ① **Character consistency** — 멀티턴 보고서 전체에서 같은 캐릭터를 유지해야 한다. 이미지 생성 모델의 reference image + fixed seed 조합으로 가능하다. ② **Germane load** — 캐릭터가 개념 설명의 일부로 동작해야 한다(germane). 단순히 옆에 서 있기만 하면 장식(decorative)이 되어 오히려 인지 분산을 일으킨다.

이 구분은 단순해 보이지만 실제 구현에서 중요하다 — "캐릭터가 도식의 화살표를 가리키며 설명"은 germane, "보고서 상단에 캐릭터가 웃으며 서 있음"은 decorative. Mayer의 coherence principle이 decorative를 제거하라고 말하는 이유다.

### 풀어야 할 것들

두 흐름을 딥리서치 안에서 잇기 위해 구체적으로 남은 과제를 꼽으면:

- **Per-figure routing policy** — 보고서의 각 figure slot에 대해 "검색할지 / 코드로 그릴지 / 개인화 재구성할지"를 독자 프로파일과 figure 유형(evidence/illustration)에 따라 자동으로 결정하는 모듈. Ptah의 Writing 단계에 네 번째 연산(Image Personalize)으로 끼워 넣는 것이 자연스러운 확장점이다.
- **Faithfulness re-verification** — 재구성 후에도 원래 주장이 정확히 인코딩되어 있는지를 다시 확인하는 단계. VEF(Visual Evidence Fidelity)처럼 hard threshold를 두되, 재구성 전·후 양쪽을 비교해야 한다.
- **Per-user evaluation** — VEF·MOSAIC은 ground-truth 기반이라 모든 독자에게 동일한 기준을 쓴다. 개인화된 figure의 효과를 재는 잣대는 reference-free per-user scoring이어야 한다. 이해도(comprehension)와 만족도(preference)를 분리해서 재는 것도 중요하다.

> 멀티모달 보고서 생성은 여기까지 왔고, 시각화 개인화도 인접 분야에서 무르익었다. 둘을 딥리서치 파이프라인 안에서 잇는 자리가 — 지금 비어 있다.
