# 📄 [Paper] MeViS: A Multi-Modal Dataset for Referring Motion Expression Video Segmentation
- **Authors**: Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Chen Change Loy (NTU Singapore)
- **Venue / Year**: ICCV 2023 / CVPR PVUW Workshop Challenge Benchmark (2024-2026)
- **Domain**: Video Understanding / Motion-Guided Segmentation / Benchmark
- **Connected**: [[paper_virst]], [[paper_lisa]], [[long_video_understanding]], [[paper_momentseeker]], [[rq_video_temporal_grounding]]

---

## 1. Problem Formulation & Frontier Blind Spot (문제 정의)
- **Unaddressed Bottleneck**: 기존의 비디오 객체 분할 벤치마크(Ref-YouTube-VOS, A2D-Sentences)는 객체의 정적 외형(Static Appearance, e.g. "노란 셔츠를 입은 소년")에만 의존해도 쉽게 정답을 맞출 수 있는 한계가 있었음.
- **Core Limitation of Prior Art**: 외형이 완전히 동일한 여러 객체 중 **"갑자기 왼쪽으로 빠르게 방향을 틀어 달리는 얼룩말"**처럼 **오직 동작과 움직임(Motion Expression)**을 통해서만 식별 가능한 실제 물리적 시간 추론 능력을 전혀 평가하지 못함.

---

## 2. Core Benchmark Characteristics (데이터셋 특징)
- **규모 및 복잡도**:
  - 2,000개 이상의 복잡 비디오, 28,000개 이상의 모션 기반 자연어 표현 문장, 430,000개 이상의 프레임별 픽셀 정답 마스크.
  - 정적 외형만으로는 절대 분간할 수 없는 군집 내 다중 동일 객체 환경 포함.
- **평가 지표**:
  - $\mathcal{J}$ (Region Similarity / IoU) & $\mathcal{F}$ (Contour Accuracy) 및 통합 점수 $\mathcal{J}\&\mathcal{F}$.

---

## 3. Findings & Lineage Impact
- 기존 SOTA 비디오 모델들이 정적 외형 단서가 제거된 모션 쿼리 앞에서 성능이 40% 이상 급락하는 취약점 규명.
- **VIRST 및 AIDAS 랩의 성과**:
  - CVPR 2026 **PVUW MeViS-Audio Challenge**에서 서울대 AIDAS 랩(도재영 교수님 연구실)이 시각 모션뿐만 아니라 오디오 사운드 단서를 결합하여 **세계 3위**를 수상.
  - VIRST 논문이 MeViS 벤치마크의 Motion-guided Segmentation 부문에서 SOTA를 달성하는 핵심 평가 기반이 됨.
