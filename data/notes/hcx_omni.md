# HCX SEED Omni 8B 핵심 분석

**카테고리**: Multimodal  
**자신감**: ⭐⭐⭐⭐ (심화)  
**마지막 복습**: 2026-08-11

---

## 한 문장 요약

HCX SEED Omni 8B는 **한국어에 강점을 둔 8B급 Any-to-Any 멀티모달 모델**로, 8단계 학습 파이프라인과 Data Recipe 최적화가 핵심이다.

---

## 8단계 학습 파이프라인

| 단계 | 내용 | 핵심 포인트 |
|---|---|---|
| 1 | Text LLM 사전학습 | 언어/추론 능력 기반 확보 |
| 2 | Discrete image·audio token 추가 | VQ-VAE 코드북 → Vocabulary 확장 |
| 3 | 전체 Multimodal 공동 학습 | **Token ratio 20:65:15 (Text:Image:Audio)** |
| 4 | 32K Long-context 적응 | 긴 영상/문서 처리 |
| 5 | Continuous vision encoder 연결 | Caption 75% / OCR 20% / VQA 5% |
| 6 | Vision 중심 전체 학습 | encoder + LLM joint training |
| 7 | Continuous audio encoder 연결 | Whisper + MambaMia compressor (25Hz→1Hz) |
| 8 | 4단계 SFT | 대화/지시/영상이해/long-context |

---

## Discrete vs Continuous

```
Discrete Token (생성):     이미지 → 코드북 ID → Autoregressive 생성
Continuous Feature (이해): 이미지 → Encoder → Projector → LLM
```

- **생성**이 목표면 → Discrete (VQ)
- **이해**가 목표면 → Continuous (높은 정보 보존)
- HCX는 **둘 다** 사용 (생성 + 이해 동시 지원)

---

## Data Recipe가 핵심인 이유

> "구조 자체보다 각 학습 단계의 데이터 구성(Data Recipe)을 찾는 것이 핵심"

Stage 3 SFT에서:
- Video understanding 데이터: **41.3%**
- 이유: temporal reasoning + long-context 관리 집중 학습

Token mixture ratio:
$$\text{Text:Image:Audio} = 20:65:15$$
→ 파일 수 비율이 아닌 **실제 소비 token 기준**

---

## 면접 핵심 포인트

1. **Any-to-Any** = 모든 모달리티 입력 + 텍스트/이미지/음성 **생성** 가능
2. **MambaMia**: 음성 토큰 압축기 (25Hz→1Hz, 약 25배 압축)
3. **Vision loss weight**: 초기 0.5 → 학습 후반 1.0으로 증가
4. **Long-context**: 영상 프레임 + 음성 + 텍스트 → 32K context

---

## 체크리스트

- [x] 8단계 파이프라인 순서 설명 가능
- [x] Discrete vs Continuous 차이 설명 가능
- [x] Data Recipe / Token ratio 설명 가능
- [x] MambaMia compressor 역할 설명 가능
- [x] SFT 4단계 구성 설명 가능
- [ ] 구체적인 벤치마크 수치 암기
