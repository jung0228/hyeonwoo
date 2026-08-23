# PyTorch 모델 생성부터 Traceback 자가 디버깅까지: 실패율 0% 루프 설계

> **저자**: 정현우 (AI Research Director)  
> **게시 카테고리**: Self-Debugging Engine  
> **발행일**: 2026-08-23  

---

## 1. 📌 연구 자동화의 최대 난관: 실행 오류(Runtime Crash)

연구 자동화 루프가 중단되는 가장 큰 이유는 코드 실행 중 발생하는 `CUDA Out of Memory`, `Dimension Mismatch`, `NaN Loss`와 같은 런타임 에러입니다.

본 문서에서는 에러가 발생해도 연구 시스템이 멈추지 않고 스스로 정정하여 **실패율 0%를 유지하는 자가 디버깅(Self-Debugging) 루프**를 다룹니다.

---

## 2. ⚙️ 자가 디버깅 4단계 프로토콜

1. **Traceback Capture (에러 로그 수집)**:
   - 파이썬 `traceback.format_exc()`를 파싱하여 에러 라인과 변수 텐서 Shape 추출.
2. **Causal Diagnosis (원인 분석)**:
   - 어텐션 차원 불일치인가, VRAM 용량 초과인가를 규칙 및 LLM으로 분석.
3. **Patch Generation (패치 생성)**:
   - `einops` 차원 재설정 또는 Gradient Accumulation Step 자동 추가.
4. **Retry & Verify (재실행 및 검증)**:
   - 최대 5회 자가 재시도하여 테스트 통과 확인 후 파이프라인 재개.
