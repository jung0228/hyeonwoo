# O-Voxel: Native and Compact Structured Latents for 3D Generation

> **CVPR 2026 Best Student Paper Award** 🏆  
> **저자**: Tsinghua University, Microsoft Research, USTC, Microsoft AI  
> **키워드**: `3D Asset Generation`, `O-Voxel (Omni-Voxel)`, `Sparse Compression VAE`, `PBR Materials`, `TRELLIS.2`

---

## 💡 핵심 아이디어

기존 3D 생성 모델(SDF, Triplane, NeRF)은 복잡한 위상(Topology)이나 열린 곡면(Open Surface), 물체 내부 구조 표현에 한계가 있었고, 재질(Material) 정보를 PBR(Physically-Based Rendering)로 복원하기 어려웠습니다.

**O-Voxel (Omni-Voxel)**은 암묵적 표면(Implicit Surface)에 의존하지 않는 **Field-Free 희소 옥셀(Sparse Voxel) 구조**를 제안하여, 고정밀 기하 구조뿐만 아니라 Albedo(색상), Metallic, Roughness, Opacity 등의 **PBR 재질 매개변수를 직접 인코딩**합니다.

---

## 🏗️ 아키텍처 및 기술적 기여

1. **O-Voxel Representation**:
   - 임의의 토폴로지(Non-manifold, 복잡한 구멍, 열린 껍질 등)를 손실 없이 표현 가능한 3D 구조화 데이터 그리드.

2. **Sparse Compression VAE**:
   - 3D O-Voxel 공간을 고도로 압축된 잠재 공간(Structured Latent Space)으로 변환하는 희소 VAE 네트워크.

3. **TRELLIS.2 백본 파운드 기반**:
   - 40억 파라미터(4B) 규모의 3D 생성 디퓨전 백본인 **TRELLIS.2**의 핵심 기반 기술로 활용되어, 텍스트/단일 이미지 입력으로부터 수 초 만에 게임/영화 프로덕션급 3D 에셋을 생성합니다.

---

## 🔗 연결 개념
- [[paper_show_o]] (Unified Generative Architecture)
- [[paper_cosmos]] (World Models & 3D Generation)
- [[rq_cross_modal_alignment]] (이종 모달리티 간 교차 정렬)
