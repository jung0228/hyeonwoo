# Paper Analysis Post Method

논문을 블로그 포스트로 정리할 때 이 저장소에서 재사용하는 작업 가이드.

원본 메모 위치:
`/Users/jhw/.claude/projects/-Users-jhw-Desktop-web-hyeonwoo/memory/paper-analysis-method.md`

## 핵심 철학

논문을 읽는다는 것은 저자가 생각한 **흐름**을 파악하는 것에 가깝다.  
글은 블록 단위로 흐름이 전환되고, Figure는 그 흐름을 돕는 시각 자료다.

- 섹션마다 세부 정보보다 먼저 흐름을 파악한다.
- 블록 단위로 요약한 뒤, 영어 원문 인용과 Figure를 연결한다.
- Figure는 장식이 아니라 해당 블록의 설명을 밀어주는 증거로 배치한다.

## 포스트 구조 패턴

각 섹션은 아래 패턴을 기본으로 쓴다.

```md
## 섹션명

<p><strong>흐름:</strong> 블록A → 블록B → 블록C</p>

### 블록 제목
<mark>한줄 요약 (형광팬)</mark>

*"영어 원문 인용"*

<figure>
<img src="img/<slug>/figN.jpg" alt="Figure N">
<figcaption><strong>Figure N</strong> — 이 흐름에서 이 그림이 무엇을 보여주는지 설명</figcaption>
</figure>
```

## 주요 규칙

- `흐름` 줄은 반드시 `<p><strong>흐름:</strong> A → B → C</p>` 형태로 한 줄에 쓴다.
- 요약은 `<mark>내용</mark>` 형광팬 처리만 사용하고, `[요약]` 같은 레이블은 붙이지 않는다.
- 블록 제목은 `### 제목` 형식으로 쓰고, `블록 1` 같은 번호 레이블은 붙이지 않는다.
- 영어 원문 인용은 blockquote(`>`)를 쓰지 않고 `*"..."*` 형태로 본문에 녹여 쓴다.
- 섹션 구분용 `---`는 frontmatter 외에는 쓰지 않는다.
- Figure는 논문 LaTeX의 figure label을 보고, 어느 섹션/블록 흐름에 들어가는지 먼저 매핑한 뒤 넣는다.
- Table은 마크다운 표보다 HTML `<table>`로 직접 작성한다.
- 강조 행은 `style="background:#fef9c3;font-weight:700;"`를 사용한다.
- Supplement가 필요하면 `## Supplement A/B/C...` 패턴으로 같은 방식으로 이어간다.

## CSS 메모

현재 빌드 쪽에서 전제로 둔 스타일 메모:

```css
mark { background: #fef08a; padding: 1px 2px; border-radius: 2px; }
.summary-highlight { background: #fef9c3; border-left: 3px solid #eab308; padding: .6rem 1rem; }
.table-caption { font-size: .82rem; color: var(--muted); margin-bottom: .4rem; }
```

## 작업 흐름

가장 중요한 원칙:

**그림을 먼저 다 읽고 나서 글을 쓴다. SVG를 직접 그리지 않는다.**

권장 순서:

1. 논문 LaTeX 소스(`sec/*.tex`, `sections/*.tex`, `main.tex`, `index.md` 등)를 섹션별로 직접 읽는다.
2. Figure 이미지를 먼저 블로그용으로 준비한다.
3. 모든 그림을 실제로 확인하고, 어떤 블록 설명에 붙을지 매핑한다.
4. 섹션별 흐름을 파악하고 블록 단위로 분해한다.
5. Figure label과 블록을 연결한 뒤 `blog/posts/md/<slug>.md`를 작성한다.
6. 필요하면 `cd blog && /usr/local/bin/node build.js`로 빌드한다.

## Figure 이미지 변환 규칙

- PDF Figure는 PyMuPDF(`fitz`)로 2x 렌더링한다.
- PIL로 흰 여백을 자동 크롭한 뒤 JPG로 저장한다.
- 출력 경로는 `blog/posts/img/<slug>/`를 기본으로 한다.
- PNG/JPG가 이미 있으면 가능한 한 그대로 활용한다.
- `sips`는 PDF 전체 페이지 기준으로 변환되어 여백이 많이 남을 수 있으니 기본 선택지로 쓰지 않는다.

표준 변환 스크립트:

```python
import fitz
from PIL import Image, ImageChops
import io, os

src = "논문폴더/figures"
out = "blog/posts/img/<slug>"

for fname in os.listdir(src):
    if not fname.endswith(".pdf"):
        continue
    pdf = fitz.open(os.path.join(src, fname))
    page = pdf[0]
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        pad = 20
        w, h = img.size
        bbox = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(w, bbox[2] + pad),
            min(h, bbox[3] + pad),
        )
        img = img.crop(bbox)
    img.save(os.path.join(out, fname[:-4] + ".jpg"), "JPEG", quality=95)
    pdf.close()
```

## `fig_tex` 폴더가 있을 때

Figure 매핑을 빠르게 확인할 때:

```bash
cd 논문폴더 && grep -rh "includegraphics" fig_tex/*.tex
cd 논문폴더 && grep -rh "label\\|caption" fig_tex/*.tex
```

이렇게 PDF 파일명, label, caption을 먼저 매칭해두면 글 배치가 빨라진다.

## 주의사항

- `sed -i '' '/^---$/d'` 같은 명령은 frontmatter까지 날릴 수 있으니 쓰지 않는다.
- Figure 경로는 `img/<slug>/figN.jpg`처럼 `posts/` 기준 상대경로로 쓴다.
- Supplement Figure도 같은 폴더 규칙으로 맞춘다.
- `node`는 `cd blog && /usr/local/bin/node build.js`처럼 전체 경로로 실행한다.
- `fitz`, `pillow`가 없으면 `pip3 install pymupdf pillow`로 준비한다.

## 이 저장소에서 적용할 때의 해석

- 새 논문을 포스트로 만들 때는 먼저 `papers/<논문폴더>`에서 원문과 Figure를 읽는다.
- 결과물은 `blog/posts/md/<slug>.md`에 저장한다.
- 이미지 자산은 `blog/posts/img/<slug>/`에 모은다.
- 글 스타일은 기존 포스트들처럼 "문제 제기 → 핵심 아이디어 → 방법 → 결과 → 의미" 흐름을 기본으로 하되, 위 구조 패턴을 우선한다.
