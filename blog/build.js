#!/usr/bin/env node
'use strict';

const fs   = require('fs');
const path = require('path');
const matter = require('gray-matter');
const { marked } = require('marked');

const BLOG_DIR    = __dirname;
const MD_DIR      = path.join(BLOG_DIR, 'posts', 'md');
const POSTS_DIR   = path.join(BLOG_DIR, 'posts');
const INDEX_PATH  = path.join(BLOG_DIR, 'index.html');
const CSS_SOURCE  = path.join(POSTS_DIR, 'example.html');   // CSS is extracted from here

// Display labels for the filter bar
const ALL_TAGS = ['All', 'Multimodal', 'LLM', 'Vision', 'Video', 'Math', 'Agent', 'Speech'];

// "Vision" button links to ?tag=Computer Vision (how existing index.html is wired)
const TAG_URL_VALUE  = { Vision: 'Computer Vision' };
// data-tags attribute in index.html cards must also use "Computer Vision" for Vision
const TAG_DATA_VALUE = { Vision: 'Computer Vision' };

function tagUrl(tag) {
  if (tag === 'All') return '../index.html';
  return `../index.html?tag=${TAG_URL_VALUE[tag] || tag}`;
}

function tagDataValue(tag) {
  return TAG_DATA_VALUE[tag] || tag;
}

// ── Extra CSS (appended after extracted CSS) ─────────────────────────────────
const EXTRA_CSS = `
    /* ─── TABLE ─── */
    .post-content table {
      width: 100%; border-collapse: collapse;
      margin: 1.5rem 0;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: .88rem;
    }
    .post-content th {
      background: var(--bg2); font-weight: 700;
      text-align: left; padding: .55rem .9rem;
      border-bottom: 2px solid var(--text);
      color: var(--text); letter-spacing: .04em;
    }
    .post-content td {
      padding: .5rem .9rem;
      border-bottom: 1px solid var(--rule-lt);
      color: var(--text); vertical-align: top;
    }
    .post-content tr:last-child td { border-bottom: none; }
`;

// ── Markdown preprocessing ────────────────────────────────────────────────────
// marked does not treat <svg> as a block element, so SVGs get wrapped in <p>.
// Wrapping in <div> forces block-level treatment.
function preprocessMd(content) {
  // marked ends HTML blocks at blank lines (CommonMark spec).
  // Strip blank lines inside SVGs, then wrap in <div> to force block treatment.
  return content.replace(/<svg[\s\S]*?<\/svg>/g, m => {
    const compact = m.replace(/\n[ \t]*\n/g, '\n');
    return `<div>${compact}</div>`;
  });
}

// ── CSS extraction ───────────────────────────────────────────────────────────
function extractCss() {
  const html = fs.readFileSync(CSS_SOURCE, 'utf8');
  const m = html.match(/<style>([\s\S]*?)<\/style>/);
  if (!m) throw new Error('Could not extract CSS from example.html');
  return m[1];
}

// ── HTML building blocks ─────────────────────────────────────────────────────
function filterBarHtml(postTags) {
  return ALL_TAGS.map(tag => {
    const active = tag !== 'All' && postTags.includes(tag) ? ' active' : '';
    return `    <button class="tag-btn${active}" onclick="location.href='${tagUrl(tag)}'">${tag}</button>`;
  }).join('\n');
}

function postMetaHtml(fm) {
  const cats = (fm.tags || [])
    .map(t => `<span class="cat">${t}</span>\n    <span class="dot">·</span>`)
    .join('\n    ');
  return `${cats}
    <span>${fm.date}</span>
    <span class="dot">·</span>
    <span>${fm.readtime}</span>`;
}

function katexHtml() {
  return `  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
      onload="renderMathInElement(document.body, {
        delimiters: [
          {left:'$$',right:'$$',display:true},
          {left:'$',right:'$',display:false}
        ]
      });"></script>`;
}

// ── Full post HTML ───────────────────────────────────────────────────────────
function generateHtml(fm, body, css) {
  const tags = fm.tags || [];
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <link rel="icon" type="image/svg+xml" href="../favicon.svg">
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${fm.title} – Hyeonwoo Jung</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
${fm.katex ? katexHtml() : ''}  <style>${css}${EXTRA_CSS}  </style>
</head>
<body>

<nav>
  <a class="nav-logo" href="../../index.html">Hyeonwoo Jung</a>
  <ul class="nav-links">
    <li><a href="../../index.html">Home</a></li>
    <li><a href="../index.html" class="active">Blog</a></li>
  </ul>
</nav>

<div class="masthead">
  <div class="masthead-bar"></div>
  <div class="masthead-inner">
    <h1 class="masthead-title">Research Blog</h1>
    <p class="masthead-sub">Paper Reviews &nbsp;·&nbsp; Ideas &nbsp;·&nbsp; Notes on AI</p>
  </div>
</div>

<div class="filter-bar">
  <div class="filter-bar-inner-wrap">
  <div class="filter-inner">
${filterBarHtml(tags)}
  </div>
  </div>
</div>

<div class="article-wrap">

  <h1 class="post-title">${fm.title}</h1>
  <p class="post-dek">${fm.dek}</p>

  <div class="post-meta">
    ${postMetaHtml(fm)}
  </div>

  <div class="post-content">
${body}
  </div>

</div>

<footer>© 2026 Hyeonwoo Jung &nbsp;·&nbsp; Research Blog</footer>

</body>
</html>`;
}

// ── index.html card injection ────────────────────────────────────────────────
function buildPostCard(fm) {
  const tags     = fm.tags || [];
  const dataTags = tags.map(tagDataValue).join(',');
  const cat      = tags[0] || '';
  const desc     = fm.desc || fm.dek;
  return `    <a class="post-card" href="posts/${fm.slug}.html" data-tags="${dataTags}">
      <div class="card-cat">${cat}</div>
      <div class="card-title">${fm.title}</div>
      <div class="card-desc">${desc}</div>
      <div class="card-meta">${fm.date} · ${fm.readtime}</div>
    </a>`;
}

function injectCard(fm) {
  let html = fs.readFileSync(INDEX_PATH, 'utf8');
  if (html.includes(`posts/${fm.slug}.html`)) {
    console.log(`  index.html: card for "${fm.slug}" already exists, skipping`);
    return;
  }
  const marker = 'id="post-grid">';
  const idx = html.indexOf(marker);
  if (idx === -1) {
    console.warn('  WARNING: could not find #post-grid in index.html — add card manually');
    console.log('\nCard HTML:\n' + buildPostCard(fm));
    return;
  }
  const insertAt = idx + marker.length;
  html = html.slice(0, insertAt) + '\n' + buildPostCard(fm) + '\n' + html.slice(insertAt);
  fs.writeFileSync(INDEX_PATH, html);
  console.log(`  index.html: injected card for "${fm.slug}"`);
}

// ── Build one file ───────────────────────────────────────────────────────────
function buildFile(mdPath, css) {
  const raw = fs.readFileSync(mdPath, 'utf8');
  const { data: fm, content } = matter(raw);

  if (!fm.slug)     { console.error(`ERROR: missing slug in ${mdPath}`); return; }
  if (!fm.title)    { console.error(`ERROR: missing title in ${mdPath}`); return; }
  if (!fm.date)     { console.error(`ERROR: missing date in ${mdPath}`); return; }
  if (!fm.readtime) { console.error(`ERROR: missing readtime in ${mdPath}`); return; }

  const body    = marked(preprocessMd(content));
  const html    = generateHtml(fm, body, css);
  const outPath = path.join(POSTS_DIR, fm.slug + '.html');

  fs.writeFileSync(outPath, html);
  console.log(`Built: ${outPath}`);
  injectCard(fm);
}

// ── Main ─────────────────────────────────────────────────────────────────────
if (!fs.existsSync(MD_DIR)) fs.mkdirSync(MD_DIR, { recursive: true });

const css  = extractCss();
const args = process.argv.slice(2);

const targets = args.length > 0
  ? args.map(a => path.resolve(a))
  : fs.readdirSync(MD_DIR)
      .filter(f => f.endsWith('.md') && !f.startsWith('_'))
      .map(f => path.join(MD_DIR, f));

if (targets.length === 0) {
  console.log('No .md files found in posts/md/');
  console.log('Usage: node build.js [file.md ...]');
  process.exit(0);
}

targets.forEach(f => buildFile(f, css));
console.log('Done.');
