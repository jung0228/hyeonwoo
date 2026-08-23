import os, json, re

def parse_md_metadata(filepath, default_category, default_tab):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = [l.strip() for l in content.splitlines() if l.strip()]
    
    title = 'Untitled Article'
    author = '정현우'
    date = '2026-08-23'
    read_time = '10 min read'
    category = default_category

    # Extract title from H1 (# Title)
    for line in lines:
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Extract summary/excerpt
    summary = ''
    for line in lines:
        if not line.startswith('#') and not line.startswith('>') and not line.startswith('---'):
            summary = line[:180].strip() + '...'
            break
    if not summary:
        summary = title

    filename = os.path.basename(filepath)
    file_id = os.path.splitext(filename)[0]

    return {
        "id": file_id,
        "title": title,
        "date": date,
        "author": author,
        "summary": summary,
        "file": filepath.replace('\\', '/'),
        "category": category,
        "readTime": read_time,
        "tab": default_tab
    }

def sync_all_folders():
    repo_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_dir)

    all_cols = []

    # 1. Scan data/columns/ folder for ✍️ 칼럼 tab
    col_dir = 'data/columns'
    if os.path.exists(col_dir):
        for f in sorted(os.listdir(col_dir)):
            if f.endswith('.md'):
                fp = os.path.join(col_dir, f)
                all_cols.append(parse_md_metadata(fp, 'Intellectual Column', 'column'))

    # 2. Scan data/auto_research/ folder for 🤖 자동화 연구 tab
    auto_dir = 'data/auto_research'
    if os.path.exists(auto_dir):
        for f in sorted(os.listdir(auto_dir)):
            if f.endswith('.md'):
                fp = os.path.join(auto_dir, f)
                all_cols.append(parse_md_metadata(fp, 'Autonomous Research Engine', 'auto-research'))

    with open('data/columns.json', 'w', encoding='utf-8') as fp:
        json.dump({"columns": all_cols}, fp, indent=2, ensure_ascii=False)

    print(f'✅ Auto-synced {len(all_cols)} markdown files from folders to data/columns.json!')

if __name__ == '__main__':
    sync_all_folders()
