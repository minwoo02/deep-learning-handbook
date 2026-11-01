"""Check page-title hierarchy, article outlines, and standalone Handbook text."""
from pathlib import Path
from urllib.parse import unquote
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1]
pages=0;outlines=0
for md in (R/'docs').rglob('*.md'):
 rel=md.relative_to(R/'docs')
 html=R/'site'/rel.with_suffix('')/'index.html' if md.stem!='index' else R/'site'/rel.parent/'index.html'
 soup=BeautifulSoup(html.read_text(),'html.parser');article=soup.select_one('article')
 assert article is not None,rel
 assert len(article.select('h1'))==1,rel
 assert not any(x in article.get_text() for x in ['원문:', '원문 보존', '이관본', '원본 목록·검증']),rel
 assert not any('velog.io' in a.get('href','') for a in article.select('a')),rel
 if '\n[TOC]\n' in md.read_text():
  anchors=article.select('.toc a[href^="#"]')
  assert anchors,rel
  for a in anchors:assert soup.find(id=unquote(a['href'][1:])),(rel,a['href'])
  assert soup.select_one('.md-nav--secondary'),rel
  outlines+=1
 pages+=1
print(f'PASS: {pages} standalone pages, one H1 each; {outlines} article outlines with valid anchors')
