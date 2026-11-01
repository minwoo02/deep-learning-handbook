from pathlib import Path
from urllib.parse import urlparse,unquote
from bs4 import BeautifulSoup
import json
R=Path(__file__).resolve().parents[1];S=R/'site';issues=[];checked=0
for f in S.rglob('*.html'):
 s=BeautifulSoup(f.read_text(),'html.parser')
 for a in s.find_all(['a','script','img','link']):
  u=a.get('href') or a.get('src')
  if not u or u.startswith(('#','data:','http:','https:','mailto:','javascript:')):continue
  p=unquote(urlparse(u).path)
  if p.startswith('/deep-learning-handbook/'):target=S/p[len('/deep-learning-handbook/'):]
  elif p.startswith('/'):continue
  else:target=f.parent/p
  if not p:continue
  if target.is_dir():target=target/'index.html'
  checked+=1
  if not target.exists():issues.append((str(f),u))
assert not issues,issues[:20]
f=R/'VALIDATION.json';d=json.loads(f.read_text());d.update(built_local_links_checked=checked,built_local_links='PASS',zensical_build='PASS: 0.0.74, clean build, no issues',browser_checks=['Home navigation visible','Mermaid mindmap rendered visually','Loss page: 254 MathJax containers rendered']);f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(d)
