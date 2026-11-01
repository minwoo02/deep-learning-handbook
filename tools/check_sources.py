"""Check source code, local images, and migrated Markdown preservation."""
from pathlib import Path
import json, hashlib
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'sources/manifest.json').read_text())
count=0
for row in manifest:
 source=json.loads((R/'sources'/f"{row['id']}.json").read_text())
 assert '[Truncated]' not in source['html'],row['id']
 assert hashlib.sha256(source['html'].encode()).hexdigest()==row['source_sha256']
 blocks=json.loads((R/'examples/fragments'/f"{row['id']}.json").read_text())
 original=BeautifulSoup(source['html'],'html.parser').find_all('pre')
 assert len(blocks)==len(original)
 document=R/'docs'/row['destination']
 if row['id'] in ('a00', 'c00'):
  document=R/'sources/legacy-guides'/f"{row['id']}.md"
 markdown=document.read_text()
 for src,block in zip(original,blocks):
  assert src.get_text()==block['text']
  assert block['text'] in markdown,(row['id'],block['index'])
  assert hashlib.sha256(block['text'].encode()).hexdigest()==block['sha256']
 count+=len(blocks)
images=json.loads((R/'sources/image-manifest.json').read_text())
for image in images:
 assert hashlib.sha256((R/'docs'/image['path']).read_bytes()).hexdigest()==image['sha256']
print(f'PASS: {len(manifest)} articles, {count} exact code/text blocks, {len(images)} local images')
