import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('Step3_산불발생_선행기상및국지임계치_심화분석.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 23의 100~200라인 출력
cell = nb['cells'][23]
source = ''.join(cell.get('source', []))
lines = source.split('\n')
for i, line in enumerate(lines[100:200]):
    print(f"{i+101}: {line}")
