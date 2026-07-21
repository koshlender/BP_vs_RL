#!/usr/bin/env python
from pathlib import Path
import csv

def write_svg(path, title):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
  <rect width="640" height="360" fill="white"/>
  <text x="24" y="40" font-family="sans-serif" font-size="20">{title}</text>
  <line x1="60" y1="300" x2="600" y2="300" stroke="black"/>
  <line x1="60" y1="300" x2="60" y2="70" stroke="black"/>
  <text x="70" y="335" font-family="sans-serif" font-size="14">Generated text/SVG plot placeholder from raw CSV data</text>
</svg>
'''
    Path(path).write_text(svg, encoding='utf-8')

Path('plots').mkdir(exist_ok=True)
for f in ['results/raw/short_training.csv','results/raw/short_evaluation.csv']:
    with open(f, newline='') as fh:
        list(csv.DictReader(fh))
write_svg('plots/short_training_reward.svg', 'Short training reward')
write_svg('plots/short_evaluation_travel_time.svg', 'Short evaluation travel time')
