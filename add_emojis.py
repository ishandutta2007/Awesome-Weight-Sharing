import os
import re

filepath = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add banner
content = re.sub(r'# Awesome-Weight-Sharing', '# 🌟 Awesome-Weight-Sharing\n\n<div align="center">\n  <img src="assets/banner.svg" alt="Banner" width="100%" />\n</div>\n', content)

# Add emojis
replacements = {
    '## Weight Sharing in AI': '## 🚀 Weight Sharing in AI',
    '## 1. The Macro Chronological Evolution': '## 🕰️ 1. The Macro Chronological Evolution',
    '## 2. Core Functional & Algorithmic Variants': '## ⚙️ 2. Core Functional & Algorithmic Variants',
    '## 3. The Distributed FSDP Weight-Sharing Pipeline': '## 🌐 3. The Distributed FSDP Weight-Sharing Pipeline',
    '## 4. Production Engineering Challenges & Cluster Solutions': '## 🛠️ 4. Production Engineering Challenges & Cluster Solutions',
    '## 5. Frontier Real-World AI Industrial Applications': '## 🏭 5. Frontier Real-World AI Industrial Applications',
    '## References': '## 📚 References'
}

for k, v in replacements.items():
    content = content.replace(k, v)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
