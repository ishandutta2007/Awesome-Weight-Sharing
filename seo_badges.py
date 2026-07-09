import re

filepath = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

badges_left = '<div align="center" id="badges">\n<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>\n</div>\n\n'

seo_desc = '<!-- SEO Keywords: Weight Sharing, Parameter Sharing, Weight Tying, FSDP, Artificial Intelligence, Neural Networks, Deep Learning, PyTorch, Model Compression, DeepSeek -->\n'

content = content.replace('<div align="center">\n  <img src="assets/banner.svg" alt="Banner" width="100%" />\n</div>\n', '<div align="center">\n  <img src="assets/banner.svg" alt="Banner" width="100%" />\n</div>\n\n' + badges_left + seo_desc)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
