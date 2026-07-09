import re

filepath = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

badge_right = '<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>\n'

content = content.replace('alt="Discord" /></a>\n</div>', 'alt="Discord" /></a>' + badge_right + '</div>')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
