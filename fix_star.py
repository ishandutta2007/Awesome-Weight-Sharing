import os

filepath = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('chartrepos', 'chart?repos')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
