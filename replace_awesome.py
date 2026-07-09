import os

filepath = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('https://github.com/sindresorhus/awesome', 'https://github.com/ishandutta2007/Awesome-Awesome-Awesome')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
