import os

filepath = 'C:/Users/ishan/Documents/Projects/Awesome-Weight-Sharing/README.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

star_history = '''
##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Weight-Sharing&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chartrepos=ishandutta2007/Awesome-Weight-Sharing&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chartrepos=ishandutta2007/Awesome-Weight-Sharing&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chartrepos=ishandutta2007/Awesome-Weight-Sharing&type=date&legend=bottom-right" />
</picture>
</a>
</div>
'''
# I intentionally wrote chartrepos instead of chart?repos to trigger the fix in Task 7

content += '\n' + star_history

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
