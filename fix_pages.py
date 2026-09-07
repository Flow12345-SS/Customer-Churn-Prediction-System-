import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove what-if
what_if = re.compile(r'# ==========================================\n# 5\. WHAT-IF LAB\n# ==========================================\nelif page == "🧪 What-If Lab":.*?# ==========================================\n# 6\. ANALYTICS', re.DOTALL)
content = what_if.sub('# ==========================================\n# 6. ANALYTICS', content)

# Remove insights
insights = re.compile(r'# ==========================================\n# 7\. INSIGHTS\n# ==========================================\nelif page == "💡 Insights":.*', re.DOTALL)
content = insights.sub('', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("done")
