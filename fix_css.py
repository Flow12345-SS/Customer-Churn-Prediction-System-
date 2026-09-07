import re
import textwrap

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    prefix = match.group(1)
    quote = match.group(2)
    inner = match.group(3)
    
    # Strip common leading whitespace
    dedented_inner = textwrap.dedent(inner)
    
    # We also want to remove any remaining leading spaces from the first line if it wasn't caught by dedent
    # Actually, dedent handles it perfectly as long as the first line is empty (which it is, since we do `\"\"\"\n`).
    # Just to be extra safe against Streamlit markdown codeblocks, we can literally left-strip every line:
    fully_stripped_inner = '\n'.join(line.lstrip() for line in inner.split('\n'))
    
    return f"st.markdown({prefix}{quote}{fully_stripped_inner}{quote}, unsafe_allow_html=True)"

pattern = re.compile(r'st\.markdown\(\s*(f?)(\"\"\"|\'\'\')(.*?)\2\s*,\s*unsafe_allow_html=True\s*\)', re.DOTALL)
new_content = pattern.sub(replacer, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Fixed CSS rendering bug!")
