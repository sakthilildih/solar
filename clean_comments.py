import re

with open('pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned = []
skip_next = False
in_docstring = False
docstring_delimiter = None

for i, line in enumerate(lines):
    original = line
    stripped = line.strip()
    
    if '"""' in line:
        if not in_docstring:
            in_docstring = True
            docstring_delimiter = '"""'
            if line.count('"""') >= 2:
                in_docstring = False
            continue
        else:
            in_docstring = False
            continue
    
    if "'''" in line:
        if not in_docstring:
            in_docstring = True
            docstring_delimiter = "'''"
            if line.count("'''") >= 2:
                in_docstring = False
            continue
        else:
            in_docstring = False
            continue
    
    if in_docstring:
        continue
    
    if stripped.startswith('#'):
        continue
    
    if '#' in line and '"' not in line.split('#')[0] and "'" not in line.split('#')[0]:
        code_part = line.split('#')[0].rstrip()
        if code_part.strip():
            cleaned.append(code_part + '\n')
        elif not stripped:
            cleaned.append(line)
    else:
        cleaned.append(line)

with open('pipeline.py', 'w', encoding='utf-8') as f:
    f.writelines(cleaned)

print(f'✅ Removed all comments. File reduced from {len(lines)} to {len(cleaned)} lines.')
