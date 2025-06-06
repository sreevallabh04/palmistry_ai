import codecs
import sys

try:
    with codecs.open('requirements.txt', 'r', encoding='utf-16') as f:
        content = f.read()
    sys.stdout.write(content)
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    sys.exit(1)