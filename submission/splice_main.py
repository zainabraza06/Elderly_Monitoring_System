"""Splice _new_main_block.py into main.py, replacing from the MAIN PIPELINE separator onward."""
with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

cut = None
for i, ln in enumerate(lines):
    if '# MAIN PIPELINE' in ln:
        for j in range(i, -1, -1):
            if lines[j].strip().startswith('# ==='):
                cut = j
                break
        break

if cut is None:
    # fallback: find def main():
    for i, ln in enumerate(lines):
        if ln.strip().startswith('def main():'):
            cut = i
            break

print(f"Cut at line {cut} (0-indexed): {repr(lines[cut][:60]) if cut is not None else 'NOT FOUND'}")
print(f"Total lines: {len(lines)}")

if cut is None:
    print("ERROR: could not find cut point")
    raise SystemExit(1)

header = ''.join(lines[:cut])

with open('_new_main_block.py', 'r', encoding='utf-8') as f:
    new_block = f.read()

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(header)
    f.write(new_block)

print("main.py written successfully")

# verify
with open('main.py', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
print(f"New total lines: {len(all_lines)}")
