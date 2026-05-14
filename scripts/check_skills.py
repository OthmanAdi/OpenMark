import sys
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

from openmark.agent.skills import list_skills, parse_slash, autocomplete_choices

ss = list_skills()
print(f"Found {len(ss)} OpenMark skills:")
for s in ss:
    print(f"  - {s['short_name']:20}  type={s['type']:10}  {s['description'][:80]}")

print()
print("parse_slash('/fast-search hello') =>", parse_slash("/fast-search hello"))
print("parse_slash('/newsletter agents') =>", parse_slash("/newsletter agents"))
print("parse_slash('plain text') =>", parse_slash("plain text"))

print()
print("autocomplete_choices:")
for label, insert in autocomplete_choices():
    print(f"  insert={insert!r}  label={label[:80]}")
