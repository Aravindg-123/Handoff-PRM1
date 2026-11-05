
with open('feature_extraction.py', 'rb') as f:
    content = f.read()

old = (
    b'    """Binary: 1.0 if entry_point function name appears in handoff, else 0.0.\r\n'
    b'    Returns 0.5 (neutral) if entry_point is unknown.\r\n'
    b'    """\r\n'
    b'    if not entry_point:\r\n'
    b'        return 0.5\r\n'
    b'    return 1.0 if entry_point in handoff else 0.0\r\n'
)

new = (
    b'    """Binary: 1.0 if entry_point function name appears as a whole word in handoff.\r\n'
    b'    Returns 0.5 (neutral) if entry_point is unknown.\r\n'
    b'\r\n'
    b'    Uses word-boundary regex so that "sort" does not match inside "sort_impl".\r\n'
    b'    Fixes the substring-trap bug: "<ep>_impl" contains "<ep>" as a prefix,\r\n'
    b'    causing a plain `in` check to return 1.0 on signature_rename handoffs.\r\n'
    b'    """\r\n'
    b'    if not entry_point:\r\n'
    b'        return 0.5\r\n'
    b'    import re as _re\r\n'
    b'    pattern = r\'\\b\' + _re.escape(entry_point) + r\'\\b\'\r\n'
    b'    return 1.0 if _re.search(pattern, handoff) else 0.0\r\n'
)

print('old found:', old in content)
if old in content:
    new_content = content.replace(old, new, 1)
    with open('feature_extraction.py', 'wb') as f:
        f.write(new_content)
    print('Patched feature_extraction.py')
else:
    print('ERROR: old block not found')
    # Show what the actual block looks like
    idx = content.find(b'    return 1.0 if entry_point in handoff')
    if idx >= 0:
        print(repr(content[idx-200:idx+100]))
