#!/usr/bin/env python3
"""
Typography and character hygiene verifier for Lumen.
Ensures zero em dashes (\\u2014) or en dashes (\\u2013) exist across all source files,
documentation, headers, scripts, and configuration.
"""
import os
import sys

IGNORED_DIRS = {
    '.git', '.pixi', 'bin', 'build', 'dist', '__pycache__',
    '.mojo_cache', 'lib', '.idea', '.vscode'
}

IGNORED_EXTS = {
    '.pyc', '.dylib', '.so', '.o', '.a', '.png', '.jpg', '.jpeg',
    '.mojopkg', '.mojoc', '.lock', '.bin', '.tar.gz', '.zip'
}

def scan_repository() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    violations = []
    scanned_files = 0
    scanned_lines = 0

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IGNORED_EXTS:
                continue
            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, repo_root)

            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    scanned_files += 1
                    for line_idx, line in enumerate(handle, 1):
                        scanned_lines += 1
                        if '\u2014' in line or '\u2013' in line:
                            violations.append((rel_path, line_idx, line.strip()))
            except (UnicodeDecodeError, PermissionError):
                continue

    print(f"Scanned {scanned_files} files ({scanned_lines:,} lines of text)...")

    if violations:
        print(f"ERROR: Found {len(violations)} dash violation(s):")
        for file_path, line_no, content in violations:
            print(f"  {file_path}:{line_no}: {content}")
        return 1

    print("VERIFICATION PASSED: Absolute zero em/en dashes found across the entire repository!")
    return 0

if __name__ == '__main__':
    sys.exit(scan_repository())
