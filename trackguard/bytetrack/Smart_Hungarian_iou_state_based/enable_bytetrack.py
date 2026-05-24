"""
Enable ByteTrack Three-Stage Association

Modify settings.py to enable ByteTrack before running evaluation.
"""

import re
from pathlib import Path

def enable_bytetrack_in_settings():
    """Enable ByteTrack by modifying settings.py file"""

    settings_path = Path(__file__).parent / "utils" / "settings.py"

    print(f"Reading {settings_path}...")
    with open(settings_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the ByteTrack setting
    pattern = r"'use_bytetrack_association':\s*False"
    replacement = "'use_bytetrack_association': True"

    if pattern in content or re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)

        # Write back
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print("✓ ByteTrack association ENABLED in settings.py")
        print("  'use_bytetrack_association': False → True")
        print("\nYou can now run: python mot_evaluator.py")
    else:
        print("⚠ Could not find ByteTrack setting in settings.py")
        print("Please manually set use_bytetrack_association = True")

def disable_bytetrack_in_settings():
    """Disable ByteTrack by modifying settings.py file"""

    settings_path = Path(__file__).parent / "utils" / "settings.py"

    print(f"Reading {settings_path}...")
    with open(settings_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the ByteTrack setting
    pattern = r"'use_bytetrack_association':\s*True"
    replacement = "'use_bytetrack_association': False"

    if pattern in content or re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)

        # Write back
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print("✓ ByteTrack association DISABLED in settings.py")
        print("  'use_bytetrack_association': True → False")
        print("\nYou can now run: python mot_evaluator.py")
    else:
        print("⚠ Could not find ByteTrack setting in settings.py")
        print("Please manually set use_bytetrack_association = False")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "disable":
        disable_bytetrack_in_settings()
    else:
        enable_bytetrack_in_settings()
