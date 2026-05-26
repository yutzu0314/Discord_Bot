import sys, os, platform, importlib, subprocess

REQUIRED = [
    ("torch", None),
    ("ultralytics", None),
    ("cv2", None),
    ("numpy", None),
    ("norfair", None),
    ("discord", "discord.py"),
    ("sqlalchemy", None),
    ("pymysql", None),
    ("dotenv", "python-dotenv"),
]

def pkg_version(modname: str):
    mod = importlib.import_module(modname)
    return getattr(mod, "__version__", "unknown")

def pip_show(pkg: str):
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "show", pkg], text=True)
        # 取第一行 Name/Version 就好
        lines = [l for l in out.splitlines() if l.startswith("Name:") or l.startswith("Version:")]
        return " | ".join(lines)
    except Exception:
        return "(pip show failed)"

def main():
    print("=== ENV VERIFY ===")
    print("sys.executable:", sys.executable)
    print("cwd:", os.getcwd())
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print()

    ok = True
    print("=== IMPORT CHECK ===")
    for modname, pipname in REQUIRED:
        try:
            ver = pkg_version(modname)
            print(f"[OK] import {modname:12s} version={ver}")
        except Exception as e:
            ok = False
            print(f"[FAIL] import {modname:12s} error={e}")
    print()

    print("=== PIP SHOW (KEY PKGS) ===")
    for modname, pipname in REQUIRED:
        pkg = pipname or modname
        if pkg in ["cv2"]:  # cv2 不是 pip 套件名
            continue
        print(f"{pkg:15s} -> {pip_show(pkg)}")
    print()

    print("=== TORCH CUDA CHECK ===")
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("device name:", torch.cuda.get_device_name(0))
            # 做一個很小的 CUDA tensor 運算，抓真正的 runtime error
            try:
                x = torch.randn(16, 16, device="cuda")
                y = x @ x
                torch.cuda.synchronize()
                print("[OK] CUDA tensor op OK")
            except Exception as e:
                print("[FAIL] CUDA tensor op failed:", repr(e))
    except Exception as e:
        ok = False
        print("[FAIL] torch cuda check error:", repr(e))

    print()
    print("=== RESULT ===")
    print("PASS ✅" if ok else "FAIL ❌ (check missing imports above)")

if __name__ == "__main__":
    main()
