"""Debug: test if main.py imports and runs Phase 1 correctly."""
import sys, traceback, io
sys.path.insert(0, r"c:\Users\User\Documents\4rth semester\AI\SisFall_dataset\submission")

log = open(r"c:\Users\User\Documents\4rth semester\AI\SisFall_dataset\submission\debug_log.txt",
           "w", encoding="utf-8")

class Tee(io.TextIOBase):
    def write(self, s):
        log.write(s)
        log.flush()
        return len(s)

sys.stdout = Tee()
sys.stderr = Tee()

try:
    print("=== DEBUG START ===")
    
    # Test import
    print("Importing main...")
    import importlib.util, os
    os.chdir(r"c:\Users\User\Documents\4rth semester\AI\SisFall_dataset\submission")
    
    spec = importlib.util.spec_from_file_location(
        "main",
        r"c:\Users\User\Documents\4rth semester\AI\SisFall_dataset\submission\main.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("Import OK")
    
    print("Calling main()...")
    mod.main()
    print("=== DONE ===")

except Exception as e:
    traceback.print_exc()
    print(f"\nFATAL: {e}")
finally:
    log.close()
