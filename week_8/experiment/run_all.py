"""
汇总跑 E1–E7（子进程隔离，单条失败不影响其它）。
单独跑某条：python e3_review_gate.py
全跑：python run_all.py
"""
import subprocess
import sys

TESTS = [
    ("e1_routing.py", "supervisor 路由正确性"),
    ("e2_task_description.py", "task_description 四要素"),
    ("e3_review_gate.py", "review_count 闸门收口"),
    ("e4_counters.py", "两计数器不串扰"),
    ("e5_isolation.py", "上下文隔离"),
    ("e6_best_so_far.py", "best-so-far 收口"),
    ("e7_two_layer_gate.py", "双层闸门嵌套"),
]

if __name__ == "__main__":
    npass = 0
    for f, name in TESTS:
        p = subprocess.run([sys.executable, f], capture_output=True, text=True)
        ok = p.returncode == 0
        npass += ok
        print(p.stdout.rstrip())
    print("-" * 60)
    print(f"总计 {npass}/{len(TESTS)}")
    sys.exit(0 if npass == len(TESTS) else 1)
