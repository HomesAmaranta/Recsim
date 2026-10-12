"""Step 1: 全量扫一遍日志，算每用户在训练期/测试期的统计量，再分层抽 N_USERS 个用户。

抽样约束：测试期真实点击数 >= PER_USER，保证每用户能出满 PER_USER 条正样本。
分层依据：训练期 CTR 五分位，避免只抽到高活跃头部用户。
"""
import os, sys, subprocess, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

AWK = r'''
NR==1 { next }
{
  split($0, f, ",")
  u=f[1]; d=f[3]+0; y=f[6]+0
  if (d >= TRS && d <= TRE)      { tn[u]++; tc[u]+=y }
  else if (d >= TES && d <= TEE) { en[u]++; ec[u]+=y }
}
END {
  for (u in tn)
    if (u in en)
      printf "%s\t%d\t%d\t%d\t%d\n", u, tn[u], tc[u], en[u], ec[u]
}
'''

def main():
    os.makedirs(C.DATA, exist_ok=True)
    if not os.path.exists(C.F_USER_STAT):
        print("[1/2] 扫描全量日志计算用户统计 (约 2-3 分钟)...")
        cmd = ["awk", "-v", f"TRS={C.TRAIN_START}", "-v", f"TRE={C.TRAIN_END}",
               "-v", f"TES={C.TEST_START}", "-v", f"TEE={C.TEST_END}", AWK, C.LOG]
        with open(C.F_USER_STAT, "w") as fo:
            subprocess.run(cmd, stdout=fo, check=True)
    else:
        print(f"[1/2] 复用已有 {C.F_USER_STAT}")

    rows = []
    for line in open(C.F_USER_STAT):
        u, tn, tc, en, ec = line.split()
        tn, tc, en, ec = int(tn), int(tc), int(en), int(ec)
        if tn < 50 or en < 30:            # 训练/测试期样本太少的用户不要
            continue
        if ec < C.PER_USER:               # 测试期正样本不够
            continue
        rows.append((u, tn, tc, en, ec, tc / tn))
    print(f"[2/2] 合格用户 {len(rows)} 个 (训练期>=50条, 测试期>=30条, 测试期点击>={C.PER_USER})")

    rows.sort(key=lambda r: r[5])
    per = C.N_USERS // C.CTR_STRATA
    rnd = random.Random(C.SEED)
    picked = []
    n = len(rows)
    for s in range(C.CTR_STRATA):
        lo, hi = n * s // C.CTR_STRATA, n * (s + 1) // C.CTR_STRATA
        band = rows[lo:hi]
        take = min(per, len(band))
        picked += rnd.sample(band, take)
        print(f"    层{s+1}: CTR [{band[0][5]:.3f}, {band[-1][5]:.3f}]  候选{len(band)}  抽{take}")

    picked.sort(key=lambda r: int(r[0]))
    with open(C.F_USERS, "w") as f:
        for r in picked:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.4f}\n")
    ctrs = [r[5] for r in picked]
    print(f"\n已选 {len(picked)} 个用户 -> {C.F_USERS}")
    print(f"训练期 CTR: min={min(ctrs):.3f} 中位={sorted(ctrs)[len(ctrs)//2]:.3f} max={max(ctrs):.3f}")
    print(f"训练期日志合计 {sum(r[1] for r in picked):,} 条, 测试期 {sum(r[3] for r in picked):,} 条 "
          f"(其中真实点击 {sum(r[4] for r in picked):,} 条)")

if __name__ == "__main__":
    main()
