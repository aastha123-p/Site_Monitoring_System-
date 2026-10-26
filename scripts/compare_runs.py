import pandas as pd

r1 = pd.read_csv('runs/detect/construction_site_yolov8n/results.csv')
r2 = pd.read_csv('runs/detect/construction_site_yolov8n-2/results.csv')
r3 = pd.read_csv('runs/detect/construction_site_yolov8n-3/results.csv')

for df in (r1, r2, r3):
    df.columns = df.columns.str.strip()

print("=" * 78)
print(f"{'Epoch':<6} | {'Run 1 mAP50':<12} | {'Run 2 mAP50':<12} | {'Run 3 mAP50':<12} | {'Run 3 vs R1':<12} | {'Run 3 vs R2'}")
print("=" * 78)

for e in range(len(r3)):
    m1 = r1.loc[e, 'metrics/mAP50(B)'] * 100
    m2 = r2.loc[e, 'metrics/mAP50(B)'] * 100
    m3 = r3.loc[e, 'metrics/mAP50(B)'] * 100
    diff1 = f"{m3 - m1:+.2f}%"
    diff2 = f"{m3 - m2:+.2f}%"
    print(f"E{e+1:<5} | {m1:6.2f}%      | {m2:6.2f}%      | {m3:6.2f}%      | {diff1:<12} | {diff2}")

print("=" * 78)

print("\n" + "=" * 78)
print(f"{'Epoch':<6} | {'Run 1 Recall':<12} | {'Run 2 Recall':<12} | {'Run 3 Recall':<12} | {'Run 3 vs R1':<12} | {'Run 3 vs R2'}")
print("=" * 78)

for e in range(len(r3)):
    rc1 = r1.loc[e, 'metrics/recall(B)'] * 100
    rc2 = r2.loc[e, 'metrics/recall(B)'] * 100
    rc3 = r3.loc[e, 'metrics/recall(B)'] * 100
    diff1 = f"{rc3 - rc1:+.2f}%"
    diff2 = f"{rc3 - rc2:+.2f}%"
    print(f"E{e+1:<5} | {rc1:6.2f}%      | {rc2:6.2f}%      | {rc3:6.2f}%      | {diff1:<12} | {diff2}")

print("=" * 78)

print("\n=== OVERALL PEAK COMPARISON (BEST VALUES) ===")
print(f"Metric        | Run 1 (31 ep)      | Run 2 (37 ep)      | Run 3 (20 ep)      | Winner")
print("-" * 75)
for metric, col in [
    ("mAP@0.50", "metrics/mAP50(B)"),
    ("Recall", "metrics/recall(B)"),
    ("mAP@0.50:0.95", "metrics/mAP50-95(B)"),
    ("Precision", "metrics/precision(B)")
]:
    b1, e1 = r1[col].max() * 100, r1[col].idxmax() + 1
    b2, e2 = r2[col].max() * 100, r2[col].idxmax() + 1
    b3, e3 = r3[col].max() * 100, r3[col].idxmax() + 1
    winner = "Run 3" if b3 >= max(b1, b2) else ("Run 1" if b1 >= b2 else "Run 2")
    print(f"{metric:<13} | {b1:5.2f}% (ep {e1:2d})   | {b2:5.2f}% (ep {e2:2d})   | {b3:5.2f}% (ep {e3:2d})   | {winner}")
