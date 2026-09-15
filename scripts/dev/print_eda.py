import json

with open("outputs/eda_archcad_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

print("=== CLASS FREQUENCY TABLE ===")
print(f"{'Class Name':<22} | {'Instances':<10} | {'Share %':<8} | {'Image Presence %':<16}")
print("-" * 65)
sorted_classes = sorted(
    rep["class_frequencies"].items(),
    key=lambda x: x[1]["instance_count"],
    reverse=True
)
for c_name, d in sorted_classes:
    c_str = c_name.ljust(22)
    cnt_str = str(d["instance_count"]).rjust(9)
    share_str = f"{d['instance_share_pct']:>6.2f}%"
    pres_str = f"{d['image_presence_pct']:>14.2f}%"
    print(f"{c_str} | {cnt_str}  | {share_str} | {pres_str}")

print("\n=== OBJECT GEOMETRY TABLE ===")
print(f"{'Class Name':<20} | {'W_med':<7} | {'H_med':<7} | {'MinSide':<7} | {'AspRatio':<8} | {'Sub-8px %':<10}")
print("-" * 70)
for c_name, _ in sorted_classes:
    g = rep["class_geometries"].get(c_name, {})
    if g.get("count", 0) > 0:
        c_str = c_name.ljust(20)
        w_m = f"{g['width_median']:>6.1f}"
        h_m = f"{g['height_median']:>6.1f}"
        ms_m = f"{g['min_side_median']:>7.1f}"
        ar_m = f"{g['aspect_ratio_median']:>8.2f}"
        sub8 = f"{g['sub_stride8_pct']:>8.2f}%"
        print(f"{c_str} | {w_m} | {h_m} | {ms_m} | {ar_m} | {sub8}")

print("\n=== CO-OCCURRENCE (TOP 5 CORRELATED CLASSES) ===")
for c_name, co_list in rep.get("co_occurrence_top5", {}).items():
    print(f"{c_name:<20}: {', '.join(co_list)}")
