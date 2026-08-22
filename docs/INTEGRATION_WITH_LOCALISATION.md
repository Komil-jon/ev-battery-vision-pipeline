# Integrating the detector with Yujie's ArUco localisation

Analysis of Yujie Wang's MSc final report, *Development and Validation of an
ArUco-Based 3D Localisation Method for Vision-Guided Robotic Manipulation*
(supervisor: S. A. Katiyar), and how it combines with the detection work in this
repository. Written 2026-08-22 for the next paper.

Code: https://github.com/xcdgdj/D405-ArUco-UR5e-Validation

---

## 1. What changed since the ICMRA submission

The ICMRA paper reported this work **in simulation**. The final report is **real
hardware**, which removes the single biggest limitation we declared.

| | ICMRA version (PyBullet) | Final report (real) |
|---|---|---|
| Platform | Simulation | **UR5e + Intel RealSense D405** |
| Depth | n/a | **Deliberately unused — RGB only** |
| Reference | 3–4 sim markers | 4 fixed 80 mm markers (ID 0–3) + 40 mm target (ID 4) |
| Trials | 30 randomised | 9 object-position configs, 270 accepted samples, **9 physical robot motions** |
| Headline | 5.67 mm mean 3D error | Height MAE **2.245 mm**, RMSE 3.177 mm; stand-off MAE **1.556 mm** |
| Execution | none | **9/9 target reaches (100%)** |

Method in one line: four fixed markers define a Board frame by joint PnP over all
visible corners; the target marker is solved separately with IPPE; frames are accepted
only at reprojection RMSE ≤ 2.0 px; the coordinate-wise median of 30 accepted samples
becomes the target; a calibrated planar Board→Base transform plus a separately
measured work-surface height produce the UR5e command; motion runs in two stages
(approach then slow descent) with workspace bounds enforced.

## 2. Reading the numbers honestly

The report is unusually candid about its own weak points, and we should preserve that
rather than quoting the headline figures flat.

**The planar figure is not what it looks like.** The reported 1.209 mm mean planar
error is dominated by six values between 0.003 and 0.015 mm — and the report states
these were measured *after manual flange-centre alignment*, so they are not
independent measurements of autonomous accuracy. The three genuinely autonomous
values (O1) average **3.605 mm with a 6.091 mm maximum**, and the planar calibration
fit itself had **RMSE 5.877 mm**. Any claim we make should use the O1 numbers or the
calibration RMSE, never the 1.209 mm.

**Reprojection error does not predict metric accuracy.** O2-P2 and O2-P3 produced
height errors of −5.330 mm and +6.910 mm at target reprojection RMSE of only 0.101
and 0.052 px. This is the most interesting single finding in the report: image-space
quality checks are necessary but not sufficient, because planar PnP depth is
sensitive to corner geometry, target tilt, paper flatness and the physical definition
of the target plane. It is worth featuring, not burying.

**Ground truth is manual.** XY and stand-off were measured by hand, so the report
correctly calls these experimental validation values rather than traceable metrology.

## 3. Why the two halves genuinely fit

This is not two methods stapled together. Each removes the other's stated limitation.

> "The current setup also assumes that a fiducial can be placed on the target, which
> is not always realistic in recycling." — Yujie's report, §4.4

That is exactly what our detector removes. Conversely, our detector returns pixels and
cannot command a manipulator; the Board frame is exactly what supplies metric
coordinates.

| | Detection (ours) | Localisation (Yujie) | Combined |
|---|---|---|---|
| Finds object without markers | yes | **no** | yes |
| Metric robot coordinates | **no** | yes | yes |
| Works across pack types | yes (0.502 mAP@50, 10 sources) | untested | yes |
| Drives a real robot | **no** | yes (9/9) | yes |

**The integration claim:** replace target marker ID 4 with a learned detection, keep
the four workspace markers as the metric reference. Markers stay in the *workcell*,
where they are cheap and permanent, and disappear from the *object*, where they were
unrealistic.

## 4. Proposed architecture

```
D405 RGB frame (depth unused)
      |
      +--> ArUco ID 0-3  --> joint PnP --> Board frame  (metric reference)
      |
      +--> YOLO11 / RF-DETR --> module + busbar instances (pixels)
                                      |
                            mask centroid (u, v)
                                      |
              back-project ray through (u,v), intersect target plane at Z
                                      |
                       target in Board frame --> Board->Base transform
                                      |
                        UR5e two-stage approach (bounded, 0.020 / 0.005 m/s)
```

Everything after the centroid is Yujie's existing, validated chain. The only new
component is the substitution of the detection for the marker.

## 5. The three hard problems

### 5.1 Where does Z come from? (the crux)
Marker ID 4 supplied target *height* directly from its PnP pose. A detection does not.
Options, roughly in order of preference:

1. **Known geometry per module class.** Battery modules of a given type have a known
   height, and our detector already distinguishes classes. Requires a lookup, and
   fails on unknown packs.
2. **Planar assumption on a known tray.** Modules sit in a pack tray at known height
   above the calibrated Board plane. Cheapest and consistent with the existing chain;
   breaks on stacked or tilted packs.
3. **Apparent-size estimation.** With known physical module width, the mask's pixel
   extent gives range. Novel and uses the segmentation masks we already hold, but
   sensitive to detection boundary error.
4. **Re-enable the D405 depth stream** for Z only, keeping ArUco for XY. Pragmatic,
   and directly answers the comparison the report lists as future work.

Recommendation: implement (2) as the baseline because it changes least, and evaluate
(4) as the comparison. (3) is the interesting research angle if time allows.

### 5.2 Centroid is not a marker corner
ArUco corners are sub-pixel and physically well defined. A bounding-box centre is
neither — it is not a point on the object, and it moves with the box. **This is where
our polygon masks matter**: a mask centroid is a physically meaningful surface point
and far more stable than a box centre. It also gives orientation via `minAreaRect`,
which the marker supplied for free. Use masks, not boxes.

### 5.3 Accuracy will get worse, and we should say so up front
Marker corners are detected at ~0.2 px reprojection RMSE. A CNN detection is nowhere
near that. At the reported intrinsics (fx ≈ 649 px) and a working distance of roughly
0.4–0.6 m, one pixel is about 0.6–0.9 mm at the object, so a centroid off by 5–10 px
costs 3–9 mm. Added to the existing 5.877 mm planar calibration RMSE, an integrated
system plausibly lands in the **5–15 mm** range rather than Yujie's 2–3 mm.

That is still usable for a coarse approach with a compliant or force-controlled final
action, which is precisely how the report frames its own stand-off accuracy. But we
should predict the degradation, measure it, and not present the marker-based numbers
as if they carried over.

## 6. What this means for the next paper

Target: **TAROS 2027** (12-page limit, UK, robotics readership, ~March deadline).

Working title: *A markerless RGB-only perception-to-manipulation pipeline for EV
battery disassembly.*

The contribution becomes a chain nobody has demonstrated end to end for this
application:

1. A detector validated **across ten pack sources**, not one facility (ours).
2. A metric localisation chain validated on **real hardware** with 9/9 execution
   (Yujie's).
3. The substitution that removes the fiducial-on-target requirement (new).
4. An honest error budget showing where accuracy is lost — reference frame,
   calibration, detection — which the report already begins and we can complete.

Experiments needed before submission:
- [ ] Detector running on D405 RGB in the actual workcell
- [ ] Mask centroid → Board frame → Base, replacing marker ID 4
- [ ] N ≥ 20 physical trials on real battery modules, repeated per configuration
- [ ] Comparison against Yujie's marker-based numbers, same workcell
- [ ] Independent metrology for ground truth if at all possible
- [ ] Optional: RGB-only vs D405 depth for Z

Carry over unchanged: the report's honesty about manual ground truth, the
post-alignment caveat, and the finding that reprojection error does not predict metric
accuracy.
