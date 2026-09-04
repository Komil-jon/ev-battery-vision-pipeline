# Where to submit the detection + localisation paper

Written 2026-09-04, after the decision not to proceed with ICMRA 2026.
Paper: `paper/icmra2026_ev_battery.tex` (currently 6 pages, IEEEtran conference).

---

## 1. Objective read on the paper

### What is genuinely strong
- **Three counterintuitive empirical results.** The 66% cross-facility drop; the
  finding that a DINOv2 backbone buys graceful degradation rather than better
  detection; and that reprojection error does not predict metric accuracy. Reviewers
  reward findings that change what other people do, and all three do.
- **A public release in a field that has none.** 16,945 polygon masks, audit tooling,
  weights, eval code. Every prior study in this area self-collected and released
  nothing. This is a real community contribution.
- **Real hardware.** 9/9 target reaches on a UR5e from a single RGB stream.
- **Unusual candour.** The post-alignment caveat, the retracted inflated score, the
  negative results section. This reads as trustworthy, which matters.

### What reviewers will attack, in order of severity
1. **The two halves are not connected.** The detector emits pixels; the localisation
   stage still uses a fiducial *on the target*. The paper admits this, but the
   headline claim is a pipeline, and the pipeline is not demonstrated end to end.
   This is the single largest rejection risk at a top venue. Expect the phrase
   "two papers stapled together".
2. **N = 9 configurations, one execution each.** No repeats, no statistics, manual
   ground truth. Robotics reviewers expect >= 20 trials with repeats.
3. **Low absolute detection numbers** (0.502 mAP@50; busbar 0.30-0.37). Defensible in
   a cross-facility setting, but it looks weak at a glance and the busbar class is
   under-annotated in the data itself.
4. **No methodological novelty.** Off-the-shelf YOLO11n, RF-DETR, ArUco + PnP. The
   contribution is analysis and data. That is welcome at CASE/ETFA, tolerated at
   IROS, and a hard sell at ICRA unless the framing carries it.
5. **Benchmark provenance.** The data is assembled from public repositories with mixed
   licences and unverified original labelling. A reviewer can turn the paper's own
   thesis against it: if annotation convention dominates, how trustworthy is a
   benchmark built out of other people's conventions?
6. **Five bibliography entries have no authors** (`screwrcnn2021`, `screwbatt2023`,
   `screwcompare2026`, `cvmodule2023`, `rapid2026`) and the `rapid2026` arXiv id
   should be re-checked. Small, but reviewers notice and it costs goodwill.

### Honest tier placement
Strong CASE / ETFA paper. Borderline IROS paper. Long-shot but non-zero at ICRA.

---

## 2. The venue landscape (verified 2026-09-04)

| Venue | Deadline | Notification | Event | Fit | Notes |
|---|---|---|---|---|---|
| **ICRA 2027** | **15 Sep 2026** | 31 Jan 2027 | Seoul, 24-28 May 2027 | Medium | 8 pages incl. refs; no over-length option |
| **CIRP CMS 2027** | abstract 11 Sep 2026, full 18 Nov 2026 | 19 Feb 2027 | Vienna, 12-14 May 2027 | Medium-high | Procedia CIRP, not IEEE; registration-funded |
| **IEEE CASE 2027** | **1 Mar 2027** | 15 May 2027 | Linz, 23-27 Aug 2027 | **Highest** | Core topics include remanufacturing and disassembly |
| **IROS 2027** | **1 Mar 2027** | ~Jun 2027 | Florence, 26 Sep-1 Oct 2027 | Medium-high | Clashes with CASE; pick one |
| ETFA 2027 | ~Mar 2027 (TBA) | - | Pisa, 7-10 Sep 2027 | High | IEEE IES; industrial automation audience |
| RCIM / J. Manuf. Syst. | rolling | - | - | High | Q1 journal route once the integration exists |

---

## 3. Recommended plan

**The timing lines up almost perfectly, and that is the whole argument.**

### Step 1 — Submit to ICRA 2027 by 15 September 2026
Free to submit, and the paper is already in IEEEtran at 6 pages against an 8-page
limit, so there are two pages of headroom rather than a squeeze. Realistic acceptance
odds: roughly 1 in 4 or 5. That is fine, because:

**ICRA notifies on 31 January 2027. CASE and IROS both close on 1 March 2027.**

A rejection arrives a month before the next deadline, with three sets of reviews
attached. There is no scenario in which submitting to ICRA costs anything except the
week of work needed to strengthen the paper, and that work is needed for CASE anyway.

### Step 2 — Between October 2026 and February 2027, run the integration experiment
Replace target marker ID 4 with a detected mask centroid; keep the four workspace
markers. Even a 3-configuration pilot removes criticism #1 above. Aim for >= 20 trials
with repeats to remove criticism #2. See `INTEGRATION_WITH_LOCALISATION.md` §5 for the
three hard problems (Z source, centroid vs corner, expected 5-15 mm degradation).

### Step 3 — 1 March 2027: submit to CASE 2027, or IROS 2027
Choose in February on the strength of the integration result:
- Integration **works and is measured** -> IROS 2027 (Florence). Stronger venue, and
  a demonstrated markerless pipeline is an IROS-shaped contribution.
- Integration **incomplete or the paper stays analysis-led** -> CASE 2027 (Linz).
  Best topical fit of any venue on this list, higher acceptance rate, and a
  dataset-and-analysis paper is squarely within scope.

### Step 4 — fallback
ETFA 2027 (~March deadline, Pisa) if both miss. Longer term, the integrated system is
a genuine Robotics and Computer-Integrated Manufacturing paper, with no deadline and
no page limit.

### Considered and not recommended as primary
**CIRP CMS 2027** (Vienna) is a decent topical fit and its abstract deadline is
7 days away, but it is Procedia CIRP rather than IEEE, carries less weight with a
robotics readership, and is funded through registration in the same way ICMRA was.
Worth keeping as a parallel option only if a non-IEEE, faster outlet becomes
attractive.

---

## 4. Before anything is submitted

- [ ] **Formally withdraw from ICMRA 2026 in writing.** The paper was accepted there.
      Not registering usually means it is never published, but an explicit withdrawal
      to the conference secretary removes any dual-submission question, and every
      venue above asks authors to declare that the work is not under review elsewhere.
- [ ] Get Yujie Wang's and Shiv's agreement on the change of venue and the author list.
- [ ] Fix the five author-less bibliography entries and verify the `rapid2026` arXiv id.
- [ ] Re-check that overlap with the published EECSS/MVML paper (doi 10.11159/mvml26.125)
      is declared clearly enough; it currently is, in the Introduction.

---

# Appendix A — Venue longlist for Paper A (vision / dataset / benchmark)

Compiled 2026-09-04 from the IAPR conference schedule, the INSTICC portal, IEEE
IES/RAS calendars, CORE Portal, WikiCFP and Clocate. Paper A = cross-facility
benchmark, detector comparison, annotation-convention finding, dataset release.
Localisation content removed.

## Open now, ordered by deadline

| Venue | Deadline | Notify | Event | Format | Indexing |
|---|---|---|---|---|---|
| VISAPP 2027 | 15 Sep 2026 | 13 Nov 2026 | Valletta, Malta, 26-28 Feb 2027 | Scitepress | Scopus + WoS; CORE B (2018) |
| ICPRAM 2027 | 15 Sep 2026 | ~13 Nov 2026 | Valletta, Malta, 20-22 Feb 2027 | Scitepress | Scopus + WoS |
| ROBOVIS 2027 | 15 Sep / 22 Oct 2026 | - | Valletta, Malta, 27-28 Feb 2027 | Scitepress | Springer CCIS |
| **ICPRS 2027** | **25 Oct 2026** | **28 Dec 2026** | Talence (Bordeaux), 8-11 Mar 2027 | **IEEE, 6+1 pp** | **IEEE Xplore**, IAPR-endorsed |
| IMPROVE 2027 | 17 Nov 2026 | - | with CLOSER/VEHITS 2027 | Scitepress | Scopus + WoS, Springer CCIS |
| IEEE ISIE 2027 | 15 Dec 2026 | - | Paris, 30 Jun-3 Jul 2027 | IEEE | IEEE Xplore |
| SCIA 2027 | 26 Jan 2027 | - | Gjovik, Norway, 8-11 Jun 2027 | Springer LNCS | Scopus; ~40-50% accept |
| IJCNN 2027 | 31 Jan 2027 | 15 Mar 2027 | Cape Town, 14-18 Jun 2027 | IEEE, 6 pp | IEEE Xplore; ~55-60% accept |
| **ICPR 2027** | **1 Mar 2027** | 24 May 2027 | **Virtual**, 4-15 Oct 2027 | Springer LNCS | Scopus; CORE B |

Further out: ICIAP 2027 (~Feb 2027, LNCS), IEEE INDIN 2027 (Lisbon, May 2027),
ICIP 2027 (Singapore, deadline TBA), BMVC 2027 (~May 2027), WACV 2028 (~Jun 2027,
has an Evaluation & Datasets track).

Closed: SII 2027, WACV 2027, CIRP LCE 2027, DICTA 2026, ETFA 2026, VCIP 2026.

## Why ICPRS 2027 is the primary target

- It lists **"dataset papers introducing new public datasets and baselines"** as an
  explicit submission category. No other venue on this list says that out loud.
- IEEE format, 6 pages + 1 for references — the paper is already IEEEtran at 6 pages.
- IEEE Xplore indexed, IAPR endorsed, IEEE sponsored, double-blind via ConfTool.
- Student paper prize; first author is a student.
- Bordeaux is cheap and visa-free from the UK.
- Small conference, so a well-matched paper has better odds than at a large one.
- Seven weeks is enough to do the split properly rather than rushing it.

## Dual-submission constraint (this drives the ordering)

VISAPP notifies **13 Nov**, which is *after* the ICPRS deadline of **25 Oct**. The
two cannot both be live. Pick one chain:

**Chain B (recommended)**
ICPRS 25 Oct -> notify 28 Dec -> if rejected, SCIA 26 Jan (4 weeks' turnaround)
-> if rejected, ICPR 2027 1 Mar. Three shots, no overlap.

**Chain A (faster start, weaker fit)**
VISAPP 15 Sep -> notify 13 Nov -> if rejected, SCIA 26 Jan -> ICPR 2027 1 Mar.
Also three shots, but forfeits ICPRS and costs EUR 725-795 if accepted.

## Cost

- VISAPP speaker registration **EUR 725-795 early**, plus EUR 50 per extra page.
  More than ICMRA charged. Confirm funding before submitting.
- ICPRS fees not yet published; comparable European conferences run EUR 400-550.
  One full-fee registration covers up to three papers.
- ICPR 2027 is virtual, so travel cost is zero.
- SCIA ~EUR 500. IJCNN IEEE rates ~USD 700-800.

## Free-to-publish alternative

Hybrid subscription journals charge nothing unless open access is chosen:
*Machine Vision and Applications* (Springer), *Robotics and Computer-Integrated
Manufacturing*, *Journal of Manufacturing Systems*, *Computers in Industry*.
No deadline, no page limit, no registration fee, higher standing than any
conference above. Slower: 6-12 months.
