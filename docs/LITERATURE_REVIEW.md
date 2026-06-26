# Literature Review — 21 passive-radar drone-detection papers (deep read)

> Full-text read of every paper in `/data/public/jeong/papers/{5G,LTE,Wifi}/`
> (2026-06-26, 7 parallel deep-read agents). Exact numbers, cited by paper.
> Companion to `paper_survey_unified.md` (the earlier structured spec) — this
> is the deeper, number-by-number version that grounds Phase-1 (`phase1.py`).

Paper IDs: `5G-21/22/23a(Network)/23b(Spectrum)/25a(Intrusion)/25b(SSB)/26(LaSen)`,
`LTE-19/20/22/23/24/25a(LIPASE)/25b(ONERA-TAES)`,
`WiFi-17/21a(Rzewuski)/21b(Milani)/22(RpF-IRS)/23a(IDP-conf)/23b(IDP-TAES)/24(compare)`.

---

## 1. Executive summary — cross-cutting findings
1. **Reference structure is the dominant design variable** (the project's thesis, confirmed everywhere): LTE works best using **CRS symbols 0 & 4 only** (LTE-23/25b: all-symbol gives +6.7 dB SNR — 24.2 vs 17.5 dB — but **~3× more false plots**, so CRS-only is the operating point); 5G is **occupancy-limited** (5G-22: ~10 % occupancy ≈ undetectable, ~70 % ≈ +24 dB; LaSen: live 5G is only **~3 % occupied on average**); Wi-Fi splits into full-packet vs **preamble-only (L-STF/L-LTF)** vs amplitude-only (IDP). → "energy ≠ detectability; use the deterministic structure you have."
2. **No paper measures a drone RCS by experiment.** The only quantitative anchors: Wi-Fi small quadrotor **≈ −20 dBsm avg (−40…0 dBsm)** at 2.48 GHz (WiFi-21a, FDTD); 5G link-budget **0.1 m² = −10 dBsm / 0.05 m² = −13 dBsm** (5G-25a/b, illustrative); reflector-augmented drone **0.07 m² = −11.5 dBsm** (LTE-22); foam/carbon micro-drone **~−25…−30 dBsm** (WiFi-17, qualitative); steel-sphere payload **0.196 m² = −7 dBsm** (LIPASE geometry). RCS is **frequency-dependent** (WiFi-21a/LTE-24 flag the Rayleigh↔Mie transition when body≈λ). → our Axis-B is a genuine gap we fill, but must be labelled estimate.
3. **All 21 are bulk translational Doppler.** Propeller micro-Doppler is only ever *future work* (LaSen, 5G-22) or **qualitative** (5G-25a: TDD-gated propeller "stripes", no blade count/RPM). Where micro-Doppler IS resolved (WiFi-23b/24) it is **human limbs, not drone blades** — do not conflate. → Phase-1's bulk-only scope matches the experimental state of the art.
4. **Zero-Doppler / bistatic-contour blind zone is universal.** LIPASE (LTE-25a) explicitly **misses the drone at 6.5–8.0 s when it rides the zero-bistatic-Doppler contour**; ONERA (LTE-25b) loses targets near zero-Doppler and near the Tx-direction notch. → directly validates Phase-1's headline result that **hover & tangential motion are blind** for bulk Doppler.
5. **Nobody did a fair cross-band Wi-Fi/LTE/5G comparison** under a fixed budget with repeatable trajectories — the project's novelty.

---

## 2. The reference-structure axis (project core) — grounded
| Band | Deterministic reference | Density / occupancy | Source |
|---|---|---|---|
| **Wi-Fi** | OFDM **PHY preamble (L-STF 0.8 µs + L-LTF, 8 µs each → 320 samples @20 MHz)**; or full packet; or amplitude-only | OFDM occupancy **95–99 %** in-packet, but **duty Fu ≈ 18 %** across time (packet bursts) | WiFi-22/23b/24, WiFi-21a |
| **LTE** | **CRS on symbols 0 & 4**, every 6th subcarrier (comb-6; comb-3 / 3-SC shift for 2 ports); PSS/SSS on 62 SC | CRS ≈ **4.8 %** of grid; carries **~25 % of DL power** (LTE-24); all-symbol = full but +3× FAR | LTE-23/25b, LTE-24 |
| **5G** | **SSB = 4 OFDM symbols × 240 subcarriers** (PSS 127 / SSS 127 / PBCH 240), period 5–160 ms; or full PDSCH | SSB nulled-symbols cost ≈ 0 vs full block (5G-25b); live grid ~3 % (LaSen); CSI-RS ≤ 200–500 Hz | 5G-25b/23b, 5G-26 |

**Key experiments to (re)produce in Sionna:** LTE **CRS-only vs all-symbol** SCR/FAR trade (LTE-23 Table I: sym0=17.5 dB, sym4=17.9 dB, all=24.2 dB but 62 vs ~19 plots; sym 2&3 with no CRS *miss the target*). 5G **occupancy → SCR** (5G-22 Fig.10: 10 %→invisible, 70 %→+24 dB). Wi-Fi **preamble-only ≈ −1…−11 dB SNR vs full** but data-independent (WiFi-24).

---

## 3. RCS grounding (Axis B) — the available anchors
DJI does not publish RCS; no paper measured it. Recommended Phase-1 anchors (label as ESTIMATE, frequency-dependent):
- **Small consumer quad (Phantom/Mavic/Air class), S/C-band (2–4 GHz): ≈ −20 dBsm** nominal, sweep **−40…0 dBsm** (WiFi-21a FDTD at 2.48 GHz is the best single source).
- **5G C-band (3.5 GHz): −10 to −13 dBsm** (0.1 / 0.05 m², 5G-25a/b link budgets).
- **Ultra-light foam/carbon micro-drone: −25…−30 dBsm** (WiFi-17).
- **With payload / corner reflector: −7 to −11.5 dBsm** (LIPASE 25 cm steel sphere = 0.196 m²; LTE-22 reflector = 0.07 m²) — an "easy"/upper bound.
- **UHF/LTE450 (461 MHz, λ≈64 cm):** body≈λ → Rayleigh/Mie transition; **+10 dB for horizontal polarization** (carbon blades, LTE-24). RCS rolls off below ~600 MHz.
- DJI body sizes (for size→RCS scaling): Phantom-class ~35 cm; Mavic Pro 30×25×8 cm, 730 g; Matrice M210 88×88×39 cm; Matrice 4E 307×388×150 mm; Mini 4 Pro 298×373×101 mm.

---

## 4. Per-band synthesis (exact)
### 5G
- **Carriers used:** 584 MHz/8 MHz (5G-21, NUDT), **3.44 GHz / 38.16 MHz TDD** (5G-22/23a/25a, Warsaw — the primary anchor, range res **7.8 m**), 3.5 & 4.85 GHz (5G-23b), 5.8 GHz (LaSen SDR), **15 GHz / 60 kHz SCS** (5G-25b sim), prediction grid **0.75 / 3.5 / 25 GHz** (5G-25a).
- **CPI:** 20 ms & **100 ms** (5G-22/25a → ΔV≈λ/T≈0.87 m/s @3.44 GHz, 100 ms). LaSen 100 ms window / 50 ms hop.
- **Clutter:** CLEAN / adaptive lattice (5G-22/25a); uplink blanked first (TDD); LaSen = mean-subtraction + atom isolation (|v|<1 m/s, R<3 m masked).
- **CFAR/budget:** classical CA-CFAR, Pfa 1e-4/1e-6/1e-8 (5G-22); detection thresholds **D0 = 10/15/20 dB**; budget Pt 45 dBm, Gt 15 dBi, Gr 10 dBi, L 10 dB, F=60 %, Br 61.44 MHz (5G-25a Table I). Predicted ranges: **6 km / 3 km / 1 km @ 0.75/3.5/25 GHz**.
- **Results:** real 3.44 GHz drone detected **60–150 m bistatic, ±7 m/s**, GPS-confirmed (5G-23a); first 1-min 5G track (5G-22). LaSen: **distance RMSE 1.06 m, velocity 0.34 m/s, up to 108 m & 20.2 m/s** (monostatic ISAC, RTK GT).
- **Micro-Doppler:** only 5G-25a, qualitative TDD-gated propeller stripes (no numbers). 5G-23b is a **CNN classifier (no RD/CFAR)** — 93 %/82 % accuracy, DJI Phantom 4.

### LTE
- **Carriers/BW:** **LTE450 461 MHz / 6.5 MHz** (LTE-24, range to **2.7 km**!), **1.87 GHz / 20 MHz** (LTE-23/25b, CRS), 2.13 GHz / **5 MHz** (LIPASE), 2.495 GHz / 20 MHz (LTE-22), 2.6 GHz (LTE-25b flights 4-6), 850 MHz / 1.4–20 MHz table (LTE-20 theory).
- **Reference:** **CRS sym 0 & 4** is THE result (LTE-23/25b). LTE-20 designs a custom TLRS waveform (0.266 % of PDSCH → 150 km MUR).
- **CPI:** **125 ms** (LTE-23/25b), **200 ms** (LIPASE, range res 30 m @5 MHz, Doppler res 5 Hz).
- **Clutter:** ECA / LS-CC (P=0, L=20, LIPASE) / frequency-angular covariance R⁻¹(k) (beats ECA ~+5 dB w/ multi-signal, LTE-25b) / Wiener-Hopf (LTE-20 ideal) / DLC@10 ms + SVD-DPI (LTE-24).
- **CFAR:** **13 dB ↔ Pfa 1e-6** (LTE-19/25b, LO-CA-CFAR); Pfa **0.01/0.001/0.0001** (LTE-24, range shrinks 2700→1500 m); LIPASE 2D CA-CFAR guard(60,1)/train(60,1) α=15 dB.
- **Results:** post-cancellation **~25 dB SNR vs 13 dB threshold** (LTE-19, Phantom 4); **LIPASE localization RMSE 1.49 m** (DGPS GT, det 71.9 %/miss 28.1 %/FA 18.2 %) — best LTE tracking. Detection-range ladder: **2.7 km (LTE450) > 180–250 m (1.87 GHz) > ~20 m (USRP+reflector)**.
- **Polarization:** horizontal ≈ **+10 dB** for quadcopter carbon blades (LTE-24).

### Wi-Fi
- **Carriers:** 2.4 GHz 802.11b DSSS (WiFi-17, ~22 MHz, beacon PRI 3 ms), 2.432 GHz ch5 802.11n (WiFi-21b, 40 MHz samp), **2.472 GHz ch13 802.11n & 5.18 GHz ch36 802.11ac** (WiFi-22/23b/24, fs 20 MHz, B_OFDM **16.6 MHz**), 2.48 GHz + DVB-T 562/594/634 MHz (WiFi-21a).
- **Reference modes:** coupler tap / demod-remod / **preamble-only L-STF/L-LTF (320 samples)** / amplitude-only IDP. **RpF (reciprocal filter) PSLR = 42 dB vs MF 25 dB**, but **~10 dB SNR loss** (I-RpF recovers to ~3-4 dB) and removes the **L-STF 0.8 µs → 240 m range ambiguity** (WiFi-22).
- **CPI:** **0.3 s (range-Doppler) / 0.5 s (Doppler-time)** (WiFi-22/24); 0.3 s aircraft / 0.5 s drone (WiFi-17).
- **Clutter:** ECA-S / ECA-S-a-priori (sliding, coeff window **0.05 s fast → 0.2 s slow**, update = packet rate, 250 m extent); lattice filter (WiFi-21a); 3-antenna spatial.
- **CFAR/budget:** CA-CFAR Pfa **1e-6** (WiFi-24); **3:3 multi-channel** (per-ch Pfa 1e-2→system 1e-6, or 1e-1.3→1e-4, WiFi-17); SNR threshold **8 dB, Pfa 1e-4, NF 6.5 dB** (WiFi-21a). **Duty Fu = 18 %** (WiFi-21a).
- **Results:** very-low-RCS micro-drone (60×60×9 cm foam) detected ~40 m bistatic, 3D incl. height (WiFi-17, **no GT**); DJI Mavic Pro fused tracking **~1.6 m** position error (WiFi-21b, GPS GT). Reference-free IDP: **no range axis**, Doppler-sign-ambiguous, OFDM background **BNR ∝ DNR²** unless preamble-only/BPSK.

---

## 5. CPI / CFAR / Doppler grounding for Phase-1
- **Realistic CPI: 100–500 ms** (5G 100 ms, LTE 125–200 ms, Wi-Fi 300–500 ms). Phase-1 currently uses **50 ms** → on the short side; bump to **100 ms** to match literature (Doppler res 10 Hz).
- **CFAR Pfa:** literature spans **1e-2 … 1e-8**; common operating points **1e-6** (LTE/Wi-Fi) and **1e-4**. Detection thresholds **8 dB (Wi-Fi), 13 dB (LTE), 10/15/20 dB (5G)** SNR.
- **Bulk Doppler scope:** ±80 Hz (LTE450) … ±250 Hz (LTE 1.87 GHz, 20 m/s) … ±300 Hz (5G radial). Phase-1's fD ≈ 88–373 Hz is consistent.
- **Clutter:** ECA / ECA-S / LS-CC / lattice are the field standard; Phase-1's **ideal static cancellation is the upper bound** — add data-driven ECA next.

---

## 6. Scenario/motion grounding (validates Phase-1)
- **Radial out-and-back** is the de-facto experimental trajectory in nearly every paper (5G-23a ±7 m/s; LTE-19/23/24 ~10 m/s; WiFi-17 soar+radial). **Speeds 1.4–20 m/s.**
- **Hover / move-stop-move** appears in WiFi-21b (move-stop-move waypoint, 1.4 m/s, hover at vertices) and WiFi-17 (vertical soar). LIPASE has a **tangential zero-Doppler leg that is missed**.
- **Tangential / zero-Doppler blind zone is documented as a real failure** (LIPASE 6.5–8 s; ONERA near-zero-Doppler & rejected-angle misses). → Phase-1's "hover/tangential blind" is literature-correct, not an artifact.
- **Waypoint** ('J', quadrilateral, curved): LIPASE 'J', WiFi-21b quadrilateral, ONERA 6 curved flights. Phase-1's 2-segment waypoint is a reasonable proxy.

---

## 7. Ground truth ladder
- **RTK / cm-level:** LaSen (5G-26), and the project's target. **DGPS sub-meter:** LIPASE (LTE-25a). **Onboard GPS (~m, unquantified):** most LTE/5G. **None / geometry-only:** WiFi-17 ("GT not available"), most reference-free Wi-Fi (cooperative, qualitative).
- Sionna's exact, free GT is **strictly better than the entire literature** — the fairness lives there.

---

## 8. Concrete Phase-1 refinements (code-level, grounded)
1. **drones.py RCS → dBsm:** replace size-only proxy with literature dBsm anchors (−20 dBsm small quad @S/C-band; −10/−13 dBsm @C-band; scale per band/size), labelled ESTIMATE. (Applied.)
2. **CPI 50 → 100 ms** (literature default; Doppler res 10 Hz). (Recommended.)
3. **Spec grid:** add **LTE450 461 MHz / ~5 MHz** (long-range, low-band anchor) and confirm LTE **1.87 GHz/20 MHz CRS sym 0&4**, 5G **3.44 GHz/38 MHz SSB**, Wi-Fi **2.472 GHz(11n)/5.18 GHz(11ac)** preamble. (Doc'd; grid editable in `SPECS`.)
4. **CFAR Pfa default 1e-6** with a sweep {1e-4,1e-6}; detection thresholds per band (8/13/15 dB) as covariates. (Recommended.)
5. **Reference structure:** keep CRS sym {0,4,7,11} comb-6 (LTE), SSB 4×240 (5G), preamble-rich (Wi-Fi) — already in `passive_radar_s2.pilot_mask`; add the **CRS-only vs all-symbol** sub-experiment to reproduce LTE-23's +6.7 dB/3× FAR result.
6. **Polarization (+10 dB H-pol, LTE450)** and **duty/occupancy (Wi-Fi Fu 18 %, 5G ~3–60 %)** as documented covariates / future knobs.
7. **Clutter:** label current ideal cancellation as upper bound; data-driven **ECA-S** is the realistic next step.

---

## 9. Gaps the project fills (novelty, evidence-based)
- **Fair cross-band Wi-Fi/LTE/5G comparison under a fixed budget** — done by none.
- **Repeatable trajectories + exact/RTK-grade GT** — only LaSen & LIPASE approach it; Sionna gives it free.
- **Motion-scenario axis (hover/radial/tangential/waypoint)** — no paper sweeps motion as a controlled variable; the zero-Doppler blind zone is observed but never benchmarked.
- **Drone-RCS as a controlled axis** — never measured in any paper.
- **(Future) propeller micro-Doppler for drones** — absent from all 21 (human limbs only in Wi-Fi); the project's optional layer would be genuinely new.

---
*Per-paper exhaustive extractions (with page/figure citations and full parameter tables) are preserved in the deep-read agent outputs; this review distils them. See `paper_survey_unified.md` for the earlier structured spec.*
