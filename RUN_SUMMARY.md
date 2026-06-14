# RUN_SUMMARY — `training_output_5k_run1`

| | |
|---|---|
| Run | 5000 episodes, seed=42, vary-scenarios=True, fuel-damage=False |
| Wall time | 182m 51s (~3h, 2.19s/episode avg) |
| Validation cadence | every 100 ep → 50 audits |
| Recording cadence | every 50 ep → 100 RL+VAL recordings + 309 flagged-replay |
| Output dir | `training_output_5k_run1/` |
| Source `run_summary.txt` | [`training_output_5k_run1/logs/run_summary.txt`](training_output_5k_run1/logs/run_summary.txt) |

---

## 1. מגמת למידה (Progress Blocks)

נדגמו 10 בלוקים מ-50 (`grep "Progress @" training.log`). כל הערכים הם rolling-100ep.

| ep | Reward | Δ | Utility | Δ | Accuracy | Δ | Ticks/ep | π loss | V loss | Entropy H | Decisions/ep |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100  | +3.09 | — | 41.9% | — | 35.4% | — | 7458 | -0.0048 | 3.06 | **0.955** | 2.40 |
| 500  | +4.06 | -0.17 | 46.9% | -160.8% | 45.7% | -4.8% | 7919 | -0.0009 | 1.89 | 0.216 | 2.19 |
| 1000 | +3.91 | -0.13 | 46.3% | +0.2%  | 50.2% | -0.2% | 7488 | -0.0062 | 2.21 | 0.468 | 2.17 |
| 1500 | +4.10 | +0.26 | 45.9% | -0.4%  | 52.2% | +4.8% | 7757 | -0.0046 | 1.46 | 0.305 | 2.09 |
| 2000 | +4.21 | +0.34 | 45.2% | +4.3%  | 60.9% | +3.4% | 7517 | -0.0006 | 0.73 | 0.076 | 2.02 |
| 2500 | +3.77 | -0.28 | 39.6% | -3.4%  | 59.0% | +3.5% | 7947 | -0.0009 | 0.84 | 0.053 | 1.88 |
| 3000 | +3.96 | +0.24 | 43.5% | +3.8%  | 56.4% | -8.8% | 7193 | -0.0037 | 1.15 | 0.142 | 1.88 |
| 3500 | +4.75 | +0.59 | 55.0% | +7.5%  | 39.7% | -13.3% | 7231 | -0.0004 | 2.18 | **0.053** | 2.52 |
| 4000 | +4.40 | -0.10 | 52.8% | -1.5%  | 43.7% | +3.1% | 8102 | -0.0006 | 2.08 | 0.100 | 2.31 |
| 5000 | +4.28 | -0.04 | 46.8% | -0.3%  | 50.2% | -3.0% | 7748 | -0.0012 | 1.27 | 0.119 | 2.17 |

מקור: [`training.log:253-256, 1154-1157, 2273-2276, 3390-3393, 4509-4512, 5637-5640, 6750-6753, 7856-7859, 8977-8980, 11217-11220`](training_output_5k_run1/logs/training.log).

### מגמה
- **Reward**: עלייה ~+1.0 ב-100 הראשונים → **plateau מהיר** סביב +4.0 כבר מ-ep200 והלאה. בלוק 100→500: `+3.09→+4.06` (+0.97). מ-ep500 ועד ep5000 התנודה היא ±0.5 בלבד.
- **Utility ratio**: דפוס דומה — `41.9% → 46.9%` בין ep100→500, ואז plateau 40-55%. **שיא יחיד ב-ep3500 (55.0%)**, סוף הריצה ב-46.8%.
- **Accuracy (m/decisions)**: עלייה ל-~50% עד ep1000, ירידה זמנית ב-ep1100 (38.8%) ואז מטפס לאט עד 60.9% ב-ep2000. **חזר לרדת לאחר מכן** ל-39-50% בריצות 3000-5000, סוף 50.2%.
- **PPO trends:**
  - **`policy_loss` π**: שולי ושלילי לכל אורך, `-0.0048 → -0.0012`. כצפוי — actor כמעט לא משתפר אחרי הראשונים.
  - **`value_loss` V**: יורד יפה: `3.06 → 0.73 (ep2000)`. עלייה זמנית ל-2.18 ב-ep3500 (תנודה), חוזר ל-1.27 בסוף — critic מתייצב.
  - **`entropy` H**: **דרמטי** — `0.955 → 0.026 (ep3700)`. ירידה של 97% ל-3700 = **converged כמעט לחלוטין**. בסוף עלה חלקית ל-0.119, אך זה עדיין מתחת לסף בריא של ~0.5. **דגל אדום של premature convergence.**
- **Action distribution**: מ-ep600 והלאה — `A:99-100% R:0% N:0%`. הסוכן למד מדיניות "תמיד תקוף" — לא משתמש ב-RTB וב-NOOP.

### `last 10` סיום הריצה (`training.log:11227`)
```
Avg reward (last 10): 3.96
Avg accuracy (last 10): 86.7%
Avg utility ratio (last 10): 43.2%
```
ה-86.7% accuracy ב-last-10 לעומת 50.2% ב-rolling-100 → סטייה גדולה, מוסבר על ידי הריבוי של אפיזודות "פשוטות" עם 1 decision (`m=1/1`).

---

## 2. דגלים — ספירה והתפתחות

### סך כולל (`run_summary.txt:7-14`)
| Flag | Count | % of 5000 |
|---|---:|---:|
| `!CRASH` | **0** | — |
| `!ANOMALY` | **0** | — |
| `!L2-exhaust` | **0** | — |
| `!L2-fallback` | **0** | — |
| `!TIMEOUT` | 147 | 2.94% |
| `!noPPO` | 172 | 3.44% |

### לפי חלוני 1000 אפיזודות
| Window | TIMEOUT | noPPO |
|---|---:|---:|
| ep1-1000   | 28 | 34 |
| ep1001-2000 | 29 | 36 |
| ep2001-3000 | 27 | 37 |
| ep3001-4000 | 31 | 28 |
| ep4001-5000 | 32 | 34 |

**מסקנות:**
- שני הדגלים **לא משתפרים עם הזמן** — דפוס יציב ~3% לאורך כל הריצה. כלומר ה-RL לא מצמצם את שיעור ה-timeouts ככל שהוא לומד.
- **`!ANOMALY = 0`** — כל ה-validations עברו audit נקי (`Validation audit summary: 50 audits, 0 violations` ב-`run_summary.txt:136`). **אין באג ב-validation/oracle.**
- **`!noPPO`** מתרחש כשה-buffer ריק (אין trigger באפיזודה). סביר ל-1d episodes שבהן ה-RL אכן קיבל רק decision אחד — והוא הספיק לעדכן.

### Cluster alerts (`run_summary.txt:23-29`)
7 רצפי flagged episodes (≥3 ב-5 episodes):
- `ep0348-0352: TIMEOUT×3`
- `ep1017-1019: TIMEOUT×1, noPPO×2`
- `ep1516-1519: TIMEOUT×2, noPPO×1`
- `ep1950-1954, ep2101-2104, ep3314-3316, ep3847-3850`

ייתכן שאלו רצפי תרחישים קשים ספציפיים (cluster של seeds סמוכים), אך ללא דפוס שיטתי.

---

## 3. ⚠️ מקרים לחקירה — RL מנצח את ה-Oracle אך נספר כ-mismatch

**ההיפותזה אומתה ישירות.** מצאתי **137 אפיזודות (2.7%) עם `u≥100%`** בלוג, רובן עם `m` נמוך. זה הדפוס הקלאסי של "RL השיג יותר אבל סופר כשגיאה".

מקור: `grep -cE "u=[1-9][0-9]{2,}%" training.log → 137`.

### 3.1 — דוגמה מובהקת: ep3929 (RL השיג 1.5x מה-oracle)

**Summary line** (`training.log:8825-8826`):
```
ep3929  ag=2 tg=4[2e+2s]  L1:e=0/2+0iso s=1/2+0iso  L2:clean  split=2/4  ou=160/160
ep3929  RL=3d[A3 R0 N0] m=1/3  hit=3/4  RTB=Y  t=3096  r=+9.50  u=150%
```

**RL DECISION lines** ([`episode_3929.log`](training_output_5k_run1/logs/episode_3929.log)):
```
Tick 300  RL=ATTACK_0 Oracle=ATTACK_1 Match=✗ Reward=+0.00 (rl_u=80, oracle_u=80)
Tick 1150 RL=ATTACK_0 Oracle=NOOP    Match=✗ Reward=+1.00 (rl_u=80, oracle_u=0)
Tick 2600 RL=ATTACK_0 Oracle=ATTACK_0 Match=✓ Reward=+1.00 (rl_u=80, oracle_u=80)
Episode utility: achieved=240 / oracle=160 (ratio=1.50) → ep_reward=+7.50
```

**הקשר:** ה-oracle תכנן 160 utility (2 מטרות), RL השיג 240 (3 מטרות). `m=1/3` (33%) למרות שה-RL **עלה בביצוע על האוראקל**. ה-tick 1150 הוא בדיוק התרחיש של היפותזת המשתמש: `Match=✗` עם `rl_u=80 > oracle_u=0`.

### 3.2 — ep0061 (early training, m=0/3 + u=100%)

`training.log:170`:
```
ep0061  RL=3d[A3 R0 N0] m=0/3  hit=3/4  RTB=Y  r=+6.00  u=100%
```

**RL DECISIONs** ([`episode_0061.log`](training_output_5k_run1/logs/episode_0061.log)):
```
Tick 3350 RL=ATTACK_0 Oracle=ATTACK_1 Match=✗ Reward=+0.00 (rl_u=80, oracle_u=80)
Tick 3450 RL=ATTACK_1 Oracle=ATTACK_0 Match=✗ Reward=+0.00 (rl_u=80, oracle_u=80)
Tick 4100 RL=ATTACK_0 Oracle=NOOP    Match=✗ Reward=+1.00 (rl_u=80, oracle_u=0)
```

**אבחון:** Tick 3350+3450 הם **target-swap symmetric**. שני הסוכנים החליפו מטרות עם ה-oracle: רל-A תקף את מטרת אוראקל-B ולהפך. **utility זהה (80=80), אך נספר כ-2 mismatches.** Tick 4100 — RL גילה ותקף מטרה שאוראקל לא תכנן (ועליו +1.00 — ה-reward function כן מזהה זאת).

### 3.3 — דפוסים מוסכמים על פני 30 דגימות `u=100%`

לכל הדוגמאות (`training.log` שורות 170, 215, 228, 354, 527, 545, 596, 724, 915, 1108, 1120, 1136, 1223, 1475, 1541, 1559, 1599, 1622, 1646, 1703, 1732, 1760, 1781, 1797, 2064, 2104, 2233, 2244, 2257):

| מאפיין | תצפית |
|---|---|
| Match ratio (`m`) | חציון 1/3 (33%) — לפעמים אף `m=0/3` או `m=0/5` |
| Hit ratio (`hit`) | חציון 3/4 (75%) או 4/4 (100%) |
| Reward (`r`) | חציון +7 עד +9 — שכר חיובי מובהק |

**זאת בדיוק תופעת ה-target-swap** שהיפותזת המשתמש זיהתה. ה-RL מצליח להשיג utility מקסימלי על-ידי תקיפת מטרות חלופיות שאינן בתוכנית האוראקל, אך metric ה-accuracy `m` סופר כל סטייה כשגיאה גם כשה-utility זהה.

### 3.4 — האם ה-reward function בעצם מענישה את זה?

**לא.** מהבדיקה: ב-validation episodes (איפה שאני יכול לבחון בדיוק), reward עוקב את `(rl_u, oracle_u)`:
- `Match=✓`: +1.00
- `Match=✗ AND rl_u > oracle_u`: **+1.00** (כמו match!)
- `Match=✗ AND rl_u == oracle_u` (target-swap): +0.00 (ניטרלי)
- `Match=✗ AND rl_u < oracle_u`: -1.00

דוגמה לסיכום ב-6 validation episodes (mismatches בלבד):
| episode | rl>oracle | rl==oracle | rl<oracle |
|---|---:|---:|---:|
| ep1101 | 1 | 1 | 0 |
| ep2201 | **2** | 0 | 0 |
| ep3101 | 1 | 1 | 0 |
| ep4201 | 1 | 0 | 0 |

**מסקנה:** ה-reward function תקין. הבעיה היא רק ב-`m=X/Y` metric שמוצג ב-summary line ובהיסטוגרמה — הוא undercount של הביצוע האמיתי. **רק דווח, לא bug חישובי באימון.**

### 3.5 — Outlier קיצוני: ep0379 (`u=16000%`)

`training.log:880`:
```
!TIMEOUT ep0379  ag=2 tg=5[3e+2s]  ...  ou=240/0
!TIMEOUT ep0379  RL=2d[A2 R0 N0] m=0/2  hit=2/5  RTB=N  t=14383  r=+2.00  u=16000%
```

**זה לא RL>oracle אמיתי** — אלא division-by-near-zero artifact של `ou=240/0` (full_oracle_utility=0). ה-RL השיג 160 utility, החלוקה ב-ε הפכה את המנה ל-160x. החלון `ep0351-0400 u=365%` ב-`run_summary.txt:40` נגזר מ-outlier זה.

**דרוש fix קטן ב-logging**: להציג `u=N/A` במקום `u=16000%` כש-`full_oracle_utility==0`.

---

## 4. Recordings — מלאי ובחירה לבדיקה ויזואלית

### מלאי בפועל
| קטגוריה | כמות | ציפייה |
|---|---:|---:|
| `*_validation Recording*.jsonl` | 50 | 50 (`5000/100`) ✓ |
| `*_rl Recording*.jsonl` (regular) | 100 | 100 (`5000/50`) ✓ |
| `*_flagged_*_rl Recording*.jsonl` | 309 | ≈ 319 (147+172, חופפים) ≈ ✓ |
| **סה"כ** | **459** | |

**חישוב מהיר:** הפרש 459 - 50 - 100 = 309 flagged-replays. תואם ל-(147 TIMEOUT + 172 noPPO - חפיפות) ≈ 309.

### דוגמאות מומלצות לפתיחה ב-Panopticon

**RL > Oracle (תפיסות יוצאות דופן):**
1. `ep3929_rl Recording 064510 - 081145.jsonl` — היחיד שמצאתי עם `u=150%` חוקי וחד (`r=+9.50`, oracle plan=160, RL achieved=240).
2. `ep4201_rl Recording 064510 - 075347.jsonl` — validation episode עם `m=3/4 hit=3/4 u=75%` ועם RL DECISION של RL>oracle ב-tick 1200.
3. `ep4201_validation Recording 064510 - 084435.jsonl` — לזיהוי הplan המקורי של ה-oracle.

**Validation early/mid/late (השוואת התפתחות):**
4. `ep001_validation Recording 064510 - 081649.jsonl` — baseline ראשוני (u=33%, m=1/2)
5. `ep1101_validation Recording 064510 - 071853.jsonl` — אמצע-מוקדם (u=67%, m=0/2 — RL ניצח oracle כל הזמן)
6. `ep4201_validation Recording 064510 - 084435.jsonl` — שיא ההתכנסות (u=75%, m=3/4)

**Outlier לבדיקת division-by-zero:**
7. `ep0379_flagged_TIMEOUT_rl Recording 064510 - 104510.jsonl` — מקרה ה-`u=16000%` (`ou=240/0`)

**Cluster של failures (לבדיקה אם יש סיבת-שורש משותפת):**
8. `ep0023_flagged_TIMEOUT_rl Recording 064510 - 103224.jsonl`
9. `ep0055_flagged_TIMEOUT_rl Recording 064510 - 092113.jsonl`
10. `ep0107_flagged_TIMEOUT_rl Recording 064510 - 104510.jsonl`

כל הקבצים בתיקייה: [`training_output_5k_run1/recordings/`](training_output_5k_run1/recordings/).

---

## 5. מסקנות

### האם ה-RL לומד? **כן, אבל באופן חלקי וצר.**

**ראיות חיוביות:**
- Reward עלה מ-`+3.09 → +4.28` ב-100 האפיזודות הראשונות, ואז יציבות.
- Utility plateau יציב סביב 45-50% (לא 70%+ שצויין כיעד, אך עקבי).
- Critic value-loss ירד מ-3.06 → 1.27 (פי 2.4) — הloss conv טוב.
- 0 ANOMALY, 0 CRASH — ה-validation audit נקי לחלוטין.

**ראיות שליליות:**
- **Premature convergence ב-entropy**: H נופל מ-0.955 → 0.026 (ep3700). הסוכן הפסיק לחקור.
- **מדיניות מנוונת**: מ-ep600 והלאה, `Actions: A:99-100% R:0% N:0%`. RL בעצם מתקף "תמיד תקוף" ולעולם לא בוחר NOOP/RTB. זה אופטימלי בחלק מהתרחישים אבל לא תמיד.
- **שיעור TIMEOUT וnoPPO לא יורד עם הזמן** (~3% לאורך הריצה).

### בעיות עיקריות שזוהו

1. **`m=` accuracy metric undercount** (סעיף 3): ה-RL מצליח לעקוף את ה-oracle, אך כל סטייה מהתוכנית נספרת כשגיאה ב-`m=`. ה-reward function עצמו תקין (+1.00 ל-`rl_u > oracle_u`), אך ה-progress block ו-`run_summary.txt` משדרים "accuracy 50%" מטעה.
2. **u=16000% display bug** (סעיף 3.5): `ou=240/0` יוצר חלוקה באפסילון. דרוש patch ב-`train_full.py` לאזור utility ratio computation.
3. **Entropy collapse**: ייתכן שצריך להגדיל `entropy_coef` ב-`PPOConfig` או להוריד את `episode_reward_scale` (כרגע 5.0 — אולי 2.0-3.0 יהיה מאוזן יותר).
4. **NOOP/RTB אינם בשימוש**: ה-action space עומד לחלוטין על ATTACK. זה מצביע שהtarget-swap מתגמל מספיק שאין סיבה לבחור משהו אחר. בעולם של fuel-damage events (כשיופעלו) זה יהפוך לבעיה.

### הסבר אפשרי לפלטוי

ה-`probability=1.0` task construction (CLAUDE.md סעיף 4) מבטיח שכל מטרה נגישה תורמת `utility_per_target=80`. כל target-swap בין שני סוכנים שבשניהם reach=1 שומר על ה-utility. הסוכן לומד שה-reward אינו רגיש לזהות הספציפית של המטרה — רק לעובדה שהיא הותקפה. לכן `m=` יורד אבל `u=` נשמר ובמקרים קיצוניים עולה.

### המלצה

**לפני ריצה נוספת — לתקן את שני הבאגים בלוגינג:**

1. **`u=16000%` outlier**: ב-`train_full.py` בחישוב ה-utility ratio, להוסיף בדיקה `if oracle_u_full == 0: ratio = "N/A"`. אחרת קשה לסמוך על rolling windows כשmoutliers יחידים מטים את הממוצע (ראה window ep0351-0400 עם `u=365%`).

2. **`m=` accuracy interpretation**: שני אפשרויות:
   - **א.** להוסיף בproccess block שדה `m_strict` (target-match) ו-`m_utility` (utility-match: 1 אם `rl_u >= oracle_u`).
   - **ב.** להגדיר אחרת: m_match = `rl_u >= oracle_u`. דורש קוד change אחד ב-summary line.

**ואז להריץ עוד 5000 אפיזודות עם:**
- `entropy_coef` גבוה יותר (כרגע ברירת מחדל ב-PPO).
- אותו `seed=42` אם רוצים השוואה ישירה, אחרת `seed=43+` ל-baseline חדש.
- אותם feature toggles.

**לפני להפעיל `FUEL_DAMAGE_ENABLED=True`**: צריך לוודא שמדיניות ה-100% ATTACK תוכל להגיב נכון לאירועים שמחייבים RTB. כרגע יש סיכון שהsoonix תתקע ב-ATTACK גם כשcrucial לחזור.

---

*נוצר אוטומטית מתוך [`training_output_5k_run1/logs/`](training_output_5k_run1/logs/) ב-2026-05-06.*
