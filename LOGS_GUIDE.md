# מדריך לוגים — מערכת אימון MAPPO + BLADE + MATCH-AOU

> מסמך הסבר מפורט על כל הפלט שמוצג ונשמר במהלך ריצת אימון.
> נבנה תחנה-תחנה לפי הסדר שבו הדברים קורים בריצה.
>
> מקורות:
> - הקוד עצמו (`train_full.py` + מודולים נלווים)
> - `LOG_INVENTORY.md` — מיפוי מלא של כל הלוגים (881 שורות, נוצר ע"י Claude Code)
> - `run_capture.log` — דוגמה אמיתית מריצה של 5 אפיזודות

---

## תוכן עניינים

1. [Startup & Cleanup — ניקוי קבצים ישנים והקמת logging](#תחנה-1-startup--cleanup)
2. [Run-init Banner — הצגת קונפיגורציית הריצה](#תחנה-2-run-init-banner)
3. [Starting Training — מעבר ללולאת האפיזודות](#תחנה-3-starting-training)
4. [Episode Separator — תחילת אפיזודה חדשה](#תחנה-4-episode-separator)
5. [Scenario Generation — יצירת תרחיש לאפיזודה](#תחנה-5-scenario-generation)
6. [Validation Phase — ריצת אוראקל ל-baseline](#תחנה-6-validation-phase)
7. [RL Setup — הכנת הקרקע ל-RL phase](#תחנה-7-rl-setup)
8. [RL Simulation Loop — לב המערכת](#תחנה-8-rl-simulation-loop)
9. [Episode End — utility, PPO update, summary](#תחנה-9-episode-end)
10. [Progress Block — מעקב התקדמות לאורך הריצה](#תחנה-10-progress-block)
11. [Training Complete — סיום הריצה](#תחנה-11-training-complete)
12. [קבצי הפלט של הריצה](#תחנה-12-קבצי-הפלט-של-הריצה)
7. [RL Setup — הכנת התשתית ל-RL phase](#תחנה-7-rl-setup)

---

## רקע: איך פלט נוצר במערכת

לפני שצוללים לתחנות, חשוב להבין את **תשתית ה-logging** של המערכת.

### שלושה יעדים לכל הודעה

המערכת משתמשת ב-`logging` של פייתון. לכל קריאה ל-`logger.info(...)` או `logger.debug(...)` יש פוטנציאל ללכת ל-**3 יעדים במקביל**:

| יעד | היכן | רמה | תלוי ב-`--verbose`? |
|---|---|---|---|
| Console (stdout) | המסך | INFO רגיל / DEBUG אם verbose | כן |
| `training_output/logs/training.log` | קובץ master לכל הריצה | INFO רגיל / DEBUG אם verbose | כן |
| `training_output/logs/episode_NNNN.log` | קובץ נפרד לכל אפיזודה | תמיד DEBUG מלא | **לא** |

**משמעות מעשית:** גם בריצה ללא `--verbose`, **לכל אפיזודה** ייווצר קובץ DEBUG מלא ומפורט תחת `training_output/logs/`. הם יכולים להיות גדולים (אפיזודה אחת בריצת הדוגמה הגיעה ל-3.9 MB).

### פורמט אחיד

כל שורת לוג בנויה ככה:
```
2026-05-03 17:16:49,970 | INFO    | train_full | <התוכן>
```

ארבעה שדות מופרדים ב-`|`:
- **timestamp** — תאריך ושעה ברזולוציית מילישנייה
- **level** — `INFO`, `DEBUG`, `WARNING`, `ERROR` (ב-7 תווים מיושר לרוחב)
- **logger name** — מאיזה מודול הגיע. הרוב מ-`train_full`, חלק מ-`match_aou.utils.blade_utils.scenario_generator`, וכו'
- **message** — תוכן ההודעה

---

## תחנה 1: Startup & Cleanup

זה השלב **הראשון** שקורה בריצה — לפני שהוא מציג שום banner על המסך, הוא מנקה אחרי הריצה הקודמת.

### מתי זה מופיע

- **על הקונסול:** *לעולם לא*. גם עם `--verbose`. (הסבר למה למטה.)
- **ב-`training.log`:** רק עם `--verbose=True`.
- **בתיקיות `recordings/` ו-`scenarios/`:** התוצאה הפיזית — תיקיות ריקות לפני שהאפיזודה הראשונה רצה.

### מה קורה בפועל

הקוד (סביב `train_full.py:2438-2475`) מבצע 5 פעולות בסדר הזה:

#### 1. ניקוי `training_output/recordings/`
נמחק כל קובץ `.jsonl`. הסיבה: כל ריצה מתחילה נקייה כדי שתוכל לפתוח את Panopticon ולא לבלבל בין ריצות שונות.

#### 2. ניקוי `training_output/scenarios/`
נמחק כל `episode_*_scenario.json`. אותו רציונל — לא לבלבל בין ריצות.

#### 3. ניקוי `training_output/logs/episode_*.log`
נמחקים קבצי הלוג הפר-אפיזודה הישנים.

> **הערה חשובה:** רק `episode_*.log` נמחקים. הקבצים `training.log`, `run_summary.txt`, `highlights.txt` נכתבים עם mode `'w'` ולכן **נדרסים בעצמם** בעת היצירה (מתחלפים אוטומטית בריצה החדשה).

#### 4. תיקיית `models/` *לא* מנוקה
ה-checkpoints מצטברים בין ריצות. כדי להתחיל לגמרי נקי, **חובה למחוק ידנית** את ה-`.pt` הישנים מ-`training_output/models/` לפני הריצה.

#### 5. הקמת logging
- **רוט-לוגר** מוקם עם פורמטר `%(asctime)s | %(levelname)-7s | %(name)s | %(message)s`
- **`StreamHandler`** מתחבר לקונסול (רמת INFO רגיל, DEBUG עם verbose)
- **`FileHandler`** מתחבר ל-`training.log` (אותה רמה כמו הקונסול)
- ה-**handler הפר-אפיזודה** ייווצר וימחק דינמית בתחילת/סוף כל אפיזודה (ראה תחנה 4)

### למה הקונסול שקט גם עם `--verbose`

הניקוי **קודם** להתחברות ה-`StreamHandler` של הקונסול. כלומר ברגע שהקוד קורא `logger.debug("Removed old recording: ...")`, אין עדיין handler מחובר שיכוון את ההודעה לקונסול. ההודעה הולכת רק ל-`training.log` (שכן נפתח לפני הניקוי).

תרגום מעשי: גם עם `--verbose=True`, הראשון על המסך זה ה-banner של תחנה 2. הניקוי לעולם לא נראה על המסך.

### למה `training.log` שקט בלי `--verbose`

שורות ה-"Removed old..." הן ברמת DEBUG. כש-`--verbose` כבוי, ה-handler של `training.log` רץ ברמת INFO ולכן הוא מסנן את כל ה-DEBUG החוצה. במצב הזה הניקוי קורה **בשקט מוחלט** — אין שום ראיה במסמכי הלוג שזה בכלל קרה.

### דוגמה (רק עם `--verbose`, רק ב-`training.log`)

```
2026-05-03 17:16:49,xxx | DEBUG | train_full | Removed old recording: training_output\recordings\ep001_rl Recording 064510 - 081341.jsonl
2026-05-03 17:16:49,xxx | DEBUG | train_full | Removed old recording: training_output\recordings\ep001_validation Recording 064510 - 081649.jsonl
...
2026-05-03 17:16:49,xxx | DEBUG | train_full | Removed old scenario: training_output\scenarios\episode_0000_scenario.json
...
```

### הערות מעשיות לקראת ריצת 5000

- **לפני הריצה:** אם אתה רוצה לשמור recordings מריצה קודמת (למשל לצורך השוואה) — **העבר אותם החוצה לפני** התחלת ריצה חדשה. הם יימחקו אוטומטית.
- **`models/` הוא חריג:** אם אתה רוצה התחלה לגמרי נקיה, מחק ידנית את ה-`.pt` הישנים. הקוד לא יעשה זאת בעצמו.
- **בלי `--verbose`** (וזה מה שמומלץ ב-5000) — הניקוי קורה בשקט. הדרך לוודא שזה עבד היא פשוט לפתוח את `training_output/recordings/` ולראות שהיא ריקה לפני שהאפיזודה הראשונה מתחילה לכתוב לתוכה.

---

## תחנה 2: Run-init Banner

זה הפלט **הראשון שתראה על המסך**. ה-banner מודפס פעם אחת בריצה, מיד אחרי שהקונסול handler נדבק. **הכל ברמת INFO** — מופיע גם בלי `--verbose`.

### מתי זה מופיע

- **על הקונסול:** תמיד, פעם אחת בתחילת הריצה.
- **ב-`training.log`:** תמיד.
- **ב-`episode_NNNN.log`:** לא מופיע שם — זה לפני שמתחילות אפיזודות.

### מה תראה (דוגמה אמיתית מ-`run_capture.log`)

```
2026-05-03 17:16:49,970 | INFO | train_full | ======================================================================
2026-05-03 17:16:49,970 | INFO | train_full | Full RL Training — MAPPO + BLADE + MATCH-AOU
2026-05-03 17:16:49,970 | INFO | train_full | ======================================================================
2026-05-03 17:16:49,970 | INFO | train_full | Base scenario:     data/scenarios/strike_training_4v5.json
2026-05-03 17:16:49,971 | INFO | train_full | Vary scenarios:    True
2026-05-03 17:16:49,971 | INFO | train_full | Episodes:          5
2026-05-03 17:16:49,971 | INFO | train_full | RL trigger:        event-driven (discovery + fuel damage)
2026-05-03 17:16:49,971 | INFO | train_full | Discovery scan:    every 50 ticks
2026-05-03 17:16:49,971 | INFO | train_full | Max ticks:         14400
2026-05-03 17:16:49,971 | INFO | train_full | Max agents:        5
2026-05-03 17:16:49,971 | INFO | train_full | Learning rate:     0.0003
2026-05-03 17:16:49,971 | INFO | train_full | Seed:              42
2026-05-03 17:16:49,971 | INFO | train_full | Fuel damage:       True
2026-05-03 17:16:49,971 | INFO | train_full | Include SAMs:      False
2026-05-03 17:16:49,972 | INFO | train_full | Allowed aircraft:  all (from pool)
2026-05-03 17:16:49,972 | INFO | train_full | Stretch ratio:     0.5
2026-05-03 17:16:49,972 | INFO | train_full | Validate every:    1 episodes
2026-05-03 17:16:49,972 | INFO | train_full | Record every:      1 episodes (0=never)
2026-05-03 17:16:49,972 | INFO | train_full | Verbose console:   True
2026-05-03 17:16:49,972 | INFO | train_full | DEBUG force flags: ['l2-fallback', 'timeout']
2026-05-03 17:16:49,973 | INFO | train_full | Output dir:        C:\...\training_output

2026-05-03 17:16:50,011 | INFO | scenario_generator | ScenarioGenerator ready: base=strike_training_4v5.json, aircraft_pool=[...], facility_pool=[...]
2026-05-03 17:16:50,012 | INFO | train_full | Time-feasibility cap: 1195 km one-way (slowest=KC-135R Stratotanker 854 km/h, ticks=14400, safety=0.3) [auto]
2026-05-03 17:16:50,013 | INFO | train_full | ScenarioGenerator: aircraft_pool=[...], facility_pool=[...], aircraft=(2-3), facilities=(2-4), red_airbases=(3-5), max_dist=2500.0km, vary_base=False

2026-05-03 17:16:50,013 | INFO | train_full | --- Setting up BLADE environment ---
2026-05-03 17:16:50,013 | INFO | train_full | BLADE registered max_episode_steps: 2000
2026-05-03 17:16:50,022 | INFO | train_full | BLADE env ready: duration=14400, max_episode_steps=14400, start_time=1699073110, current_time=1699073110

2026-05-03 17:16:50,023 | INFO | train_full | --- Creating RL components (MAPPO) ---
2026-05-03 17:16:50,037 | INFO | train_full | ActorCriticNetwork: actor=12,549 params, critic=27,649 params
2026-05-03 17:16:50,037 | INFO | train_full |   Actor:  obs[30] → 128 → 64 → logits[5]
2026-05-03 17:16:50,040 | INFO | train_full |   Critic: global[150] → 128 → 64 → V(s)[1]
2026-05-03 17:16:53,967 | INFO | train_full | PPOTrainer ready

2026-05-03 17:16:53,967 | INFO | train_full | ======================================================================
2026-05-03 17:16:53,967 | INFO | train_full | Starting Training
2026-05-03 17:16:53,968 | INFO | train_full | ======================================================================
```

### פירוק לפי בלוקים

ה-banner נשבר ל-**4 בלוקים** עם תפקידים שונים:

#### 2.1 — Config dump

ה-echo המלא של פרמטרי הריצה. **המטרה: ראיה משפטית.** אם בעוד שבוע תרצה לדעת מאיזה seed רצה ריצה X, או באיזה learning-rate — תפתח את `training.log` הראשון של אותה ריצה והכל שם.

ארבע שורות שכדאי להבין לעומק:

##### `Vary scenarios: True`
זה **המאפיין המרכזי של איך אתה מאמן.** מחליט אם תאמן על תרחיש קבוע אחד או על מגוון תרחישים שונים באפיזודות שונות.

**`Vary scenarios: False`**
```python
ep_scenario_path = args.scenario   # תמיד אותו קובץ
```
כל 5000 האפיזודות יקבלו את **אותו** `strike_training_4v5.json` בדיוק. אותם 4 מטוסים, אותן 5 מטרות, באותם מיקומים. הסוכנים ילמדו לשנן תרחיש אחד. שימושי לדיבוג, **לא טוב לאימון** — ה-policy יוכל רק את התרחיש הספציפי הזה ויכשל מיד על אחרים.

**`Vary scenarios: True`** (מה שתפעיל ב-5000)

`ScenarioGenerator` יוצר תרחיש טרי לכל אפיזודה. מה שמשתנה (בריצת ברירת המחדל):
- מספר המטוסים הכחולים: 2 או 3
- סוגי המטוסים (B-2 / F-35 / KC-135R / F-16) — sampling מתוך ה-pool
- מספר המתקנים האדומים: 2-4
- מספר ה-airbases האדומים: 3-5
- **מיקומי כל המטרות והמתקנים** — אקראיים בתוך bounds
- כמה מהמטרות "stretch" (רחוקות, גבוליות מבחינת fuel) — לפי `stretch_ratio=0.5`

מה לא משתנה:
- מיקום הבסיס הכחול (אלא אם הוספת `--vary-base`)
- ה-pool של המטוסים והמתקנים
- ה-bounds של אזור ההצבה

**דטרמיניזם:** הקוד משתמש ב-`seed=args.seed + episode` — אם תריץ פעמיים עם אותו `--seed`, תקבל בדיוק אותם תרחישים בכל אפיזודה. **קריטי לרפרודוקטיביות מחקר.**

**עלות זמן:** ייצור התרחיש קורה **בכל אפיזודה** (לא ב-startup). כולל דגימת מספרים, הצבת מטרות עם rejection sampling, ולידציית discovery chain (עד 20 ניסיונות חוזרים), כתיבת JSON לדיסק, וטעינתו ב-BLADE. בדרך כלל 30-200ms לאפיזודה. על פני 5000 = ~8 דקות overhead נטו. את הביטוי בלוגים נראה בתחנה 5 (Scenario generation).

##### `RL trigger: event-driven (discovery + fuel damage)`
ה-RL **לא** מקבל החלטה כל tick. הוא מחכה לאירוע — או סוכן גילה מטרה חדשה (discovery), או ספג נזק לדלק (fuel damage). זה הבסיס לכל ההיגיון של למה יש כל-כך מעט "RL DECISION" בלוגים.

##### `Discovery scan: every 50 ticks`
סריקת ראדאר מתבצעת כל 50 ticks. גילוי **לעולם לא יקרה** ב-tick 1247 — רק ב-1250, 1300 וכו'. זכור את זה כשתסתכל על העיתוי של DISCOVERY events בתחנה 8.

##### `DEBUG force flags: ['l2-fallback', 'timeout']`
**זה כלי בדיקה למפתחים, לא לאימון אמיתי.**

מה זה עושה? קוראים את הקוד ב-`train_full.py:2666-2669`:
```python
ep_force = set()
if "timeout" in debug_force_flags_set and (episode % 5 == 0):
    ep_force.add("timeout")
if "l2-fallback" in debug_force_flags_set and (episode % 5 == 1):
    ep_force.add("l2-fallback")
```

ההיגיון:
- אם ביקשת `timeout` → באפיזודות **0, 5, 10, 15...** (0-indexed) הקוד מזריק את ה-flag `timeout` למטא-דאטא של האפיזודה, גם אם האפיזודה הסתיימה תקין.
- אם ביקשת `l2-fallback` → באפיזודות **1, 6, 11, 16...** מוזרק `l2-fallback`.
- אם לא העברת את ה-flag הזה כלל → הפיצ'ר רדום.

**איך מפעילים:**
```bash
python train_full.py --debug-force-flags timeout,l2-fallback
# או רק אחד:
python train_full.py --debug-force-flags timeout
```

**למה זה קיים בכלל?**
1. לוודא שמסלול הקוד של flagged-episode עובד, בלי לחכות 200 אפיזודות עד שיקרה timeout באמת.
2. לוודא שטבלאות הסיכום (`run_summary.txt`, `highlights.txt`) מטפלות נכון בכל סוגי ה-flags.
3. לדבג לוגיקת flagged-replay (replay של אפיזודה flagged כדי להפיק לה recording).

**למה לא להשתמש בזה ב-5000:**
ב-5000, **flags צריכים להיות אינדיקטור אמיתי לבעיות:**
- `!TIMEOUT` → האפיזודה הגיעה ל-max_ticks בלי שכל הסוכנים חזרו לבסיס
- `!L2_FALLBACK` → ה-RL בחר action שלא היה תקף ונפל ל-fallback
- `!ANOMALY` → תקפו מטרה שלא הייתה בטווח

אילוץ flags **מזהם את הסטטיסטיקות.** `run_summary.txt` ידווח על אלפי flags שלא משקפים בעיות אמיתיות, וה-Progress block יציג למשל `Flags(window): TIMEOUT=20%` שיסטה אותך מהאמת.

**ב-5000 פשוט תשמיט את הארגומנט הזה לגמרי** (או תיתן לו string ריק, שזה ברירת המחדל).

#### 2.2 — ScenarioGenerator setup

```
ScenarioGenerator ready: base=strike_training_4v5.json, aircraft_pool=[...]
Time-feasibility cap: 1195 km one-way (slowest=KC-135R Stratotanker 854 km/h, ticks=14400, safety=0.3) [auto]
ScenarioGenerator: aircraft=(2-3), facilities=(2-4), red_airbases=(3-5), max_dist=2500.0km, vary_base=False
```

הגנרטור נבנה **פעם אחת** בתחילת הריצה ויעשה sample חדש לכל אפיזודה (כשמופעל `--vary-scenarios`).

##### `Time-feasibility cap: 1195 km`
חישוב **אוטומטי** של מקסימום מרחק פרקטי שאפשר להציב מטרה. הנוסחה: המטוס האיטי ביותר ב-pool (KC-135R, 854 km/h) צריך להגיע ולחזור תוך max_ticks (14400 ticks ≈ 4 שעות) עם safety margin של 30%. זה מבטיח שכל מטרה שתיווצר באמת ניתנת להגעה. ה-`[auto]` אומר שהמשתמש לא דרס את זה ידנית.

##### `max_dist=2500.0km`
זה ה-bound הקשיח של אזור ההצבה (ברירת מחדל בקונפיגורציה). שונה מ-time-feasibility cap — `max_dist` הוא bound גיאוגרפי גס, time-feasibility cap הוא חישוב מתקדם שיכול להגביל יותר.

##### `vary_base=False`
בסיס הכחול נשאר באותו מיקום בכל אפיזודה. רק המטוסים שיוצאים ממנו והמטרות מסביבו משתנים.

#### 2.3 — BLADE environment setup

```
--- Setting up BLADE environment ---
BLADE registered max_episode_steps: 2000
BLADE env ready: duration=14400, max_episode_steps=14400, start_time=1699073110, current_time=1699073110
```

שתי שורות שלכאורה סותרות:
- שורה ראשונה: BLADE רושמת את עצמה ל-gymnasium עם **2000** ticks (ערך ברירת מחדל פנימי של BLADE).
- שורה שנייה: אחרי שעולה ה-scenario, מתעדכן ל-**14400** ticks (כפי ש-`--max-ticks` הגדיר).

ההפרש הזה נורמלי. רק אומר ש-BLADE נרשמת קודם, ואז ה-scenario מעדכן את האורך הסופי.

**הערה על UserWarning של gymnasium:**
ייתכן שתראה ב-stderr הודעת warning בנוסח:
```
WARN: The obs returned by the `reset()` method is not within the observation space.
```
זה רעש לא מזיק — gymnasium עושה passive validation על ה-observation וזה נכשל כי ה-observation הראשון מ-BLADE בא ב-format שונה ממה שהיא מצפה. **לא משפיע על הריצה.**

#### 2.4 — Network architecture (MAPPO)

```
ActorCriticNetwork: actor=12,549 params, critic=27,649 params
  Actor:  obs[30] → 128 → 64 → logits[5]
  Critic: global[150] → 128 → 64 → V(s)[1]
PPOTrainer ready
```

זה ה-**MAPPO architecture summary**. שדה-שדה:

| שדה | משמעות |
|---|---|
| `actor=12,549 params` | ה-policy network שמייצר actions. **decentralized** — כל סוכן מקבל רק את ה-observation שלו. |
| `critic=27,649 params` | ה-value network. **centralized** — רואה את כל ה-state הגלובלי. זה ה-CTDE pattern של MAPPO (Centralized Training, Decentralized Execution). |
| `obs[30]` | וקטור התצפית של סוכן בודד. 30 features, מיוצר על ידי `observation_builder.py`. |
| `global[150]` | 30 features × MAX_AGENTS=5. גם אם בריצה הספציפית יש רק 2 סוכנים, ה-critic מקבל וקטור באורך 150 עם padding. |
| `logits[5]` | מרחב הפעולות. 5 פעולות אפשריות. |
| `PPOTrainer ready` | הקריטריון/אופטימייזר/buffer בנויים. |

### סיכום תחנה 2

ה-banner הזה הוא **ה-snapshot של תחילת הריצה**: מה הקונפיגורציה, מה ה-network, מה ה-environment. הוא מודפס פעם אחת ולעולם לא חוזר על עצמו.

**בריצת 5000 ללא `--verbose`:** *בדיוק את אותו banner*. כולו INFO. אם תכוון `grep "INFO" training.log | head -50` תקבל את כל ה-banner הזה ועוד כמה שורות אחריו.

**מה ישתנה ב-5000:** רוב הערכים (Episodes, Seed וכו') יהיו אחרים. זאת הסיבה שה-banner חשוב — הוא מבדיל בין ריצה לריצה.

---

## תחנה 3: Starting Training

זה ה-**marker של מעבר** משלב הקמת המערכת אל לולאת האימון. תחנה קצרה במיוחד כי הפלט עצמו מינימלי.

### מתי זה מופיע

- **על הקונסול:** תמיד, פעם אחת, מיד אחרי `PPOTrainer ready` של תחנה 2.
- **ב-`training.log`:** תמיד.
- **ב-`episode_NNNN.log`:** לא — זה עדיין לפני האפיזודה הראשונה.

### מה תראה

```
2026-05-03 17:16:53,967 | INFO | train_full | PPOTrainer ready
2026-05-03 17:16:53,967 | INFO | train_full |
======================================================================
2026-05-03 17:16:53,967 | INFO | train_full | Starting Training
2026-05-03 17:16:53,968 | INFO | train_full | ======================================================================
```

זה הכל. שלוש שורות, ברמת INFO. **לא תלוי ב-`--verbose`.**

### מה זה בעצם

ב-`train_full.py:2592-2594`. ברגע שאתה רואה את "Starting Training", הקוד נכנס ל-`for episode in range(args.episodes)` — מכאן והלאה כל מה שתראה הוא **חזרתי לכל אפיזודה.**

### למה זה חשוב

##### 1. זה ה-anchor שלך לחפש את תחילת לולאת האפיזודות

אם תרצה לדלג על כל ה-banner ולקפוץ ישר לאפיזודה הראשונה ב-`training.log` של 5000 אפיזודות, הדרך הכי מהירה:
```bash
grep -n "Starting Training" training.log
```
זה ייתן לך את מספר השורה. כל מה שאחריו הוא לולאת אפיזודות.

##### 2. ההפרש הזמני בין השורה הזו לבין `Episode 1/N` קטן מאוד

בדוגמה שלנו, "Starting Training" יצא ב-`17:16:53.967` ו-"Episode 1/5" יצא ב-`17:16:53.968` — מילישנייה אחת. אין שום עיכוב משמעותי בין שתי השורות האלה.

##### 3. הזמן הכבד נמצא **לפני** "Starting Training"

בדוגמה שלנו, ה-network נבנה ב-`17:16:50.040` אבל `PPOTrainer ready` הגיע רק ב-`17:16:53.967` — **כמעט 4 שניות שתיקה.** הסיבה: torch ו-`PPOTrainer` initialization (יצירת אופטימייזר, buffer, networks). זה לא קשור ל-`--vary-scenarios` או לטעינת BLADE — זה התשתית של PPO.

אם אצלך השתיקה הזו ארוכה משמעותית מ-4 שניות בריצת ה-5000, שווה לבדוק שלא קרה משהו בעייתי באתחול.

### הערות מעשיות לקראת ריצת 5000

- **לא משתנה כלום ב-5000.** זאת אותה שורה שהופיעה בריצה של 5 אפיזודות. רק תופיע פעם אחת.
- **לא צריך לבחון שום דבר כאן.** זה רק marker של מעבר.
- **ה-anchor לחיפוש:** `grep "Starting Training"` להגיע מהר ללולאת האפיזודות.

---

## תחנה 4: Episode Separator

זאת השורה ש**מסמלת התחלת אפיזודה חדשה**. תחנה קצרה, אבל חשובה כי היא ה-anchor לחיפוש בלוגים, ובמקביל מתבצעות מאחורי הקלעים פעולות חשובות.

### מתי זה מופיע

- **על הקונסול:** רק עם `--verbose`. בלי verbose — שקט מוחלט בין אפיזודות (תראה רק את שורות הסיכום בסוף כל אפיזודה).
- **ב-`training.log`:** רק עם `--verbose` (DEBUG).
- **ב-`episode_NNNN.log`:** תמיד (תמיד DEBUG מלא).

### מה תראה (מ-`run_capture.log` שורות 292-295)

```
2026-05-03 17:16:53,968 | DEBUG | train_full |
==================================================
2026-05-03 17:16:53,968 | DEBUG | train_full | Episode 1/5
2026-05-03 17:16:53,968 | DEBUG | train_full | ==================================================
```

זה הכל. שלוש שורות בלבד, ברמת DEBUG.

### מה קורה ברקע

הקוד ב-`train_full.py:2609-2611`. אבל מאחורי השורות הוויזואליות האלה, קורה **דבר חשוב מאוד שאתה לא רואה**: יצירת ה-handler הפר-אפיזודה.

#### יצירת `episode_NNNN.log`

ב-`train_full.py:2657`, מיד אחרי ה-banner של "Episode 1/5", הקוד מבצע:

```python
ep_handler = logging.FileHandler(
    f"training_output/logs/episode_{episode:04d}.log",
    mode='w'
)
ep_handler.setLevel(logging.DEBUG)  # תמיד DEBUG מלא
ep_handler.setFormatter(...)
root.addHandler(ep_handler)
```

**מה זה אומר:**
- בכל אפיזודה נוצר קובץ חדש `episode_NNNN.log` (4 ספרות, padded).
- הקובץ הזה מקבל **כל** הודעת `logger.debug(...)` שתבוצע במהלך האפיזודה — לא משנה אם הקונסול ב-INFO או DEBUG.
- ה-handler מוסר **לפני שורות הסיכום של האפיזודה** (תחנה 9). הסיבה: שורות הסיכום צריכות להגיע ל-console + `training.log`, אבל לא צריכות להיכנס לקובץ הפר-אפיזודה (שתפקידו לתעד את התוכן של האפיזודה עצמה, לא את הסיכום שלה).
  - **המסלול הרגיל:** הסרה ב-`train_full.py:2715-2716` — אחרי שהאפיזודה הסתיימה בהצלחה, לפני הדפסת הסיכום.
  - **מסלול ה-CRASH:** אם האפיזודה זרקה exception, ההסרה מתבצעת ב-`train_full.py:2708-2709` (בתוך ה-`except`), לפני `continue` לאפיזודה הבאה.

> **הקובץ הזה הוא ה-"firehose"** של האפיזודה — מכיל את כל הפרטים, כולל רעש Pyomo. הוא המצרף של תחנות 4-9. ראה הסעיף [הקבצים של הריצה](#הקבצים-של-הריצה) בסוף המסמך לדוגמה מלאה ופירוט.

#### הקדמה לחישובים שיקרו אחר כך

מיד אחרי ה-banner, באותו תא של הקוד (שורות 2647-2670), נעשים שלושה חישובים שמשפיעים על האפיזודה:

```python
should_record = (args.record_every > 0 and episode % args.record_every == 0)
is_validation_episode = (args.validate_every > 0 and episode % args.validate_every == 0)

ep_force = set()  # כפיית debug flags (תחנה 2.1)
if "timeout" in debug_force_flags_set and (episode % 5 == 0):
    ep_force.add("timeout")
if "l2-fallback" in debug_force_flags_set and (episode % 5 == 1):
    ep_force.add("l2-fallback")
```

המשמעות:
- **`should_record`** — האם האפיזודה הזו תייצר קובץ recording? תלוי ב-`--record-every`. אם הוא 50, אז רק אפיזודה 0, 50, 100... ייצרו recording.
- **`is_validation_episode`** — האם תרוץ לפני ה-RL ריצת validation עם oracle בלבד? תלוי ב-`--validate-every`.
- **`ep_force`** — flags שנכפים על האפיזודה הזו (תחנה 2.1).

אף אחד מאלה לא מודפס ל-log כשורה נפרדת — אבל הם משפיעים על מה שיקרה הלאה. אם האפיזודה היא validation, מיד אחרי ה-banner תראה את `--- Validation run (oracle only, no RL) ---` (תחנה 6).

### בלגן האינדקסים: 0-indexed לעומת 1-indexed

זה **מקור בלבול נפוץ** ושווה להבין אותו בבירור עכשיו, כי הוא יחזור לאורך כל המסמך.

#### למה יש שתי ספירות בכלל

ב-Python, לולאות מתחילות מ-0:
```python
for episode in range(5):    # episode = 0, 1, 2, 3, 4
    ...
```

אבל בני אדם סופרים מ-1: "אפיזודה ראשונה, שנייה, שלישית". הקוד הנוכחי בחר באופן לא-עקבי — בחלק מהמקומות הציג 1-indexed (לעיני המשתמש), בחלק 0-indexed (כמו במשתנה הפנימי).

#### בריצת 5 אפיזודות, איך נראים האינדקסים בפועל

| איטרציה של הלולאה | `episode` (משתנה פנימי) | הצגה למשתמש (`episode + 1`) |
|---|---|---|
| 1 | 0 | 1 |
| 2 | 1 | 2 |
| 3 | 2 | 3 |
| 4 | 3 | 4 |
| 5 | 4 | 5 |

#### מה כל קובץ/banner משתמש

| מקום | פורמט | מה מקבל איטרציה ראשונה |
|---|---|---|
| Banner על המסך | `Episode {episode+1}/{N}` | `Episode 1/5` |
| `episode_NNNN.log` (קובץ הלוג) | `episode_{episode+1:04d}` | `episode_0001.log` |
| `ep_NNN_rl Recording...jsonl` | `ep{episode+1:03d}` | `ep001_rl Recording...jsonl` |
| `checkpoint_epN.pt` | `checkpoint_ep{episode+1}` | `checkpoint_ep1.pt` |
| Per-episode summary line (תחנה 9) | `ep{episode+1:04d}` | `ep0001 [VAL]...` |
| **`episode_NNNN_scenario.json`** (התרחיש) | `episode_{episode:04d}` | **`episode_0000_scenario.json`** ← 0-indexed! |
| **`% N == 0` ב-debug-force-flags** | משתמש ב-`episode` הפנימי | **איטרציה ראשונה (`episode 0`) נופלת לזה** ← 0-indexed! |

שתי השורות התחתונות הן **חריגות** מהמוסכמה. כל השאר 1-indexed, אבל קובץ התרחיש ובדיקת ה-modulo משתמשים ב-`episode` הפנימי (0-indexed).

#### הקשר לדוגמה הקיימת

בריצת ה-5 אפיזודות עם `--debug-force-flags timeout,l2-fallback`:

| Banner בלוגים | יצר קובץ תרחיש | flags שנדלקו | גודל קובץ הלוג |
|---|---|---|---|
| `Episode 1/5` | `episode_0000_scenario.json` | `!TIMEOUT` (כפוי, מ-`% 5 == 0`) | 36KB |
| `Episode 2/5` | `episode_0001_scenario.json` | `!L2-fallback` (כפוי, מ-`% 5 == 1`) | 38KB |
| `Episode 3/5` | `episode_0002_scenario.json` | (כלום) | 38KB |
| `Episode 4/5` | `episode_0003_scenario.json` | `!TIMEOUT` **אמיתי** (הגיע ל-tick 14382) | **3.9MB** |
| `Episode 5/5` | `episode_0004_scenario.json` | (כלום) | 41KB |

**שים לב להבחנה ב-`Episode 4/5`:** זה היה `!TIMEOUT` **אמיתי** — האפיזודה באמת הגיעה כמעט ל-`max_ticks` (14382 מתוך 14400). זה לא היה כפוי, כי `3 % 5 ≠ 0`. בגלל זה הקובץ ענקי — תמיד DEBUG מלא לכל tick שנכלל.

לעומת זאת, `!TIMEOUT` של `Episode 1/5` היה מזויף — האפיזודה הסתיימה תקין ב-tick 5293, רק שה-flag נכפה.

> **תזכורת לעתיד:** אינקונסיסטנציית האינדקסים היא משהו שכדאי לתקן בקוד בעתיד (אם תרצה הגיון אחיד — כולם 1-indexed, או כולם 0-indexed). זה דיון נפרד שכדאי לעשות אחרי הסיור הזה.

### הערות מעשיות לקראת ריצת 5000

##### 1. בלי `--verbose` הקונסול שקט בין אפיזודות
תראה ב-stdout רק את 2 שורות הסיכום בסוף כל אפיזודה (תחנה 9), לא את ה-banner של ההתחלה. זה מכוון — 5000 banners יציפו את הקונסול.

##### 2. כל `episode_NNNN.log` ייווצר לכל אפיזודה
בריצה של 5000, **תקבל 5000 קבצים בתיקיית `logs/`**. הגודל הממוצע יהיה כנראה ~50KB-200KB (מהדגמה: ep1, ep2, ep3, ep5 היו 36-41KB). אפיזודות שיגיעו ל-`!TIMEOUT` אמיתי יכולות להיות מגה-בייטים. סך הכל סביר — בערך 1-2 GB של logs נטו, אבל ספציפית **5000 קבצים בתיקייה אחת** יכול להאט את ה-Explorer/Finder. אם אתה רוצה שזה יהיה נוח לעיון, אולי שווה לחשוב על subdirectories (כמו `logs/ep_0001-1000/`). זה שינוי קוד קטן שאפשר לעשות לפני הריצה — אבל לא חובה.

##### 3. ה-anchor לחיפוש
עם `--verbose`:
```bash
grep -n "Episode [0-9]*/[0-9]*$" training.log
```

בלי verbose (התרחיש המעשי שלך ב-5000), השורות האלה לא יהיו ב-`training.log`. תצטרך לחפש לפי שורות הסיכום:
```bash
grep -nE "ep[0-9]{4} \[(VAL|TRN)\]" training.log
```

---

## תחנה 5: Scenario Generation

זה השלב **הראשון שקורה בתוך כל אפיזודה** — `ScenarioGenerator` יוצר תרחיש חדש ובלעדי לאפיזודה הזו, ו-BLADE טוענת אותו. כל זה תלוי ב-`--vary-scenarios=True`. בלי הדגל הזה, השלב הזה לא רץ והקוד פשוט שב לתרחיש הקבוע.

### מתי זה מופיע

- **על הקונסול:** רק עם `--verbose` (כל ההודעות פה DEBUG).
- **ב-`training.log`:** רק עם `--verbose`.
- **ב-`episode_NNNN.log`:** תמיד.
- **בתיקיית `scenarios/`:** **תמיד** נכתב קובץ JSON, גם בלי verbose.

### מה תראה (מ-`run_capture.log`, אפיזודה 1)

```
17:16:53,969 | DEBUG | scenario_generator |   include_sams=False → removed all SAM facilities
17:16:53,969 | DEBUG | scenario_generator |   Stretch zone collapsed by time-feasibility cap (stretch_max=1195 ≤ stretch_min=1560)
17:16:53,969 | DEBUG | scenario_generator | Discovery chain: easy relocated=2/3 isolated=0, stretch relocated=0/0 isolated=0 (min fleet radar=93 km)
17:16:53,969 | DEBUG | scenario_generator | Reachability audit: 3/3 targets reachable by at least one agent
17:16:54,005 | DEBUG | train_full | Reloaded scenario from training_output\scenarios\episode_0000_scenario.json
17:16:54,006 | DEBUG | train_full |   Generated scenario: episode_0000_scenario.json
```

שש שורות לאפיזודה הזו. שים לב — **בכל אפיזודה התוכן יהיה שונה.** השורות `Stretch zone collapsed`, `Stretch targets disabled`, `Stretch target fell back`, `Discovery chain: could not connect target`, `No fuel tier for class` — כל אלה הן **שורות מותנות** שיופיעו או לא לפי המצב.

### פירוק לפי שלבים

יצירת התרחיש היא תהליך מדורג שמתבצע ב-`scenario_generator.py`. הוא קורה בקריאה אחת מ-`train_full.py:2638-2640`:

```python
ep_scenario_path = str(scenario_gen.generate(
    episode=episode, config=ep_config,
))
```

מאחורי השיטה הזו יש **6 שלבים** — חלקם ידפיסו לוג, חלקם לא:

#### שלב 1 — Sample כמויות

הגנרטור דוגם *מספר* מטוסים, מתקנים, ו-airbases מהטווח שהוגדר ב-`VariationConfig`. למשל: 2-3 מטוסים, 2-4 מתקנים, 3-5 airbases.

**אין לוג.** הספירות מופיעות רק ב-stats שנשמרים בסוף.

#### שלב 2 — אם `include_sams=False`, הסר SAMs

ב-`scenario_generator.py:625`:
```
include_sams=False → removed all SAM facilities
```

זה מסיר *כל* מתקני SAM (Surface-to-Air Missile) מהתבנית. בריצה של ברירת-המחדל זה תמיד יקרה (כי `include_sams=False` הוא ברירת-המחדל).

#### שלב 3 — חישוב Stretch zone (אזור המטרות הרחוקות)

אזור ההצבה של המטרות מתחלק ל-**שני אזורים**:
- **Easy** — מטרות "קלות", במרחק קצר מהבסיס (לדוגמה ≤ 800km).
- **Stretch** — מטרות "מאתגרות", רחוקות יותר (לדוגמה 1000-2500km), שמחייבות תכנון fuel קפדני.

`stretch_target_ratio=0.5` (ברירת המחדל) אומר שניסיון מצופה שחצי מהמטרות יהיו stretch.

אבל יש שלושה מצבים שיכולים לקרות פה:

##### 3.1 — Stretch זמין (תקין)
ב-`scenario_generator.py:776-777`:
```
Target placement: 2 easy (≤800km), 3 stretch (1000–2500km)
```
זה המצב הרצוי — שני האזורים פעילים.

##### 3.2 — Stretch קרס בגלל time-feasibility cap (`scenario_generator.py:769`)
מה שראינו בדוגמה: `Stretch zone collapsed by time-feasibility cap (stretch_max=1195 ≤ stretch_min=1560)`.

מה זה אומר? בתחנה 2 דיברנו על `Time-feasibility cap: 1195 km` — חישוב אוטומטי של המרחק המקסימלי שאפשר להגיע אליו תוך max_ticks. אם ה-cap הזה (1195km) **קטן** מה-stretch_min (1560km, גבול תחתון של אזור stretch), אזור ה-stretch לא קיים. כל המטרות יוצבו ב-easy.

זה לא באג, אלא הגיון בריא — המערכת מסרבת להציב מטרות שלא ניתן להגיע אליהן בזמן.

##### 3.3 — Stretch מבוטל בגלל fleet range gap (`scenario_generator.py:785`)
```
Stretch targets disabled: fleet range gap (35km) too small for differentiation
```

אם כל המטוסים ב-pool יש להם טווחים דומים (פחות מ-50km הפרש), אין משמעות להבחנה בין easy ו-stretch. הקוד מוותר על stretch כי הוא לא מספק שונות אמיתית באתגר.

#### שלב 4 — הצבת מטרות

הקוד מנסה להציב כל מטרה במיקום אקראי בתוך האזור שנבחר עבורה (easy או stretch). יכולים לקרות:

##### 4.1 — Stretch fallback (`scenario_generator.py:835`)
```
Stretch target fell back to easy zone
```

לפעמים ניסיונות ההצבה ב-stretch נכשלים (אזור צר מדי, עומסים, וכו'). הקוד אז מציב את המטרה ב-easy zone במקום זה.

#### שלב 5 — Discovery chain (קריטי להבנה)

זה אחד מהחלקים החכמים יותר של הגנרטור.

**הרעיון:** חלק מהמטרות יוסתרו מ-RL בהתחלה (target hidden). כדי שלא יישארו "תקועות לעד", הקוד מוודא שלכל מטרה מוסתרת **יש מטרה אחרת בטווח הראדאר שלה** — כך שאם הסוכן יגלה את "מטרת השכן", סריקת הראדאר ב-tick הזה תכלול גם את המטרה המוסתרת.

לוג הסיכום (`scenario_generator.py:1000`):
```
Discovery chain: easy relocated=2/3 isolated=0, stretch relocated=0/0 isolated=0 (min fleet radar=93 km)
```

פירוק:
- **`easy relocated=2/3`** — מתוך 3 מטרות easy שצריכות להיות בשרשרת גילוי, 2 הוצרכו לשנות מיקום כדי לוודא שיש להן שכן ראדאר.
- **`isolated=0`** — אף מטרה לא נשארה "מבודדת" (בלי שכן ראדאר). זה המצב הרצוי.
- **`stretch relocated=0/0`** — באפיזודה הזו אין מטרות stretch (הזכרנו ש-stretch קרס).
- **`min fleet radar=93 km`** — הראדאר עם הטווח הקצר ביותר ב-fleet הוא 93 ק"מ. זה ה-threshold לקביעה אם שתי מטרות "שכנות".

##### 5.1 — Discovery chain failure (`scenario_generator.py:1096`)
```
Discovery chain: could not connect target 'AirField_3' within zone bounds [800-2500 km]; leaving isolated
```
WARNING. אם הקוד לא הצליח אחרי 20 ניסיונות לחבר מטרה לשרשרת, הוא משאיר אותה מבודדת. זה אומר שאם זאת תהיה מטרה מוסתרת, היא **לעולם לא תתגלה** והסוכן לעולם לא יוכל לתקוף אותה. זה כבר תמרור אזהרה.

#### שלב 6 — Fuel tiers + Reachability audit + שמירה לדיסק

##### Fuel tiers (`scenario_generator.py:1299-1305`)
ברירת-המחדל היא להגדיר את ה-fuel של כל מטוס לפי ה-class שלו (B-2 = X liters, F-35 = Y liters, וכו'). אם class לא מוכר, מודפס DEBUG:
```
No fuel tier for class 'F-22 Raptor'; keeping template fuel
```

אם speed או fuelRate לא תקינים:
```
Cannot compute fuel for 'B-2 Spirit' (speed/fuelRate invalid); keeping template fuel
```

בריצה הזו של ברירת-המחדל אף לא אחת מהשורות האלה הופיעה כי כל ה-classes ב-pool מוכרים.

##### Reachability audit (`scenario_generator.py:683`)
```
Reachability audit: 3/3 targets reachable by at least one agent
```

בדיקה אחרונה — לכל מטרה, האם יש לפחות מטוס אחד ב-fleet שיכול להגיע אליה ולחזור בלי לאזול דלק? בריצה תקינה זה צריך להיות `T/T` (כל המטרות).

אם מטרה אחת או יותר לא ניתנות להשגה:
```
Target 'AirField_2' is unreachable by all agents - expected behavior for stretch targets
```
WARNING. זה מקובל עבור מטרות stretch (זה החלק "stretch" בהן), אבל לא עבור easy.

##### שמירה לדיסק (`scenario_generator.py:696-697`)
```python
with open(out_path, "w") as f:
    json.dump(scenario, f, indent=2, ensure_ascii=False)
```
אין לוג על השמירה עצמה. הקובץ נכתב ל-`training_output/scenarios/episode_NNNN_scenario.json` (0-indexed, כפי שראינו בתחנה 4).

#### שלב 7 — טעינה ב-BLADE (`train_full.py:221, 2641-2642`)

```
Reloaded scenario from training_output\scenarios\episode_0000_scenario.json
  Generated scenario: episode_0000_scenario.json
```

`reload_scenario` קורא ל-`game.update_scenario(...)` של BLADE שמכניסה את התרחיש החדש ל-engine. שתי השורות הן בעצם אותו דבר — אחת מ-`reload_scenario` והשנייה אחריה מ-`train_full.py` ישירות.

### מה ייכתב לדיסק

קובץ אחד: `training_output/scenarios/episode_NNNN_scenario.json` (0-indexed). תוכן: BLADE scenario schema מלא — מטוסים, מתקנים, מיקומים, fuel, capabilities, וכו'.

מה-`LOG_INVENTORY` סעיף 4: בריצת הדוגמה הקבצים האלה היו 10-13 KB. בריצת 5000, אם נחשב ממוצע של ~12KB:
- **5000 × 12KB ≈ 60MB סה"כ** — לא דרמטי.
- כל ריצה חדשה מנקה את התיקייה הזו (תחנה 1), אז זה לא מצטבר.

### הערות מעשיות לקראת ריצת 5000

##### 1. שורות מותנות יקרו במציאות
ב-5000 אפיזודות, כמעט בטוח שתראה לפחות פעם אחת מכל אחת מהשורות הבאות, גם בלי force flags:
- `Stretch zone collapsed` — תלוי ב-pool של המטוסים שדגמת. אם דגמת בעיקר מטוסים מהירים, אזור stretch יהיה גדול וזה לא יקרה. אם דגמת KC-135R איטי, יקרה.
- `Stretch target fell back to easy zone` — קורה כשתאי אזור stretch צפוף.
- `Discovery chain: could not connect target` — נדיר אבל יקרה.

אלה לא בעיות, אלא חלק נורמלי מהווריאציה.

##### 2. WARNING שווה לעקוב אחריו
שני ה-WARNINGS הבאים כן שווה לסקור אחרי הריצה:
- `Discovery chain: could not connect target` — אם זה קרה הרבה, אולי ה-bounds לא מספיקים.
- `Target ... is unreachable by all agents` — אם זה קרה במטרות **easy** (לא stretch), זה באג בקונפיגורציה.

ב-`training.log` (גם ללא verbose), WARNINGS תמיד נכנסים כי הם ברמה INFO+.

##### 3. ה-anchor לחיפוש
```bash
grep "Generated scenario" training.log
```
ייתן לך את שם הקובץ של כל אפיזודה. אבל זה DEBUG — בלי verbose לא יהיה ב-`training.log`. במקרה הזה השתמש ב:
```bash
ls training_output/scenarios/
```
זה ידיוק יראה לך כל תרחיש שנוצר.

---

## תחנה 6: Validation Phase

זה השלב **השני באפיזודה** (אחרי יצירת התרחיש), אבל הוא **לא רץ בכל אפיזודה** — רק כשהאפיזודה היא `is_validation_episode`. הוא מריץ את התרחיש **בלי RL** — רק עם oracle של MATCH-AOU מלא, כדי לקבל "התנהגות אידיאלית" כ-baseline להשוואה.

### למה צריך validation phase בכלל?

הבן את ההיגיון: בריצה של RL, הסוכן רואה רק חלק מהמטרות (`partial`) ומגלה את השאר תוך כדי. הוא יכול לפספס מטרות, לבחור פעולות לא-אופטימליות, וכו'. **איך תדע אם RL הצליח טוב או רע?** צריך baseline.

ה-validation phase מריצה את **אותו התרחיש בדיוק** עם oracle שיודע הכל ויש לו את כל הפתרון של MINLP, ועוקבת אחר התוצאה. זה מייצר:
- **recording של "ההתנהגות הנכונה"** — שאתה יכול לפתוח ב-Panopticon ולהשוות חזותית
- **utility baseline** — סך ה-utility שהאוראקל השיג. זה ה-`oracle_total_utility` שאחר כך משמש לחישוב `utility_ratio` של ה-RL.
- **audit metrics** — האם האוראקל עצמו הצליח לבצע את התוכנית? כמה מטרות לא הושגו?

### מתי זה מופיע

- **תלוי ב-`--validate-every`:** הוא רץ באפיזודות שבהן `episode % validate_every == 0`. אם `validate_every=10`, יקרה ב-episodes 0, 10, 20...
- **על הקונסול:** רוב התוכן DEBUG (רק עם verbose), **חוץ מה-audit block שהוא INFO** (יופיע גם בלי verbose).
- **ב-`training.log`:** אותו דבר.
- **ב-`episode_NNNN.log`:** הכל DEBUG (תמיד).
- **בתיקיית `recordings/`:** קובץ `ep<NNN>_validation Recording...jsonl` נכתב, **אם** `should_record=True`.

### מה זה `VAL` בלוגים?

`VAL` הוא קיצור של **Validation**. כל לוג שמופיע בו `VAL` שייך לשלב ה-validation phase. הקוד משתמש ב-3 תגי-מקור שונים שמופיעים בלוגים של פעולות BLADE:

| תג | משמעות | מתי מופיע |
|---|---|---|
| `[VAL ]` | פעולה שנשלחה במהלך validation phase | תחנה 6 |
| `[EXEC]` | פעולה שנשלחה ע"י `BladeExecutorMinimal` במהלך RL phase | תחנה 8 |
| `[RL  ]` | פעולה שנשלחה ע"י ה-RL agent (אחרי החלטת policy) | תחנה 8 |

זה ה-**source tag** — מאיפה הפעולה הגיעה. שלוש שורות אלה נראות זהות במבנה (`Tick X [tag] ACTION: ...`), רק התג משתנה. בקובץ פר-אפיזודה אתה רואה את שני השלבים יחד (validation + RL), והתג מבדיל ביניהם.

### הפלט המלא מ-`run_capture.log` (אפיזודה 1)

נעבור עליו לפי הזמנים בתוך הריצה. אסמן מה DEBUG (רק verbose) ומה INFO (תמיד נראה).

#### 6.1 — Banner ומידע בסיסי (DEBUG)

```
17:16:54,009 | DEBUG | train_full | --- Validation run (oracle only, no RL) ---
17:16:54,011 | DEBUG | scenario_factory | Generated 3 enemy tasks
17:16:54,011 | DEBUG | train_full | Validation: 2 agents, 3 tasks
```

נכתב ב-`train_full.py:705, 730` ו-`scenario_factory.py:204`.

מה קורה:
1. השורה הראשונה רק מסמנת התחלה.
2. `Generated 3 enemy tasks` — `generate_all_enemy_tasks(observation, ATTACKING_SIDE_COLOR)` רץ ויוצר Task object לכל מטרת אויב (ב-`train_full.py:725`). שלוש מטרות = 3 tasks.
3. השורה השלישית מסכמת: יש 2 סוכנים ו-3 מטרות. זה ה-input ל-MINLP.

#### 6.2 — Pyomo בונה את מודל ה-MINLP (DEBUG, רק עם verbose)

זה מה שראית ב-`run_capture.log` בין שורות 305-378 — **70 שורות של Pyomo**. בלי verbose זה נעלם לחלוטין. עם verbose, זה הקטע הכי רועש בריצה.

מה אתה רואה:
- `Constructing ConcreteModel`, `Constructing IndexedVar`
- מבנה המשתנים: `x[A,T,S]` (האם סוכן A מבצע task T בשלב S), `y[T]` (האם task T נבחר בכלל)
- objective function: `maximize 80*y[0]*(1 - 1e-06**(x[0,0,0] + x[1,0,0])) + ...` — מיקסום utility, עם פקטור `(1-1e-06^N)` שמעודד שני סוכנים על אותה מטרה (redundancy).
- כל ה-constraints: `task_step_allocation`, `movement_budget`, וכו'.

**זה לא משהו שאתה צריך להבין שורה-שורה.** זה רעש Pyomo. אם אתה רוצה לראות את המודל המתמטי המלא, ב-`match_aou_MINLP_solver.py` שווה הרבה יותר. אבל לוודא — זה הולך ל-`episode_NNNN.log` תמיד, אז יישמר.

ההפעלה של ה-solver:
```
17:16:57,748 | DEBUG | pyomo.opt | Running ['bonmin.exe', 'tmpzo26z97v.pyomo.nl', '-AMPL']
```

bonmin רץ על קובץ זמני AMPL. זה יכול לקחת 1-3 שניות בתרחיש קטן.

#### 6.3 — סיכום של הפתרון (DEBUG)

```
17:16:59,303 | DEBUG | train_full |   → 5 assignments, 0 unselected
```

מ-`train_full.py:266`. הפתרון של MATCH-AOU המלא יש בו 5 assignments — זה אומר 5 צמדים (agent, task) שנבחרו. `0 unselected` = כל המטרות נבחרו.

אם תכפיל 2 סוכנים × 3 מטרות = 6 צמדים אפשריים, אז 5 מתוך 6 נבחרו. (זה כי MATCH-AOU מאפשר **redundancy** — שני סוכנים על אותה מטרה אם זה משתלם, אבל לא חייב להיות כך).

#### 6.4 — VAL plan לפי סוכן (DEBUG)

```
17:16:59,303 | DEBUG | train_full |   VAL plan: agent=be31019b → tasks=['e3626956', '6c6f7990']
17:16:59,303 | DEBUG | train_full |   VAL plan: agent=0a14f756 → tasks=['e3626956', '5880c13a', '6c6f7990']
```

מ-`train_full.py:786`. **התוכנית של האוראקל לכל סוכן.** מה הוא יעשה במהלך ה-validation:
- `agent be31019b` (ה-B-2 Spirit) יתקוף 2 מטרות: `e3626956` ו-`6c6f7990`.
- `agent 0a14f756` (ה-KC-135R Stratotanker) יתקוף 3 מטרות: `e3626956`, `5880c13a`, `6c6f7990`.

שים לב — `e3626956` ו-`6c6f7990` מתוקפים ע"י **שני** הסוכנים. זה ה-redundancy — להבטיח שהמטרה תיפגע.

ה-IDs מקוצרים ל-8 התווים הראשונים לקריאות.

#### 6.5 — Launch (DEBUG)

```
17:16:59,327 | DEBUG | train_full |   Validation LAUNCH: B-2 Spirit #698 (id=be31019b..) from airbase a3616929..
17:16:59,328 | DEBUG | train_full |   Validation LAUNCH: KC-135R Stratotanker #76 (id=0a14f756..) from airbase a3616929..
```

מ-`train_full.py:806`. שני המטוסים יוצאים מהבסיס הכחול. שם המטוס מופיע מלא, ה-IDs מקוצרים.

#### 6.6 — לולאת הסימולציה (DEBUG)

זה הלב של ה-validation. בכל tick הקוד:
1. מבקש מה-`BladeExecutorMinimal` את הפעולה הבאה.
2. מבצע אותה ב-BLADE.
3. רושם לוגים על מה קרה.

הפעולות שראית:
```
17:16:59,329 | DEBUG | train_full |   Tick     0 [VAL ] MOVE:   agent 0a14f756.. → (37.46175940933924, 38.749287831649916)
17:17:01,445 | DEBUG | train_full |   Tick  2140 [VAL ] ATTACK: agent be31019b.. → target e3626956..
17:17:01,633 | DEBUG | train_full |   Tick  2341 [VAL ] RTB:    agent be31019b..
17:17:03,602 | DEBUG | train_full |   Tick  4432 VAL RTB: agent be31019b.. landed
```

##### ההבדל הקריטי בין `[VAL ] RTB:` לבין `VAL RTB ... landed`

זה שני אירועים שונים בזמן:

**`[VAL ] RTB:` — הפקודה נשלחה (התחיל לחזור)**

מודפס ב-`train_full.py:651` כשהקוד **שולח** ל-BLADE את הפקודה `return_to_base(...)`. בנקודה הזו:
- הסוכן עדיין באוויר (`airborne`)
- הוא רק קיבל את הפקודה לחזור
- ה-route שלו השתנה לכיוון הבסיס
- אבל הוא רחוק מאוד מהבסיס ויחזור ב-tick הרבה יותר מאוחר

**`VAL RTB ... landed` — הסוכן באמת נחת**

מודפס ב-`train_full.py:864` **רק כשהסוכן נעלם מרשימת ה-airborne** של BLADE. הקוד עוקב כל tick:

```python
airborne_ids = {str(getattr(ac, "id", "")) for ac in observation.aircraft}
for aid in agent_ids:
    if aid not in returned and aid not in airborne_ids:
        returned.add(aid)
        logger.debug(f"  Tick {tick:5d} VAL RTB: agent {aid[:8]}.. landed")
```

מה שקורה: BLADE מחזיק רק מטוסים *שבאוויר* ברשימת ה-aircraft של ה-observation. ברגע שמטוס נחת בבסיס, BLADE מוציאה אותו מהרשימה הזאת. הקוד שלנו רואה את זה — סוכן שהיה ברשימה לפני וכבר לא — ויודע שהוא נחת.

##### דוגמה מוחשית מהריצה האמיתית

תסתכל על הסוכן `be31019b` (B-2 Spirit) באפיזודה 1:

```
Tick  2341 [VAL ] RTB:    agent be31019b..       ← פקודת RTB נשלחה (יצא לכיוון הבסיס)
Tick  4432 VAL RTB: agent be31019b.. landed       ← סוכן באמת נחת
```

**ההפרש הוא 2,091 ticks** — זה הזמן שלקח לו פיזית לטוס מהמטרה האחרונה חזרה לבסיס. הוא קיבל את הפקודה ב-tick 2341, אבל רק ב-tick 4432 הוא היה בבסיס.

זאת ההבחנה החשובה: **פקודה נשלחה ≠ פקודה הושלמה.**

##### וידוא שהפקודות בוצעו

הקוד **כן עוקב באופן אקטיבי** ובודק כל tick אם הסוכן נחת. רק אז הוא נחשב `returned`. בסוף הקוד בודק (ב-`train_full.py:894`):

```python
if tick > 100 and len(returned) == len(agent_ids):
    logger.debug(f"  Validation: all agents RTB at tick {tick}")
    break
```

**רק כש-*כל* הסוכנים נחתו פיזית** הקוד שובר את לולאת הסימולציה. אם סוכן קיבל פקודת RTB אבל לא הצליח לחזור פיזית (אזל לו fuel באמצע, נתקע, וכו'), הוא לא נכנס ל-`returned` והלולאה ממשיכה עד `max_ticks` ואז יש timeout.

זה גם הסיבה שב-audit block יש אבחנה בין `plan` (תוכננו) ל-`hit` (הותקפו בפועל) — הקוד מודד את התוצאה האמיתית, לא רק את הכוונה.

הסוף הצפוי לסימולציה תקינה:

```
17:17:04,594 | DEBUG | train_full |   Tick  5481 VAL RTB: agent 0a14f756.. landed
17:17:04,594 | DEBUG | train_full |   Validation: all agents RTB at tick 5481
```

מ-`train_full.py:895`. **כל הסוכנים חזרו לבסיס**, ה-validation הסתיים בהצלחה.

#### 6.7 — END-ZONE diagnostic block (DEBUG, רק במקרי timeout)

זה בלוק אבחוני שמודפס **רק כשהסימולציה לא הסתיימה בזמן**. הקוד:

```python
ticks_remaining = max_ticks - tick
if ticks_remaining <= 100 and ticks_remaining % 10 == 0:
    # ...print END-ZONE block...
```

##### מתי זה מופיע?

**רק ב-100 הטיקים האחרונים לפני max_ticks**, וגם אז רק כל 10 טיקים. במשוואה שלנו (max_ticks=14400):
- מ-tick 14300 עד tick 14400
- כל 10 טיקים: 14300, 14310, 14320, ..., 14400
- סך הכל **11 הדפסות** (אם הסימולציה אכן הגיעה לשם)

##### זה לא מופיע בריצה תקינה

חשוב להבין: **בריצה תקינה** הסימולציה מסתיימת בשני מצבים:

1. **כל הסוכנים נחתו** (`all agents RTB`) — הסימולציה שוברת ב-`break` הרבה לפני max_ticks. ה-END-ZONE block לעולם לא יודפס.
2. **הסימולציה הגיעה ל-max_ticks** — או כי סוכן תקוע באוויר, או כי המשימות לקחו יותר זמן מהצפוי. רק במקרה הזה הקוד מגיע ל-100 הטיקים האחרונים, ואז ה-END-ZONE block מודפס.

**אז END-ZONE block מודפס רק במקרי timeout. בריצה תקינה הוא לעולם לא יופיע.**

##### למה הוא קיים? (תועלת מעשית)

המטרה היא **לתת לך מידע אבחוני כשמשהו השתבש**, *בלי* שתצטרך לפתוח את ה-recording ב-Panopticon.

תאר לעצמך: רצת 5000 אפיזודות. בוקר אחר כך אתה מסתכל בתוצאות. אתה רואה ש-23 אפיזודות עברו timeout. השאלה: למה? יש כמה תרחישים אפשריים:
- סוכן תקוע באוויר ולא חוזר (route bug?)
- אזל הדלק ב-fuel באמצע (planning bug?)
- BLADE לא שלחה את הפקודות הנכונות
- האפיזודה היתה גדולה מדי בזמן

**בלי END-ZONE block:** אתה צריך לפתוח כל recording אחד-אחד ב-Panopticon ולנסות להבין מה קרה. עבודה רצינית.

**עם END-ZONE block:** אתה פותח את `episode_NNNN.log` ורואה:

```
── Tick 14290 [VAL END-ZONE] ── remaining=110 | airborne=2 | returned=0/2 | terminated=False | truncated=False
    B-2 Spirit (id=be31019b..): pos=(37.51,38.20) fuel=15234 rtb=False route_pts=12
    KC-135R (id=0a14f756..): pos=(35.20,40.10) fuel=200 rtb=True route_pts=3
```

מיד אתה רואה:
- **B-2 Spirit:** `rtb=False` — הוא בכלל לא בדרך חזרה. למה? כנראה תקוע על משימה שלא הצליח לסיים (שווה לבדוק את ה-route).
- **KC-135R:** `rtb=True` אבל `fuel=200` — אזל לו הדלק כמעט. הוא לא יספיק לחזור.

תוך 5 שניות הבנת את התקלה בלי לפתוח Panopticon.

##### פירוק הבלוק שדה-שדה

הבלוק מודפס כיחידה אחת כל 10 טיקים:

```
── Tick 14290 [VAL END-ZONE] ── remaining=110 | airborne=2 | returned=0/2 | terminated=False | truncated=False
    B-2 Spirit (id=be31019b..): pos=(37.51,38.20) fuel=15234 rtb=False route_pts=12
    KC-135R (id=0a14f756..): pos=(35.20,40.10) fuel=200 rtb=True route_pts=3
```

| שדה | משמעות |
|---|---|
| `remaining=110` | טיקים שנשארו עד `max_ticks` (`max_ticks - tick`). כשמגיע ל-0 → truncation |
| `airborne=2` | מספר המטוסים באוויר (`len(observation.aircraft)`) |
| `returned=0/2` | כמה מהסוכנים שלנו (`agent_ids`) כבר נחתו / סך הסוכנים שלנו |
| `terminated` / `truncated` | gym flags מ-`env.step()` |
| `pos=(lat, lon)` | מיקום נוכחי של המטוס |
| `fuel` | `ac.current_fuel` ביחידה פנימית של BLADE |
| `rtb` | flag — האם המטוס בדרך חזרה לבסיס (`ac.rtb`) |
| `route_pts` | מספר ה-waypoints שנשארו ב-route (`len(ac.route)`). כשמתקדם — יורד |

**הבחנה בין `airborne` ל-`returned`:** `airborne` כולל את כל המטוסים ב-`observation.aircraft`, ו-`returned` סופר רק מתוך הסוכנים שלנו. בריצה תקינה: `airborne + returned = n_agents`.

##### האם להסיר אותו?

**מומלץ להשאיר.** הסיבות:

1. **עלות תפעולית: אפס.** הוא רץ רק ב-100 הטיקים האחרונים, וגם אז רק כל 10. בריצה תקינה הוא לא רץ בכלל.
2. **בריצת 5000 הוא יציל לך זמן.** אם 50 אפיזודות עברו timeout, אתה תרצה את המידע הזה אגרגטיבי.
3. **הוא DEBUG, לא INFO.** לא יופיע על המסך גם עם verbose מוגבל. רק ב-`episode_NNNN.log`.

**מתי כן הייתי שוקל להסיר:**
- אם תוסיף בעתיד מערכת telemetry/dashboard עם metrics אגרגטיביים שכבר יודעים לסכם תקלות — אז ה-END-ZONE יהיה כפילות.

**ההמלצה:** השאר. זה אבחון "חינם" שיהיה שווה זהב כשמשהו ישתבש.

#### 6.8 — Validation Audit Block (INFO!)

זה **הקטע החשוב ביותר** בכל ה-validation. הוא מודפס ברמת **INFO** — כלומר תופיע גם **בלי `--verbose`**.

```
17:17:04,595 | INFO | train_full |   --- Validation audit ---
17:17:04,595 | INFO | train_full |     t=e3626956 reach=[0a14,be31] plan=[be31,0a14] hit=Y cheapest=be31:21505
17:17:04,595 | INFO | train_full |     t=5880c13a reach=[0a14,be31] plan=[0a14] hit=Y cheapest=be31:22682
17:17:04,596 | INFO | train_full |     t=6c6f7990 reach=[0a14,be31] plan=[be31,0a14] hit=Y cheapest=be31:23342
17:17:04,596 | INFO | train_full |     agent=be31 budget=120057 cap=60028 used=44847/60028 plan=[e3626956,6c6f7990]
17:17:04,596 | INFO | train_full |     agent=0a14 budget=205998 cap=102999 used=82764/102999 plan=[e3626956,5880c13a,6c6f7990]
17:17:04,597 | INFO | train_full |   Hit: plan=3/3 reachable=3/3 unreachable=0/0 dropped_reachable=0 oracle_violations=0
```

נפרק את זה — קודם **שורות per-target**:

```
t=e3626956 reach=[0a14,be31] plan=[be31,0a14] hit=Y cheapest=be31:21505
```

| שדה | משמעות |
|---|---|
| `t=e3626956` | מזהה המטרה (8 תווים) |
| `reach=[0a14,be31]` | אילו סוכנים יכולים להגיע למטרה (ולחזור) — מבחינת fuel budget |
| `plan=[be31,0a14]` | אילו סוכנים *תוכננו* לתקוף לפי האוראקל |
| `hit=Y/N` | האם המטרה אכן הותקפה במהלך הסימולציה |
| `cheapest=be31:21505` | הסוכן הזול ביותר להגעה למטרה (ביחידות fuel) — `be31` עולה 21505 fuel |

**מה אתה לומד מזה:**
- אם `plan=[]` ו-`reach=[X,Y]` — האוראקל בחר לא לתקוף מטרה שניתן להגיע אליה. זה ה-`dropped_reachable`.
- אם `hit=N` ו-`plan=[X]` — האוראקל תכנן אבל לא הצליח לבצע. זה רע!
- אם `reach=[]` — שום סוכן לא יכול להגיע למטרה. תיקח את זה לתחנה 5 ולבדוק את reachability audit.

ואז שורות **per-agent**:

```
agent=be31 budget=120057 cap=60028 used=44847/60028 plan=[e3626956,6c6f7990]
```

| שדה | משמעות |
|---|---|
| `agent=be31` | מזהה הסוכן (4 תווים) |
| `budget=120057` | סך ה-fuel שיש לו |
| `cap=60028` | התקציב שאוראקל מאפשר לעצמו להשתמש (`budget * (1 - RISK)` = 50% מהתקציב). RISK=0.5 לפי הקוד |
| `used=44847/60028` | כמה fuel נצרך מתוכן ה-cap. 44847 מתוך 60028 = 75% מנוצל |
| `plan=[e3626956,6c6f7990]` | המטרות שתוכננו לסוכן הזה |

**הקונספט של `cap` (חצי מ-budget):**
ה-RISK=0.5 הוא safety margin — האוראקל לא מנצל את כל ה-fuel, רק חצי. זה מבטיח שיש מרווח לחזור גם אם יש סטיות. אם `used > cap` בפועל זה תקלה.

אחרון, **שורת ה-headline**:

```
Hit: plan=3/3 reachable=3/3 unreachable=0/0 dropped_reachable=0 oracle_violations=0
```

| שדה | משמעות | המצב הרצוי |
|---|---|---|
| `plan=3/3` | מטרות מתוך התוכנית שהותקפו / סך מטרות בתוכנית | `X/X` (הכל) |
| `reachable=3/3` | מטרות ניתנות-להשגה שהותקפו / סך ניתנות-להשגה | `X/X` |
| `unreachable=0/0` | מטרות לא-ניתנות-להשגה שהותקפו / סך לא-ניתנות-להשגה | `0/0` (שניהם) |
| `dropped_reachable=0` | מטרות ניתנות-להשגה שהאוראקל לא תכנן | `0` |
| `oracle_violations=0` | סוכנים שתוכננו לתקוף מטרות שהם לא יכולים להגיע אליהן | `0` |

**זה התקציר הכי חשוב של validation.** זאת השורה שתסקור כדי לדעת אם validation עבד נכון.

#### 6.9 — שורות נוספות שעלולות להופיע

##### `Dropped reachable targets` (INFO)
```
Dropped reachable targets (oracle chose not to plan): ['t1', 't2']
```
מ-`train_full.py:947`. אם `dropped_reachable > 0`, מודפסת גם רשימת המטרות. זה מצב שבו האוראקל בחר מסיבותיו לא לתקוף מטרה שאפשר היה. למשל אם זה לא משתלם בגלל fuel.

##### `ANOMALY: unreachable target(s) attacked` (ERROR!)
```
ANOMALY: unreachable target(s) attacked: ['xyz']
```
מ-`train_full.py:952`. זה **לא אמור לקרות**. אם זה קורה, יש באג — או ב-reachability audit, או ב-AOU solver, או במשהו אחר. ב-5000 אם תראה אפילו פעם אחת — זה תמרור אזהרה רציני.

##### `Oracle plan incomplete in execution` (WARNING)
```
Oracle plan incomplete in execution — missed: ['t1']
```
מ-`train_full.py:958`. האוראקל תכנן לתקוף מטרה אבל מסיבה כלשהי הסוכן לא הצליח להגיע (אזל לו fuel באמצע, נתקע, וכו'). זו לא בהכרח שגיאה — לפעמים זה מקרה קצה — אבל אם זה קורה הרבה זה דגל אדום.

#### 6.10 — Recording export (DEBUG, רק אם `should_record=True`)

```
17:17:04,696 | DEBUG | train_full |   Validation recording exported: ep001_validation
Recording exported to 'training_output\recordings/ep001_validation Recording 064510 - 081649.jsonl'
```

שתי שורות:
- ראשונה — DEBUG מ-`train_full.py:967`.
- שנייה — `print()` מ-BLADE. זה הולך ל-stdout ישירות, לא דרך ה-logging system. כך שב-`training.log` תראה רק את הראשונה.

הקובץ נכתב ל-`training_output/recordings/ep<NNN>_validation Recording <start> - <end>.jsonl` (1-indexed). השמה הזמנים זה start/end clock-time של הסימולציה (לא tick-time).

### בעיות שעלולות להופיע

##### `Validation: no agents found, skipping` (WARNING)
```
17:xx:xx | WARNING | train_full | Validation: no agents found, skipping
```
התרחיש לא יצר סוכנים תוקפים. סימן לבאג ב-ScenarioGenerator או ב-scenario factory.

##### `Validation: solver returned empty solution, skipping` (WARNING)
```
17:xx:xx | WARNING | train_full | Validation: solver returned empty solution, skipping
```
ה-MINLP לא מצא פתרון. בדוק שיש מטוסים, מטרות, ושאין constraints סותרים.

### מה ייכתב לדיסק

| נתיב | מתי |
|---|---|
| `training_output/recordings/ep<NNN>_validation Recording...jsonl` | רק אם `should_record=True` |

הגדלים בריצה: 1.7-11.4 MB לכל recording. בריצת 5000 אם `validate_every=10` ו-`record_every=50`:
- recordings יקרו רק באפיזודות שגם validation וגם record מתאימים — אפיזודות 0, 50, 100... = 100 recordings × ~5MB = 500MB.
- אם תיתן `--record-every 1` (כמו בריצת ה-5), זה היה הופך ל-5000 × 5MB = 25GB. **זה הרבה.**

### עלות זמן

מהדגמה: ~10-15 שניות לאפיזודה ל-validation בלבד (כולל פתרון MINLP + סימולציה מלאה של ~5500 ticks). בריצת 5000 עם `validate_every=10`:
- 500 validations × ~12 שניות = 6000 שניות = **100 דקות = 1.7 שעות** רק על validation.

עם `validate_every=50`:
- 100 validations × 12 שניות = 1200 שניות = **20 דקות**.

**המלצה לריצת ה-5000:** השתמש ב-`--validate-every 50` או אפילו 100. אתה רוצה baseline אבל לא 500 פעמים.

### הערות מעשיות לקראת ריצת 5000

##### 1. ה-audit block הוא הדבר היחיד INFO
בלי verbose, מתוך כל החגיגה הזו תראה רק את ה-audit block (8 שורות). זה מספיק לרוב המקרים. אם משהו לא נראה תקין, תפתח את `episode_NNNN.log` ותראה את הכל.

##### 2. ה-anchor לחיפוש
```bash
grep -n "Validation audit" training.log    # ימצא כל ה-audits
grep -n "ANOMALY" training.log             # ימצא בעיות חמורות
grep -n "Hit: plan=" training.log          # ימצא רק את ה-headline שורות
```

##### 3. אזהרות שכדאי לעקוב אחריהן בריצת 5000
- **`ANOMALY` כלשהו** — אפילו פעם אחת. זה buggy.
- **`Oracle plan incomplete` חוזר** — אם זה קורה ביותר מ-5% מה-validations, יש בעיה בביצוע (BladeExecutorMinimal לא עובד נכון, או fuel chain שבור).
- **`dropped_reachable > 0` חוזר** — האוראקל לא מתכנן באופן יעיל. אולי קונפיגורציית RISK גבוהה מדי.

##### 4. ה-recordings תופסים מקום
שקול להפחית את `--record-every` אם המקום בעייתי. אבל אם אין בעיית מקום — זה החומר הויזואלי הכי טוב לקראת בדיקה ויזואלית ב-Panopticon.

---

## תחנה 7: RL Setup

זה השלב **שמכין את הקרקע ל-RL phase** (תחנה 8). הוא בא **אחרי** validation (אם רץ), או מיד אחרי scenario generation (אם לא רץ validation). השלב הזה תמיד רץ — בכל אפיזודה.

### הסדר המלא של אפיזודה אחת — ראייה מקיפה

לפני שנצלול לתחנה 7 עצמה, חשוב להבין איך היא משתלבת בזרימה הכוללת של אפיזודה. **זה אחד המקומות שהכי קל להתבלבל בהם.**

```
[התחלת איטרציה episode N]
│
├─ אם vary_scenarios=True:                    ← תחנה 5
│   ├─ ScenarioGenerator.generate()
│   │   ├─ דוגם סוגי מטוסים, כמויות, מיקומים
│   │   ├─ מחשב stretch zones
│   │   ├─ מציב מטרות (rejection sampling)
│   │   ├─ מבצע discovery chain ב-scenario level
│   │   ├─ מבצע reachability audit
│   │   ├─ מחיל fuel tiers
│   │   └─ שומר ל-`episode_NNNN_scenario.json`
│   │
│   └─ reload_scenario(game, path)            ← טעינה ראשונה ל-BLADE
│
├─ פותח episode_NNNN.log handler             ← תחנה 4
│
├─ אם is_validation_episode:                  ← תחנה 6
│   ├─ run_validation_episode():
│   │   ├─ env.reset() (איפוס BLADE)
│   │   ├─ מייצר agents + ALL tasks (full)
│   │   ├─ פותר MATCH-AOU על FULL (oracle)   ← קריאת סולבר #1
│   │   ├─ משגר מטוסים
│   │   ├─ רץ הסימולציה במלואה (executor בלבד, בלי RL)  ← סימולציה #1
│   │   ├─ מדפיס audit block
│   │   └─ מייצא recording
│   │
│   └─ reload_scenario(game, path)            ← טעינה שנייה — איפוס לפני RL
│
└─ train_episode():                           ← תחנה 7+8 (RL phase)
    ├─ env.reset() (איפוס BLADE שוב)
    ├─ split_tasks() → partial + full + hidden
    ├─ פותר MATCH-AOU על PARTIAL             ← קריאת סולבר #2 (או #1 באפיזודה רגילה)
    ├─ פותר MATCH-AOU על FULL (oracle)        ← קריאת סולבר #3 (או #2 באפיזודה רגילה)
    ├─ משגר מטוסים
    ├─ מריץ הסימולציה עם RL                   ← סימולציה #2 (או #1 באפיזודה רגילה)
    ├─ מבצע PPO update
    └─ מייצא recording
```

### טבלת סיכום: פעולות סולבר וסימולציה

| | אפיזודה רגילה | אפיזודת validation |
|---|---|---|
| קריאות סולבר | 2 (partial + full) | 3 (full + partial + full) |
| ריצות סימולציה | 1 (RL על partial) | 2 (oracle + RL על partial) |
| Reload scenario | 1 (אחרי הגנרציה) | 2 (אחרי הגנרציה + אחרי validation) |

### למה ה-full plan נפתר באפיזודה רגילה אם לא רצים עליו?

זאת שאלה חשובה. ה-full plan **לא רץ בסימולציה באפיזודה רגילה** — אז למה לפתור אותו בכלל?

ה-full plan משמש לשני דברים:

1. **חישוב `oracle_total_utility`** — סך ה-utility שהאוראקל היה משיג. זה ה-baseline ל-reward של ה-RL (תחנה 9). בלי לפתור את ה-full, אין דרך לדעת מה הביצוע "האידיאלי".

2. **חישוב `target_extraction`** — איזו מטרה כל סוכן "צריך ללמוד לתקוף". ה-ORACLE SETUP block יציג: `→ Agent should learn to attack: {...}`. זה משמש מאוחר יותר ב-RL לחישוב `Match=✓/✗` (האם ה-RL החליט נכון).

**ה-full plan קיים בעיקר כמידע** — לא כתוכנית ביצוע. הוא ה"תשובה הנכונה" שמשמשת ללמד את ה-RL מה אמור היה לקרות.

### מטרת השלב

יש 6 חלקים שצריך להכין לפני שה-RL יוכל לעבוד:

1. **טעינה מחדש של התרחיש** — כי validation שינתה את מצב BLADE (אם רץ)
2. **Task split** — חלוקה ל-`partial` (מטרות שהסוכן יודע מההתחלה) ו-`full` (כל המטרות, כולל מוסתרות)
3. **שתי קריאות ל-MATCH-AOU** — אחת לפתרון partial, אחת לפתרון full (oracle)
4. **Pre-launch** — הוצאת המטוסים מהבסיס לפני שהסימולציה מתחילה
5. **Setup ה-Executor** — ה-`BladeExecutorMinimal` עם ה-partial plan
6. **חישובים מקדימים ל-Oracle** — מה כל סוכן "צריך ללמוד" לתקוף

### מתי זה מופיע

- **על הקונסול:** רוב התוכן DEBUG (רק עם verbose). חלקים מסוימים מופיעים תמיד (תזכרו — בלי verbose, רוב הדבר הזה שקט).
- **ב-`training.log`:** אותו דבר.
- **ב-`episode_NNNN.log`:** הכל DEBUG (תמיד נכתב).

### חלוקה לרזולוציה — שני מצבי הדפסה

כל החלקים שאני הולך להראות מתחלקים לשני מצבים שכדאי להבחין ביניהם:

1. **קריאות `logger.debug` רגילות** — שורות סיכום קצרות (אחת עד כמה). תמיד נכתבות ל-`episode_NNNN.log` (handler שם הוא DEBUG תמיד). ב-console וב-`training.log` יופיעו רק עם `--verbose`.
2. **קריאות עטופות ב-`if verbose:`** — בלוקי דיבוג מורחבים עם 10-30 שורות. בלי `--verbose` הן לא נקראות בכלל — לא יופיעו בשום מקום, גם לא ב-`episode_NNNN.log`. אלה הבלוקים עם הכותרות `============================================================`.

קל להבדיל ביניהם: אם יש כותרת `=========`, זה בלוק verbose בלבד.

### הפלט המלא מ-`run_capture.log` (אפיזודה 1)

#### 7.1 — טעינה מחדש של התרחיש (DEBUG)

```
17:17:04,698 | DEBUG | train_full | Reloaded scenario from training_output\scenarios\episode_0000_scenario.json
17:17:04,698 | DEBUG | scenario_factory | Generated 3 enemy tasks
```

מ-`train_full.py:221, 2683` ו-`scenario_factory.py:204`.

מה קורה: ה-validation שינתה את מצב BLADE (סוכנים נחתו, מטרות הותקפו וכו'). הקוד טוען את התרחיש **שוב** מאותו JSON file כדי לאפס. גם `generate_all_enemy_tasks` רץ שוב כדי לבנות את ה-Task objects מחדש.

> **הערה חשובה:** זאת הסיבה ש-validation ו-RL רצים על "אותו תרחיש" אבל כל אחד מקבל reset נקי. ההתחלה זהה, מה שקורה אחר כך — שונה.
>
> שורת ה-`Reloaded scenario` הספציפית הזאת מופיעה רק באפיזודות validation. באפיזודה רגילה, ה-`Reloaded scenario` היחיד הוא זה שאחרי `scenario_gen.generate()` (תחנה 5).

#### 7.2 — סיכום קומפקטי של התרחיש (DEBUG)

```
17:17:04,698 | DEBUG | train_full | Scenario: 2 agents ['B-2 Spirit', 'KC-135R Stratotanker'] | Blue base: (32.85, 35.31)
17:17:04,698 | DEBUG | train_full |   Targets (3): Red Airbase (37.46, 38.75), Red Airbase (37.55, 39.32), Red Airbase (37.95, 39.00)
```

מ-`train_full.py:1092-1097`. סיכום של מה יש בתרחיש:
- כמה סוכנים ואיזה סוגים
- מיקום הבסיס הכחול
- כמה מטרות ואיפה הן

זה תקציר יעיל שעוזר לזהות מהר מה הולך לקרות באפיזודה. בריצת 5000 ללא verbose, **השורות האלה לא יופיעו** (DEBUG).

#### 7.3 — בלוק verbose: AGENTS dump (רק עם `--verbose`)

```
============================================================
AGENTS
============================================================
  Agent 0: be31019b-75b4-4474-a90b-1b6249d735da
    Name:      (from scenario)
    Location:  (32.3542, 34.8124)
    Budget:    120057
    Weapon ID: 59a5a12e-a168-4a95-bcf3-8d14bd6fcea1
    Home base: a3616929-2446-4345-af5a-3a9986908c0d
    Capabilities: ['attack', 'attack', 'attack']
  Agent 1: 0a14f756-13f2-4c78-8aa8-446da245aee5
    ...
```

מ-`train_full.py:1102-1114`. דחיסה מלאה של כל סוכן: ID, מיקום, fuel budget, weapon ID, home base, ו-capabilities.

`Capabilities: ['attack', 'attack', 'attack']` אומר שלסוכן יש 3 capabilities של attack — שזה נראה מוזר. למעשה זה אומר שהוא יכול לבצע 3 attacks (כי הוא לא מתחדש ב-weapons בלי לחזור לבסיס). פרטים ברמת הקוד נמצאים ב-`capability.py`.

**זה בלוק verbose בלבד** — לא יופיע ב-5000 בלי verbose. הוא מופיע רק לדיבוג כשרוצים לראות בדיוק מה יש.

#### 7.4 — בלוק verbose: ALL TASKS dump (רק עם `--verbose`)

```
============================================================
ALL TASKS (3 total)
============================================================
  Task 0:
    Target ID: e3626956-04af-4440-990f-a1088445cc9b
    Utility:   80
    Location:  (37.4618, 38.7493)
    Action:    handle_aircraft_attack('AGENT_ID', 'e3626956-04af-4440-990f-a1088445cc9b', 'WEAPON_ID', 2)
  Task 1:
    ...
```

מ-`train_full.py:1116-1128`. **כל המטרות** עם פרטים: target ID, utility, location, action template.

ה-`'AGENT_ID'` ו-`'WEAPON_ID'` הם placeholders — בעת ביצוע, הם מוחלפים ב-IDs אמיתיים של הסוכן והנשק שלו.

#### 7.5 — Task Split (תמיד מודפס בעת DEBUG)

זה **הקטע הקריטי ביותר ב-RL setup.** הקוד מחלק את המטרות לשתי קבוצות:

```
17:17:04,715 | DEBUG | train_full | Discovery chain (split): clean (hidden=1, known=2, isolated_pinned=0, min_radar=93 km)
17:17:04,715 | DEBUG | train_full | Task split: 2 partial, 3 full, 1 hidden
```

מ-`train_full.py:424-431`.

##### מה זה Task Split?

הרעיון המרכזי של מערכת ה-RL הזאת: **הסוכן לא יודע מהתחלה את כל המטרות.** הוא מכיר רק חלק מהן (`partial`), ומוצא את השאר תוך כדי הסימולציה דרך discovery (סריקת ראדאר). ה-RL צריך ללמוד **איך לעדכן את התוכנית בזמן אמת** כשנתגלות מטרות חדשות.

Task split מחליט:
- **`partial`** — מטרות שהסוכן יודע עליהן מההתחלה. ל-`MATCH-AOU` יוצרים תוכנית מקורית רק עליהן.
- **`full`** — כל המטרות (כולל המוסתרות). ל-`MATCH-AOU` יוצרים תוכנית "אוראקל" שיודעת הכל.
- **`hidden`** = `full - partial` = מטרות שצריכות להתגלות תוך כדי.

ברירת המחדל: `PARTIAL_RATIO = 2/3`, כלומר 2/3 מהמטרות `partial`, 1/3 `hidden`.

##### השדות בלוג

`Task split: 2 partial, 3 full, 1 hidden`:

| שדה | משמעות |
|---|---|
| `partial=2` | 2 מטרות שהסוכן יודע מההתחלה |
| `full=3` | 3 מטרות סך הכל |
| `hidden=1` | 1 מטרה שתתגלה תוך כדי הסימולציה |

##### Discovery Chain — מה זה?

זאת המורכבות ה"חכמה" של ה-split. תיאור הבעיה: אם נסתיר מטרה אקראית, יש סיכוי שלעולם לא תתגלה. למה? כי הסוכן צריך **לעוף לטווח הראדאר** כדי לגלות מטרה חדשה. אם המטרה המוסתרת היא הרחוקה ביותר ואין לסוכן סיבה אחרת לעוף לאזור — הוא לעולם לא יראה אותה.

הפתרון: `split_tasks` מוודאת שלכל מטרה מוסתרת **יש לפחות מטרה מוכרת** בטווח הראדאר. כך שכשהסוכן יעוף לתקוף את ה"שכן המוכר", הראדאר שלו יקלוט גם את המטרה המוסתרת.

> **הערה — שתי "discovery chain" שלא לבלבל ביניהן:**
>
> 1. **Discovery chain ב-`scenario_generator`** (תחנה 5) — בודק שכנויות בין מטרות בתרחיש כולו, ומשנה **מיקומים** אם צריך כך שאפשר יהיה ליצור שרשרת גילוי.
>
> 2. **Discovery chain ב-`split_tasks`** (תחנה 7, פה) — מחליט **איזו** מטרה תהיה partial ואיזו hidden. המיקומים כבר נקבעו בתחנה 5 — כאן רק מחליטים מי מתגלה ומי לא.
>
> זה אותו רעיון בסיסי (שרשרת גילוי), אבל בשני שלבים שונים. הראשון על המיקומים, השני על הסיווג.

##### השדות ב-Discovery Chain log

`Discovery chain (split): clean (hidden=1, known=2, isolated_pinned=0, min_radar=93 km)`:

| שדה | משמעות |
|---|---|
| `clean` | התוצאה: הצלחה בניסיון הראשון. ערכים אפשריים: `clean`, `resampled (attempt N)`, `warn-fallback`, `no-chain`, `exhaust` |
| `hidden=1` | 1 מטרה מוסתרת |
| `known=2` | 2 מטרות מוכרות לסוכן (שהן ה-partial) |
| `isolated_pinned=0` | 0 מטרות שהיה צריך "לכפות" להיכנס לקבוצת המוכרות (כי אין להן שכן) |
| `min_radar=93 km` | טווח הראדאר הקצר ביותר ב-fleet (משמש לקביעת "שכנות") |

##### תוצאות אפשריות (`outcome`)

הקוד מנסה עד 20 פעמים לעשות sampling אקראי שיעמוד בתנאי ה-discovery chain:

| Outcome | משמעות |
|---|---|
| `clean` | הצליח בניסיון הראשון. הכי טוב. |
| `resampled (attempt N)` | הצליח אחרי N ניסיונות. עדיין תקין. |
| `exhaust` | יותר מדי מטרות מבודדות (אין להן שכנים). חלקן הוסתרו ולא ניתן להגיע אליהן. |
| `warn-fallback` | אחרי 20 ניסיונות לא הצלחנו למצוא split תקין. השאיר את הניסיון האחרון, חלק מהמוסתרות אולי לא יתגלו. |
| `no-chain` | בדיקה דולגה (אין observation או fleet ללא ראדאר) |

##### בלוק verbose: TASK SPLIT (רק עם `--verbose`)

```
============================================================
TASK SPLIT
============================================================
  Partial tasks (2):
    [0] target=e3626956-..., utility=80
    [1] target=5880c13a-..., utility=80
  Full tasks (3):
    [0] target=e3626956-..., utility=80
    [1] target=5880c13a-..., utility=80
    [2] target=6c6f7990-..., utility=80 *** HIDDEN ***
  Hidden targets: {'6c6f7990-d33c-495b-a60f-67c66f03253e'}
```

מ-`train_full.py:1150-1163`. רק עם verbose. רואים בדיוק איזו מטרה היא partial, איזו הוסתרה.

ה-`*** HIDDEN ***` מסומן ליד מטרה שהיא ב-full אבל לא ב-partial.

#### 7.6 — Solving MATCH-AOU twice

הקוד פותר את ה-MINLP **פעמיים** לכל אפיזודת RL:

```
17:17:04,730 | DEBUG | train_full | Solving MATCH-AOU (partial)...
[...60+ שורות Pyomo עם verbose...]
17:17:07,xxx | DEBUG | train_full |   → 2 assignments, 0 unselected

17:17:07,xxx | DEBUG | train_full | Solving MATCH-AOU (full / oracle)...
[...60+ שורות Pyomo עם verbose...]
17:17:07,xxx | DEBUG | train_full |   → 5 assignments, 0 unselected
```

מ-`train_full.py:1172, 1181`.

##### למה פעמיים?

- **Partial solution** — התוכנית שהסוכן מתחיל איתה (יודע רק על partial). זאת ה-baseline שלו ב-tick 0. כל "תיקונים" שה-RL יעשה אחר כך הם תיקונים על התוכנית הזאת.
- **Full solution (oracle)** — התוכנית שאיש "כל-יודע" היה עושה. זה ה-baseline להשוואה. ה-reward של ה-RL מבוסס על כמה הוא מתקרב לזה.

> **חשוב — מה רץ בסימולציה ומה לא:**
>
> - ה-**partial plan** הוא מה שמשמש את ה-`BladeExecutorMinimal` בסימולציה. זאת התוכנית שמבוצעת בפועל.
> - ה-**full plan** הוא רק לחישובים — `oracle_total_utility`, `target_extraction`. הוא **לא** רץ בסימולציה באפיזודה רגילה.
> - ב-validation phase, ה-full plan **כן** רץ בסימולציה — אבל זה בשלב נפרד שמסתיים לפני RL phase מתחיל (ראה דיאגרמת הסדר למעלה).

##### בלוקי verbose בנושא MATCH-AOU

עם `--verbose`, יש כמה בלוקי דיבוג מורחבים:

```
============================================================
MATCH-AOU SOLUTIONS
============================================================
Solving MATCH-AOU (partial)...
  --- Partial Solution ---
  Total assignments: 2
  Agent be31019b-...:
    task=0 step=0 level=0 → target=e3626956-...
  Agent 0a14f756-...:
    task=1 step=0 level=0 → target=5880c13a-...

Solving MATCH-AOU (full / oracle)...
  --- Full (Oracle) Solution ---
  Total assignments: 5
  Agent be31019b-...:
    task=0 step=0 level=0 → target=e3626956-...
    task=2 step=0 level=0 → target=6c6f7990-...
  Agent 0a14f756-...:
    task=0 step=0 level=0 → target=e3626956-...
    task=1 step=0 level=0 → target=5880c13a-...
    task=2 step=0 level=0 → target=6c6f7990-...

  --- Comparison ---
  Targets in partial: {'5880c13a-...', 'e3626956-...'}
  Targets in full:    {'6c6f7990-...', '5880c13a-...', 'e3626956-...'}
  NEW in full (what RL should learn to attack): {'6c6f7990-...'}
```

מ-`train_full.py:1167-1197`.

ה-`Comparison` block בסוף הוא מאוד שימושי — הוא **מסכם בדיוק מה ה-RL צריך ללמוד**: המטרות שיש ב-full אבל לא ב-partial הן מה שהסוכן צריך לגלות ולתקוף.

#### 7.7 — Pre-Launch (תמיד DEBUG)

```
17:17:07,564 | DEBUG | train_full |   LAUNCH: B-2 Spirit #698 (id=be31019b..) from airbase a3616929..
17:17:07,564 | DEBUG | train_full |   LAUNCH: KC-135R Stratotanker #76 (id=0a14f756..) from airbase a3616929..
17:17:07,616 | DEBUG | train_full |   Airborne after launch: 2 aircraft — ['B-2 Spirit #698', 'KC-135R Stratotanker #76']
```

מ-`train_full.py:1237, 1246`. כל המטוסים יוצאים מהבסיסים שלהם. ה-source בלי tag (לא `[VAL ]` ולא `[EXEC]`/`[RL  ]`) כי זה לפני שהסימולציה התחילה — הקוד שולח פקודות `launch_aircraft_from_airbase` ישירות.

עם verbose יש בלוק כותרת:
```
============================================================
PRE-LAUNCH
============================================================
```

##### למה צריך pre-launch?

ב-BLADE, מטוס שב-airbase לא יכול לקבל פקודות תנועה. צריך קודם להוציא אותו (= "להמריא"). ה-pre-launch מבצע 3 דברים:
1. רץ 5 ticks ריקים — כדי שב-recording יראו את המטוסים בבסיס לפני המראה (מצב התחלתי).
2. שולח `launch_aircraft_from_airbase` לכל מטוס.
3. רץ עוד 10 ticks ריקים — כדי שהמטוסים יספיקו להתייצב באוויר.

#### 7.8 — בלוק verbose: EXECUTOR QUEUE (רק עם `--verbose`)

```
============================================================
EXECUTOR QUEUE
============================================================
  Agent be31019b-...: 1 assignments
    task=0, step=0, level=0, target=e3626956-...
  Agent 0a14f756-...: 1 assignments
    task=1, step=0, level=0, target=5880c13a-...
```

מ-`train_full.py:1260-1269`. מציג מה ה-`BladeExecutorMinimal` הולך לבצע (לפי ה-partial plan). זאת הרשימה של המשימות שהוא יקצה לכל סוכן.

כשהסימולציה תתחיל, ה-executor יעבור על הרשימה הזאת ויבצע אחת אחרי השנייה (תחנה 8).

#### 7.9 — בלוק verbose: ORACLE SETUP (רק עם `--verbose`)

```
============================================================
ORACLE SETUP
============================================================
  Partial target IDs (known): {'5880c13a-...', 'e3626956-...'}
  Full targets for be31019b-...: {'6c6f7990-...', 'e3626956-...'}
    → Agent should learn to attack: {'6c6f7990-...'}
  Full targets for 0a14f756-...: {'6c6f7990-...', '5880c13a-...', 'e3626956-...'}
    → Agent should learn to attack: {'6c6f7990-...'}
```

מ-`train_full.py:1287-1297`.

זה ה-blueprint של מה ה-RL צריך ללמוד. לכל סוכן — אילו מטרות יש לו ב-full plan, ואילו מהן הוא **לא** מכיר עכשיו (ולכן יצטרך לתקוף אותן רק אחרי שיגלה).

#### 7.10 — SIMULATION START + Utility setup (DEBUG עם verbose)

```
============================================================
SIMULATION START
============================================================
  Utility map: {'e3626956-...': 80, '5880c13a-...': 80, '6c6f7990-...': 80}
  Max utility: 80
  Oracle total utility: 240.0
```

מ-`train_full.py:1298-1323`.

##### משמעות השדות

| שדה | משמעות |
|---|---|
| `Utility map` | מילון של target_id → utility. במקרה זה כל מטרה שווה 80 נקודות |
| `Max utility` | הערך המקסימלי של מטרה אחת — 80 |
| `Oracle total utility: 240.0` | סך ה-utility שהאוראקל יכול להשיג (3 מטרות × 80 = 240) |

ה-`Oracle total utility` הוא **המשמעותי ביותר** — זה ה-baseline. אחרי האפיזודה, ה-RL יחושב כמה הוא השיג מתוך ה-240, ויקבל ratio (תחנה 9).

##### איפה ה-utility נקבע?

ה-utility נקבע ב-`scenario_factory.py` כשהוא יוצר את ה-Task objects מתצפית BLADE. בדוגמה הזו כל המטרות זהות (Red Airbase) ולכן כולן 80. בתרחישים יותר מורכבים אפשר לתת utilities שונות.

### בעיות שעלולות להופיע

##### `Partial solution empty, skipping episode` (WARNING)
```
17:xx:xx | WARNING | train_full | Partial solution empty, skipping episode
```
מ-`train_full.py:1200`. הפותר MATCH-AOU על partial לא הצליח למצוא פתרון. זה אומר שהאפיזודה כולה נדחית — אין RL phase. ה-metrics יהיו ריקים.

מתי זה יקרה? אם partial set יוצא **ללא מטרות** (`PARTIAL_RATIO * num_tasks = 0`) — זה לא אמור לקרות כי הקוד מבטיח לפחות 1 partial. או אם הסולבר נכשל מסיבה כלשהי.

##### `Discovery chain (split): isolated=N exceeds partial budget=M` (WARNING)

```
17:xx:xx | WARNING | train_full | Discovery chain (split): isolated=4 exceeds partial budget=2; 2 isolated target(s) will be hidden and undiscoverable
```
מ-`train_full.py:392`. יש יותר מטרות מבודדות (בלי שכן ראדאר) ממה שאפשר להכניס ל-partial. חלקן יישארו hidden — אבל בלי דרך להתגלות.

זה דגל אדום אם זה קורה הרבה. סימן ש-discovery chain ב-scenario_generator לא עובד טוב או שיש קונפיגורציה לא נכונה.

##### `Discovery chain (split): no valid split after 20 attempts` (WARNING)

```
17:xx:xx | WARNING | train_full | Discovery chain (split): no valid split after 20 attempts; some hidden targets may have no known radar neighbour
```
מ-`train_full.py:445`. אחרי 20 ניסיונות אקראיים, הקוד לא הצליח למצוא split תקין. יישאר אחרון, ייתכן שחלק ממטרות מוסתרות לא יתגלו.

נדיר אבל אפשרי בתרחישים עם הרבה מטרות "מבודדות".

### מה ייכתב לדיסק

**שום דבר חדש בשלב הזה.** קבצי scenario נכתבו בתחנה 5, recordings יתחילו בקרוב (בלולאה של תחנה 8).

### עלות זמן

הפתרון של MATCH-AOU תופס את רוב הזמן בשלב הזה — בערך **2-3 שניות לכל פתרון**, סך הכל ~5 שניות לאפיזודה ל-RL setup.

זה כולל את ה-validation phase (תחנה 6) שעשתה כבר solve אחד. לכן באפיזודות validation, יש 3 קריאות MATCH-AOU בסך הכל — full ל-validation, partial ל-RL, full ל-oracle.

### הערות מעשיות לקראת ריצת 5000

##### 1. רוב הפלט שקט בלי verbose
מתוך כל החגיגה הזו, **בלי verbose תראה רק WARNINGS** (אם היו). הסיכום הקומפקטי, ה-task split, ה-pre-launch, ה-verbose blocks — כולם DEBUG.

##### 2. ה-anchor לחיפוש ב-`training.log`
```bash
grep -nE "Discovery chain.*exceeds|no valid split" training.log    # split warnings
grep -n "Partial solution empty" training.log                        # episodes שדולגו
```

##### 3. דברים לעקוב אחריהם בריצת 5000
- **תדירות `warn-fallback` ו-`exhaust` ב-task split** — אם מעל 5% מהאפיזודות, יש בעיה גיאומטרית.
- **`Partial solution empty, skipping episode`** — נדיר, אבל אם קורה, זה bug אמיתי.

##### 4. ניסיון ה-MATCH-AOU השני בכל אפיזודה
זה גמר זמן יקר (3 שניות × 5000 = 4 שעות נטו רק על oracle solves). אם תחליט בעתיד לחסוך — אפשר לחשוב על cache (אותו תרחיש = אותה תוצאה אוראקל). אבל זה לא רלוונטי לריצה הנוכחית.

---

## תחנה 8: RL Simulation Loop

זה **הלב של המערכת.** כל מה שעשינו עד עכשיו — יצירת תרחיש, validation, MATCH-AOU, pre-launch — היה **הכנה** לרגע הזה. כאן ה-RL agent באמת עושה משהו.

הסימולציה רצה עד 14,400 ticks (max_ticks). בכל tick הקוד עושה את אותם דברים — אבל **רוב הזמן ה-RL לא מתערב**. הוא מתעורר רק כשקורה אירוע מסוים.

### הקונספט המרכזי: Event-Driven RL

ראית את זה בתחנה 2:
```
RL trigger: event-driven (discovery + fuel damage)
```

זה קריטי להבנת התחנה הזו. הסוכן **לא מחליט בכל tick.** הוא מחכה לאחד משני אירועים:

1. **Discovery** — סריקת ראדאר חשפה מטרה חדשה (כל 50 ticks).
2. **Fuel damage** — אירוע נזק לדלק מתרחש על אחד הסוכנים (כל tick — מיד).

ברירת המחדל היא ש-`BladeExecutorMinimal` (שהוקם בתחנה 7) מבצע את ה-partial plan באופן **אוטומטי**. הוא שולח MOVE, ATTACK, RTB ב-ticks המתאימים. ה-RL קופץ למסך רק אם משהו "חדש" קורה, ואז יש לו אופציה לעקוף את ה-executor.

### למה event-driven ולא every-tick?

זאת החלטת עיצוב מרכזית. הסיבות:

1. **רעש לאלגוריתם** — רוב ה-ticks אין שום דבר חדש. הסוכן רק טס. בלי trigger אין מה ללמוד.
2. **יקר חישובית** — `build_observation_vector()` הוא יקר. עשייה של זה 14,400 פעמים × 5,000 אפיזודות = 72 מיליון קריאות.
3. **הגיון אקדמי-אמיתי** — בעולם האמיתי החלטות תכנון לא נעשות כל שנייה. הן נעשות כשמשהו משתנה.

### מתי זה מופיע

- **על הקונסול:** רוב התוכן DEBUG (רק עם verbose). חלק מהאירועים נדירים מספיק שיופיעו במלואם בפר-אפיזודה log.
- **ב-`training.log`:** אותו דבר.
- **ב-`episode_NNNN.log`:** הכל DEBUG (תמיד).
- **בתיקיית `recordings/`:** קובץ `ep<NNN>_rl Recording...jsonl` נכתב **אם** `should_record=True`.

### החלוקה לחלקים

הלולאה משלבת 6 רכיבים שכדאי לראות בנפרד:

1. **Executor action** — ה-`BladeExecutorMinimal` מחליט מה לעשות ב-tick הזה
2. **Event detection** — האם יש fuel damage? האם זה scan tick ויש discovery?
3. **Observation building** — בונים observation אם צריך
4. **RL decision** (רק אם trigger) — האם לעקוף את executor?
5. **Action execution** — שולחים ל-BLADE
6. **Periodic logging + RTB tracking** — מעקב אחרי מצב

### הפלט המלא — אפיזודה 1 (ללא fuel damage), בסדר ההופעה

#### 8.1 — Fuel damage planning (DEBUG)

הראשון שמופיע ב-RL phase, **לפני שהלולאה מתחילה**:

```
17:17:07,646 | DEBUG | fuel_damage | Fuel damage: no damage this episode (dice roll)
```

או אם הוגרל נזק:

```
17:18:10,628 | DEBUG | fuel_damage |   Fuel damage planned: agent=266942a5.. tick=7827 factor=0.29
```

מ-`fuel_damage.py:137, 161`. הקוד "מגלגל קוביה" בתחילת כל אפיזודה כדי להחליט אם יקרה fuel damage event, ואם כן — באיזה tick ולאיזה סוכן.

##### למה זה קיים?

זאת המכניקה של "אירועי הפתעה בזמן אמת". הסוכן לומד תוכנית מבוססת על fuel budget מלא — אבל בעולם האמיתי דברים לא תמיד הולכים לפי התכנית. fuel damage מסמל "פגיעת אויב גרמה לדליפת דלק" או "תקלה טכנית הפחיתה יעילות". ה-RL צריך **להגיב** לזה — אולי לחזור מיד לבסיס במקום להמשיך למשימה הבאה.

##### השדות

`Fuel damage planned: agent=266942a5.. tick=7827 factor=0.29`:

| שדה | משמעות |
|---|---|
| `agent=266942a5..` | על איזה סוכן |
| `tick=7827` | באיזה tick יקרה הנזק |
| `factor=0.29` | הדלק יורד ל-29% מהערך הנוכחי |

#### 8.2 — Periodic progress (DEBUG, כל 1000 ticks)

```
17:17:08,452 | DEBUG | train_full |   ── Tick  1000 ── airborne: 2/2 | RL decisions: 0 | reward: +0.00 | targets attacked: 0/3
17:17:09,199 | DEBUG | train_full |   ── Tick  2000 ── airborne: 2/2 | RL decisions: 0 | reward: +0.00 | targets attacked: 0/3
```

מ-`train_full.py:669-675` (הפונקציה `_log_progress`).

מודפס **כל 1000 ticks** (`PROGRESS_LOG_INTERVAL = 1000`). זאת תמונת מצב קומפקטית.

##### השדות

| שדה | משמעות |
|---|---|
| `Tick 1000` | tick נוכחי |
| `airborne: 2/2` | כמה מטוסים באוויר מתוך הסך |
| `RL decisions: 0` | כמה החלטות RL התקבלו עד כה |
| `reward: +0.00` | סך ה-reward המצטבר |
| `targets attacked: 0/3` | כמה מטרות הותקפו עד כה (מתוך הסך) |

זה הקצב שבו הסימולציה "מתקדמת" — אתה רואה תקף עם RL=0 כל עוד אין discovery. ברגע שמתחיל לקרות משהו, ה-counters יקפצו.

#### 8.3 — Discovery event (DEBUG)

זה **אחד משני הטריגרים** ל-RL. דוגמה:

```
17:17:09,409 | DEBUG | train_full |   Tick  2250 DISCOVERY: agent be31019b.. sees target 6c6f7990..
```

מ-`train_full.py:1413-1415`.

##### מתי זה קורה?

הקוד עושה `discovery scan` **רק כל 50 ticks** (לא כל tick). אם זה scan tick (2250 % 50 == 0), הקוד בונה observation לכל סוכן, ובודק אם יש מטרה ב-observation שלא הייתה ב-partial plan המקורי.

`processed_discoveries` מבטיח שכל מטרה תיגרור trigger רק **פעם אחת** לכל סוכן. אם הסוכן ראה את `6c6f7990` ב-tick 2250, הוא לא יקבל trigger נוסף עליה ב-tick 2300.

##### השדות

`Tick 2250 DISCOVERY: agent be31019b.. sees target 6c6f7990..`:

- **`Tick 2250`** — תמיד מתחלק ב-50 (`DISCOVERY_SCAN_INTERVAL`)
- **`agent be31019b..`** — הסוכן שגילה (8 תווים)
- **`target 6c6f7990..`** — המטרה שהתגלתה (8 תווים)

מיד אחרי השורה הזאת, יודפס `RL DECISION` (8.4 בהמשך).

#### 8.4 — RL Decision (DEBUG)

**הקטע המרכזי של כל המערכת.** זה הרגע שבו ה-RL agent עושה משהו:

```
17:17:09,516 | DEBUG | train_full |   Tick  2250 RL DECISION: be31019b.. | trigger=discovery | RL=NOOP Oracle=ATTACK_1 Match=✗ Reward=-1.00 (rl_u=0, oracle_u=80)
```

מ-`train_full.py:1497-1505`.

זה השדה הכי חשוב במסמך הזה. בוא נפרק אותו לעומק:

##### השדות

| שדה | משמעות |
|---|---|
| `Tick 2250` | מתי ה-decision התקבל |
| `RL DECISION: be31019b..` | על איזה סוכן ה-RL החליט |
| `trigger=discovery` | מה גרם להחלטה? `discovery` או `fuel_damage` |
| `RL=NOOP` | מה ה-RL בחר. אופציות: `NOOP`, `ATTACK_0`, `ATTACK_1`, `ATTACK_2`, `RTB` |
| `Oracle=ATTACK_1` | מה האוראקל ("הצעיף החכם") היה בוחר |
| `Match=✗` | האם RL == Oracle? `✓` אם כן, `✗` אם לא |
| `Reward=-1.00` | ה-reward שניתן להחלטה הזאת |
| `(rl_u=0, oracle_u=80)` | utilities — מה כל אחד היה משיג |

##### חמש האפשרויות לפעולה

זה ה-**action space** של ה-RL agent. 5 אופציות:

| Action | משמעות |
|---|---|
| `NOOP` (0) | אל תעשה כלום. תן ל-executor להמשיך עם התוכנית המקורית |
| `ATTACK_0` (1) | תקוף את המטרה ב-slot 0 (מהתצפית) |
| `ATTACK_1` (2) | תקוף את המטרה ב-slot 1 |
| `ATTACK_2` (3) | תקוף את המטרה ב-slot 2 |
| `RTB` (4) | חזור מיד לבסיס |

ה-`slot` הוא רעיון מ-observation — הסוכן רואה את 3 המטרות הקרובות אליו. ה-action בוחרת **באיזו מהן לתקוף**, לא ב-target ID ספציפי. זה נותן את ה-policy גמישות לתפקד עם כל מספר מטרות.

##### דוגמת השוואה

תסתכל על שתי דוגמאות מה-run האמיתי:

**דוגמה 1 (RL טעה):**
```
Tick 2250 RL DECISION: be31019b.. | trigger=discovery | RL=NOOP Oracle=ATTACK_1 Match=✗ Reward=-1.00 (rl_u=0, oracle_u=80)
```
ה-RL בחר `NOOP` (לא לעשות כלום), בעוד שהאוראקל היה תוקף. ההפסד: 80 utility. ה-reward שלילי — מענישים אותו.

**דוגמה 2 (RL הסכים עם האוראקל):**
```
Tick 2700 RL DECISION: 0a14f756.. | trigger=discovery | RL=ATTACK_0 Oracle=ATTACK_0 Match=✓ Reward=+1.00 (rl_u=80, oracle_u=80)
```
ה-RL בחר `ATTACK_0`, אותו דבר כמו האוראקל. שניהם משיגים 80 utility. ה-reward חיובי — מתגמלים אותו.

##### למה Action `NOOP` לפעמים נכון?

לא כל discovery דורש פעולה. אם הסוכן בדרך לתקוף מטרה X, ופתאום הוא מגלה מטרה Y שתיקח אותו ממסלולו — אולי **NOOP זה הנכון** (תמשיך עם התוכנית, תתקוף Y בסיבוב הבא או בעזרת סוכן אחר).

האוראקל יודע מה הצעד האופטימלי כי יש לו את ה-full plan. ה-RL צריך ללמוד **לחקות את ההחלטה הזאת** רק על סמך התצפית הלוקאלית שלו.

#### 8.5 — Action invalid (DEBUG, נדיר)

```
17:xx:xx | DEBUG | train_full |   RL action 1 invalid for be31019b: target slot 0 has no target
```

מ-`train_full.py:1518`.

ה-RL בחר action שלא חוקי במצב הנוכחי (למשל `ATTACK_0` כשאין מטרה ב-slot 0, או `RTB` כשהסוכן כבר נחת). הקוד נופל ל-fallback — לא שולח action override, ה-executor ממשיך כרגיל.

זה לא אמור לקרות הרבה בגלל ה-`action_mask` שמסנן actions לא חוקיים מההתחלה. אם זה כן קורה, סימן לבאג ב-mask.

#### 8.6 — Fuel damage activation (DEBUG)

```
17:17:27,113 | DEBUG | fuel_damage |   *** FUEL DAMAGE at tick 3845: agent=a80f591e.. fuel reduced to 30% ***
```

מ-`fuel_damage.py:188`.

ב-tick שמתוכנן (כפי שראינו ב-8.1), הנזק מתרחש. ה-fuel של הסוכן מתעדכן ב-observation, ומיד אחרי זה יוצא **RL DECISION עם trigger=fuel_damage**.

מהריצה האמיתית:
```
*** FUEL DAMAGE at tick 6802: agent=be31019b.. fuel reduced to 23% ***
Tick  6802 RL DECISION: be31019b.. | trigger=fuel_damage | RL=RTB Oracle=NOOP Match=✗ Reward=+0.00 (rl_u=0, oracle_u=0)
Tick  6802 [RL  ] RTB:    agent be31019b..
```

ה-RL החליט `RTB` (לחזור מיד). האוראקל היה ממשיך כרגיל (`NOOP`). אין match — אבל ה-reward 0 כי שניהם השיגו 0 utility באירוע הזה (זה לא היה moment של utility, אלא של survival).

**לב המקרה:** השורה השלישית `[RL  ] RTB` מראה שה-action האמיתי שנשלח ל-BLADE היה ה-RL override, לא executor. ה-tag `[RL  ]` (4 תווים, padded) מבדיל בין שני המקורות.

#### 8.7 — BLADE actions (DEBUG)

הפעולות שנשלחות ל-BLADE בכל tick — אלה המוכרות מ-validation phase:

```
Tick     0 [EXEC] MOVE:   agent 0a14f756.. → (37.46175940933924, 38.749287831649916)
Tick  2140 [EXEC] ATTACK: agent be31019b.. → target e3626956..
Tick  2341 [EXEC] RTB:    agent be31019b..
Tick  2250 [RL  ] ATTACK: agent be31019b.. → target 6c6f7990..
```

מ-`train_full.py:1544`.

##### ההבדל בין `[EXEC]` ל-`[RL  ]`

זוכר את התגים מתחנה 6? בתחנה 8 רואים את שניהם:

- **`[EXEC]`** — ה-`BladeExecutorMinimal` שלח את הפעולה. זאת התוכנית המקורית.
- **`[RL  ]`** — ה-RL agent עקף את ה-executor עם override. זה קרה כתגובה לטריגר.

הלוגיקה ב-`train_full.py:1539, 1543`:
```python
final_action = rl_override_action if rl_override_action else executor_action
source = "RL" if rl_override_action else "EXEC"
```

אם RL בחר משהו לא-NOOP, זה מתבצע. אחרת ה-executor ממשיך.

#### 8.8 — Agent landed (DEBUG)

```
17:17:30,098 | DEBUG | train_full |   Tick  1024 RTB:     agent a80f591e.. landed
```

מ-`train_full.py:1555`.

זוכר את ההבחנה מתחנה 6 בין `[VAL ] RTB:` (פקודה נשלחה) לבין `VAL RTB ... landed` (סוכן נחת)? אותו דבר כאן, רק בלי הקידומת `VAL`. הקוד עוקב כל tick אם סוכן נעלם מ-`airborne_ids`, ואז מסמן אותו כנחת.

ההבדל בין ה-`Tick X [EXEC] RTB:` (פקודה) ל-`Tick Y RTB: agent ... landed` (נחת) יכול להיות אלפי ticks.

#### 8.9 — END-ZONE diagnostic block (DEBUG, רק במקרי timeout)

זה **בדיוק כמו ב-validation phase** (תחנה 6.7), רק עם הסיומת `[END-ZONE]` במקום `[VAL END-ZONE]`:

```
── Tick 14290 [END-ZONE] ── remaining=110 | airborne=2 | returned=0/2 | terminated=False | truncated=False
    B-2 Spirit (id=be31019b..): pos=(37.51,38.20) fuel=15234 rtb=False route_pts=12
```

מ-`train_full.py:1564-1583`. **רק אם הסימולציה הגיעה ל-100 הטיקים האחרונים**, וכל 10 טיקים. כלי דיבוג חיוני אם יש !TIMEOUT.

#### 8.10 — סוף האפיזודה (DEBUG)

יש שני סופים אפשריים:

##### סוף מוצלח: כל הסוכנים חזרו

```
17:18:37,072 | DEBUG | train_full |   Tick 12891 RTB:     agent be31019b.. landed
17:18:37,076 | DEBUG | train_full |   All agents returned to base at tick 12891 — ending episode
```

מ-`train_full.py:1587`. הקוד שובר את הלולאה ברגע ש-`returned_agents == attacking_agents`. זה הסוף הנפוץ.

##### סוף בכפייה: terminate/truncate

```
17:xx:xx | DEBUG | train_full |   Episode ended at tick 14399: terminated=False, truncated=True (env step count ≈ 14416)
```

מ-`train_full.py:1592-1596`. אם BLADE החזירה `terminated=True` או `truncated=True`, האפיזודה מסתיימת בכפייה. `truncated=True` בדרך כלל אומר שהגענו ל-`max_ticks` — וזה ה-`!TIMEOUT` שהוזכר.

`env step count` הוא ה-step counter של gymnasium (כולל את ה-pre-launch ticks הריקים). זה לא חופף ב-100% ל-tick של BLADE.

### בעיות שעלולות להופיע

##### `Tick X: Executor error (skipping): ...` (DEBUG)
```
Tick 2156: Executor error (skipping): no assignment available
```
מ-`train_full.py:1351`. ה-`BladeExecutorMinimal.next_action` זרק exception. הקוד לא מתקלקל — פשוט שולח action ריק (`""`) ל-BLADE. הסיבות הנפוצות: כל ה-assignments הסתיימו, או כל הסוכנים נחתו.

##### `Tick X: Can't observe agent_id: ...` (DEBUG)
```
Tick 5234: Can't observe a80f591e: target slot mismatch
```
מ-`train_full.py:1400`. בניית ה-observation נכשלה. סוכן נחת? התרסק? קיצוני אבל אפשרי.

### מה ייכתב לדיסק

| נתיב | מתי |
|---|---|
| `training_output/recordings/ep<NNN>_rl Recording...jsonl` | רק אם `should_record=True` |
| `training_output/recordings/ep<NNNN>_flagged_<TAGS>_rl Recording...jsonl` | אם האפיזודה carrier flag, ו-`should_record` היה False (replay נפרד) |

### עלות זמן

הסימולציה היא **החלק היקר ביותר** בריצה. בריצת הדגימה, אפיזודה ממוצעת לקחה ~5-15 שניות. אפיזודה עם `!TIMEOUT` (מגיעה ל-14,400 ticks) לוקחת ~30-60 שניות.

בריצת 5000 — סך הכל **כמה שעות עד יום שלם** של חישובי סימולציה. רוב הזמן הולך לבניית observations והרצת BLADE step אחר step.

### הערות מעשיות לקראת ריצת 5000

##### 1. ה-RL DECISION lines הם הזהב
זאת השורה היחידה שמראה לך **למה ה-RL מקבל את ה-reward שמקבל**. בריצת 5000 ללא verbose **לא תראה אותן ב-`training.log`** — אבל הן ב-`episode_NNNN.log` עם DEBUG מלא.

##### 2. ה-anchor לחיפוש
```bash
grep "RL DECISION" episode_0247.log     # כל ההחלטות באפיזודה ספציפית
grep -c "Match=✓" episode_0247.log      # ספירת match-ים באפיזודה
grep -c "Match=✗" episode_0247.log      # ספירת mis-match באפיזודה
grep "FUEL DAMAGE" training.log         # מתי קרו fuel damage events (DEBUG, רק עם verbose)
```

##### 3. אם רוב ההחלטות הן `Match=✗` בריצה הראשונה — זה תקין
ה-RL מתחיל אקראי לגמרי. סביר ש-80% מההחלטות הראשונות יהיו mismatch. אם אחרי 1000-2000 אפיזודות זה לא משתפר משמעותית — יש בעיה. (תחנה 10 — Progress block — תעזור לראות את זה.)

---

## תחנה 9: Episode End

זה **סוף האפיזודה** — חישוב סופי של utility, עדכון PPO, ייצוא recording, והדפסת שורות הסיכום הקומפקטיות. כל זה קורה ברצף אחרי ש"All agents returned to base" או אחרי timeout.

זאת התחנה שתעקוב אחריה בעיקר בריצת 5000 — שורות הסיכום הן **שורות ה-INFO** היחידות שיופיעו על המסך לכל אפיזודה.

### מתי זה מופיע

- **על הקונסול:**
  - חישובי utility, PPO update, recording export — DEBUG (רק עם verbose).
  - **שורות הסיכום הקומפקטיות (2 שורות) — INFO** (תמיד נראות, גם בלי verbose).
- **ב-`training.log`:** אותו דבר.
- **ב-`episode_NNNN.log`:** הכל DEBUG. **שורות הסיכום נכתבות *אחרי* שה-handler נסגר** — כלומר הן **לא** נכנסות ל-`episode_NNNN.log`. זה מכוון (תחנה 4).

### מה תראה (מ-`run_capture.log`, אפיזודה 1)

```
17:17:12,618 | DEBUG | train_full |   Episode utility: achieved=80 / oracle=240 (ratio=0.33) → ep_reward=+1.67
17:17:13,035 | DEBUG | train_full |   PPO update: policy_loss=-0.0120, value_loss=5.6369, entropy=1.2424, clip_frac=0.000
17:17:13,050 | DEBUG | train_full |   Recording exported: ep001_rl
17:17:13,050 | WARNING | train_full | !TIMEOUT ep0001 [VAL]  ag=2 tg=3[3e+0s]  L1:e=2/3+0iso s=0/0+0iso  L2:clean         split=2/3  ou=160/240
17:17:13,051 | WARNING | train_full | !TIMEOUT ep0001 [VAL]  RL=2d[A1 R0 N1] m=1/2  hit=1/3  RTB=Y  fd=0/0  t= 5294  r= +1.67  u= 33%
17:17:13,144 | DEBUG | ppo_trainer | Saved PPO checkpoint: training_output\models\checkpoint_ep1.pt
```

### פירוק לפי שלבים

#### 9.1 — Episode utility computation (DEBUG)

```
Episode utility: achieved=80 / oracle=240 (ratio=0.33) → ep_reward=+1.67
```

מ-`train_full.py:1626-1631`.

##### השדות

| שדה | משמעות |
|---|---|
| `achieved=80` | סך ה-utility שה-RL השיג (`rl_attacked_target_ids` × utility per target) |
| `oracle=240` | ה-`oracle_total_utility` שראינו בתחנה 7 |
| `ratio=0.33` | היחס. 80/240 = 33% |
| `ep_reward=+1.67` | ה-episode-end reward **בלבד** (לא כולל step rewards). מחושב: `ratio × episode_reward_scale` |

**איך הגענו ל-1.67?** ה-`episode_reward_scale` הוא **5.0** (`reward.py:69`), אז: `0.33 × 5.0 ≈ 1.67`.

##### ההבדל בין step rewards לעומת episode rewards

יש שני סוגי rewards שמתווספים:

1. **Step rewards** — ניתנים לכל RL DECISION בנפרד (תחנה 8.4). למשל +1.0 ל-Match=✓ ו--1.0 ל-Match=✗.
2. **Episode-end reward** (`ep_reward`) — ניתן בסוף האפיזודה על סמך ה-utility ratio. הוא מתווסף ל-**transition האחרון** ב-buffer.

הרעיון: step rewards מלמדים את ה-RL **לחקות את האוראקל** ברגע ספציפי. episode-end reward מלמד אותו **להשיג תוצאה** — לא משנה כמה החלטות נכונות עשית, אם בסוף הצלחת לתקוף רק 1/3 מהמטרות זה לא מספיק.

**ה-`episode_reward_scale` קובע את המאזן בין Imitation ל-Outcome.** כרגע 5.0 — ריצה מושלמת נותנת ep_reward=5.0, בסדר גודל דומה לסכום step rewards של ~5 החלטות מושלמות. שני האותות מתחרים על תשומת הלב של הסוכן באופן מאוזן.
- אם תגדיל את ה-scale, אות ה-Outcome יבלוט יותר — הסוכן יתעדף תוצאה כוללת גם במחיר החלטות בודדות לא-תואמות.
- אם תקטין, אות ה-Imitation יבלוט יותר — הסוכן יתמקד בלחקות את האוראקל בכל צעד.

#### 9.2 — PPO update (DEBUG)

```
PPO update: policy_loss=-0.0120, value_loss=5.6369, entropy=1.2424, clip_frac=0.000
```

מ-`train_full.py:1644-1648`. זה הרגע שבו ה-RL **באמת לומד** מהאפיזודה.

##### מה קורה ברקע

ה-PPO trainer:
1. לוקח את כל ה-transitions שנאספו ב-`buffer` במהלך האפיזודה
2. מחשב advantages עם GAE (Generalized Advantage Estimation)
3. רץ כמה epochs של gradient descent על ה-actor ו-critic
4. מחזיר metrics

##### השדות

| שדה | משמעות | טווח טיפוסי | משמעות הערך |
|---|---|---|---|
| `policy_loss` | ה-loss של ה-actor (מינוס expected return) | קרוב ל-0 | שלילי = השיפור הצליח, חיובי = יתעלם |
| `value_loss` | ה-loss של ה-critic (MSE על value estimate) | 0-10 | יורד עם הזמן ככל שהcritic לומד |
| `entropy` | exploration measure | 1.0-1.6 | גבוה = sample הרבה אפשרויות, נמוך = converge |
| `clip_frac` | חלק ה-updates שנקצצו (PPO clip) | 0-0.5 | גבוה = updates גדולים, יציבות נמוכה |

ה-`entropy` חשוב במיוחד — בתחילת הריצה הוא יהיה גבוה (~log(5) = 1.6). ככל שה-policy מתכנס, הוא יורד. **אם הוא נופל מהר מדי**, זה אומר שה-RL מתכנס לפתרון תת-אופטימלי (premature convergence).

#### 9.3 — Recording exported (DEBUG)

```
Recording exported: ep001_rl
Recording exported to 'training_output\recordings/ep001_rl Recording 064510 - 081649.jsonl'
```

מ-`train_full.py:1666` ו-BLADE's `print()`. רק אם `should_record=True`. שתי שורות:
- ראשונה: DEBUG מהקוד שלנו, לתוך ה-logging system
- שנייה: `print()` ישירות מ-BLADE, ל-stdout

ההבחנה הזאת זהה למה שראינו ב-validation (תחנה 6.10).

#### 9.4 — שורות הסיכום הקומפקטיות (INFO!)

זה **הקטע החשוב ביותר ב-5000 episodes ללא verbose** — זה כמעט הכל שתראה על המסך לכל אפיזודה:

```
ep0001 [VAL]  ag=2 tg=3[3e+0s]  L1:e=2/3+0iso s=0/0+0iso  L2:clean         split=2/3  ou=160/240
ep0001 [VAL]  RL=2d[A1 R0 N1] m=1/2  hit=1/3  RTB=Y  fd=0/0  t= 5294  r= +1.67  u= 33%
```

מ-`train_full.py:1879-1908`.

##### חשוב: prefix של flags

אם האפיזודה carrier flags, **כל שתי השורות** מקבלות prefix משולב, **ורמת ה-log עולה ל-WARNING**:

```
!TIMEOUT ep0001 [VAL]  ag=2 tg=3[3e+0s]  L1:...
!TIMEOUT ep0001 [VAL]  RL=2d[...]  hit=1/3  ...
```

זה מקל על חיפוש אפיזודות בעייתיות עם `grep "!"` ב-`training.log`.

##### Line 1 — מטא-דאטא של התרחיש

```
ep0001 [VAL]  ag=2 tg=3[3e+0s]  L1:e=2/3+0iso s=0/0+0iso  L2:clean         split=2/3  ou=160/240
```

| שדה | משמעות |
|---|---|
| `ep0001` | מספר האפיזודה (1-indexed) |
| `[VAL]` | התקיים validation phase. אם לא — `      ` (6 רווחים) |
| `ag=2` | מספר סוכנים |
| `tg=3` | מספר מטרות |
| `[3e+0s]` | חלוקה ל-zones — 3 easy + 0 stretch |
| `L1:e=2/3+0iso s=0/0+0iso` | תוצאות discovery chain (תחנה 5). easy: 2 מתוך 3 מטרות **הוזזו במיקום**, 0 נשארו isolated. stretch: היה 0/0+0. זאת לא הסתרת מטרות (ראה הבהרה למטה). |
| `L2:clean` | תוצאת `split_tasks` discovery chain (תחנה 7) |
| `split=2/3` | partial=2, full=3 |
| `ou=160/240` | utilities. 160 = `partial_oracle_utility`, 240 = `full_oracle_utility` |

**הבחנה חשובה — `L1` הוא לא הסתרה:**
- ה-`L1:e/s` סופר כמה מטרות ה-`scenario_generator` **הזיז במיקום** כדי שכל מטרה תהיה בטווח רדאר של אחרת. זה תנאי הכרחי לכך שהסתרה תוכל לעבוד (מטרה נסתרת חייבת להיות גלויה דרך מטרה ידועה).
- ההסתרה עצמה קורית בשדה `split=2/3` באותה שורה: partial=2 (RL מכיר 2 מטרות), full=3 (קיימות 3 בסך הכל). כלומר 1 מטרה מוסתרת מתוך 3 — תואם ליחס ~1/3 שצפית.

##### Line 2 — תוצאות הביצוע

```
ep0001 [VAL]  RL=2d[A1 R0 N1] m=1/2  hit=1/3  RTB=Y  fd=0/0  t= 5294  r= +1.67  u= 33%
```

| שדה | משמעות |
|---|---|
| `RL=2d` | 2 RL decisions באפיזודה |
| `[A1 R0 N1]` | התפלגות actions: 1 ATTACK, 0 RTB, 1 NOOP |
| `m=1/2` | matches: 1 מתוך 2 (50% accuracy) |
| `hit=1/3` | מטרות שהותקפו: 1 מתוך 3 |
| `RTB=Y` | האם **כל** הסוכנים חזרו לבסיס? `Y`/`N` |
| `fd=0/0` | fuel damage events — `fired/planned`. 0/0 = לא הוגרל ולא נורה. 1/1 = הוגרל ונורה. |
| `t= 5294` | סך ה-ticks שלקח לאפיזודה |
| `r= +1.67` | סך ה-reward לאפיזודה (כולל episode-end) |
| `u= 33%` | utility ratio (achieved/oracle), %. כאן 33% = 80/240 |

##### דוגמה עם flag

```
!TIMEOUT,ANOMALY ep0247  ag=3 tg=5[2e+3s]  L1:e=1/2+0iso s=2/3+0iso  L2:warn-fallback  split=3/5  ou=200/400
!TIMEOUT,ANOMALY ep0247  RL=8d[A3 R5 N0] m=2/8  hit=2/5  RTB=N  fd=0/1  t=14400  r= -3.45  u= 50%
```

זאת אפיזודה עם 2 דגלים. שמות הflag-ים מופרדים בפסיקים. הdדגלים האפשריים:
- `!TIMEOUT` — האפיזודה הגיעה ל-`max_ticks` בלי שכל הסוכנים חזרו
- `!L2-fallback` — split_tasks לא הצליח אחרי 20 ניסיונות
- `!L2-exhaust` — יותר מטרות מבודדות מהתקציב
- `!ANOMALY` — validation גילתה תקיפה של מטרה לא ניתנת להשגה
- `!noPPO` — ה-buffer היה ריק, לא היה update

#### 9.5 — Checkpoint save (DEBUG)

```
Saved PPO checkpoint: training_output\models\checkpoint_ep1.pt
```

מ-`ppo_trainer.py`. נשמר מודל **כל** `--save-freq` אפיזודות. בריצת ה-5 הוא היה 1, אז כל אפיזודה שמורה.

ב-5000 עם save-freq=100, ייווצרו רק 50 checkpoints. כל אחד ~50KB.

### הערות מעשיות לקראת ריצת 5000

##### 1. שתי שורות הסיכום הן ה-feed הראשי שלך
**אלה השורות שתעקוב אחריהן בזמן אמת בריצת 5000.** הן מספיקות לדעת אם ה-training בכיוון הנכון.

##### 2. ה-anchors הקריטיים
```bash
# כל האפיזודות שיש להן flag:
grep "^!" training.log
# או רק TIMEOUTs:
grep "!TIMEOUT" training.log
# רק שורות summary:
grep -E "ep[0-9]{4}" training.log
```

##### 3. עקוב אחר utility ratio (`u=`) — זה ה-metric החשוב
- ב-100 אפיזודות הראשונות: צפוי ~30-40%
- אם אחרי 1000 אפיזודות עדיין מתחת ל-50%: ייתכן שיש בעיה
- מטרה: 70%+ אחרי 5000

##### 4. matches ratio (`m=`) חשוב פחות
ה-RL לא חייב **לחקות** את האוראקל מושלמת — הוא צריך להשיג תוצאה טובה. אם `u=70%` אבל `m=40%`, זה עדיין טוב — הוא מצא דרכים אחרות (אפילו תת-אופטימליות) להשיג utility גבוה.

---

## תחנה 10: Progress Block

זה **הבלוק שמופיע כל `--progress-every` אפיזודות** ומציג את **הטרנד** של הלמידה. זאת התחנה הכי חשובה לעקוב אחריה כדי לדעת אם ה-RL לומד.

### מתי זה מופיע

- **על הקונסול:** **תמיד** (INFO). כל `--progress-every` אפיזודות.
- **ב-`training.log`:** תמיד.
- **ב-`episode_NNNN.log`:** לא — זה אחרי שה-handler נסגר.

### מה תראה

#### בלוק ראשון (אפיזודה 1, אין delta)

```
========== Progress @ ep0001 | checkpoint saved | rolling 1ep ==========
  Reward   :  +1.67            Utility :  33.3%        Accuracy:  50.0%
  Ticks/ep :   5294             Actions : A:50.0% R: 0.0% N:50.0%   Decisions: 2.00/ep
  PPO loss : π=-0.0120  V=  5.64  H=1.242    Flags(window): !TIMEOUT=1
========================================================================
```

#### בלוק שני (אפיזודה 5, יש delta)

```
========== Progress @ ep0005 | checkpoint saved | rolling 1ep ==========
  Reward   :  +3.67  Δ +2.42 ↑    Utility :  33.3%  Δ  +8.3% ↑  Accuracy:  50.0%  Δ +16.7% ↑
  Ticks/ep :  12892  Δ  -1490 ↓     Actions : A:50.0% R:50.0% N: 0.0%   Decisions: 4.00/ep
  PPO loss : π=-0.0032  V=  2.53  H=1.267    Flags(window): (none)
========================================================================
```

מ-`train_full.py:1990-2026`.

### פירוק לפי שורות

#### Header

```
========== Progress @ ep0005 | checkpoint saved | rolling 1ep ==========
```

מציג:
- מספר האפיזודה הנוכחית
- האם נשמר checkpoint
- חלון ה-rolling — כמה אפיזודות מצטמצמות לבלוק הזה

#### Line 1 — Reward, Utility, Accuracy

| שדה | מהי |
|---|---|
| `Reward` | ה-reward הממוצע לאפיזודה בחלון הזה — **הסכום הכולל** של step rewards + ep_reward, לא רק החלק של episode-end. ראה תחנה 9.1 |
| `Δ` | השינוי ביחס לחלון הקודם |
| `↑` / `↓` | חץ מקודד צבע (אדום=ירוד, ירוק=עלייה) — ב-stdout רק ASCII ↑/↓ |
| `Utility` | utility ratio ממוצע (אחוזים) |
| `Accuracy` | matches/decisions ממוצע (אחוזים) |

זאת **השורה החשובה ביותר** של כל בלוק progress. שלושה metrics:
- **Reward** עולה → המודל לומד להשיג episode-end + step rewards
- **Utility** עולה → המודל מצליח להשיג מטרות
- **Accuracy** עולה → המודל מתחיל לחקות את האוראקל

#### Line 2 — Ticks, Actions, Decisions

| שדה | מהי |
|---|---|
| `Ticks/ep` | ממוצע ticks לאפיזודה |
| `Actions` | התפלגות action types — A=ATTACK, R=RTB, N=NOOP |
| `Decisions` | ממוצע RL decisions לאפיזודה |

##### מה אתה רוצה לראות

**Ticks/ep יורד עם הזמן** — סימן שהסוכן מסיים את משימותיו מהר יותר.

**Actions בריאים: A:60% R:30% N:10%** — הסוכן בעיקר תוקף, חוזר כשצריך, לא יותר מדי NOOPs. אם NOOPs > 50% — הסוכן "נרדם" יותר מדי.

**Decisions גדל עם הזמן** — סימן שהסוכן מקבל יותר triggers (יותר discoveries, יותר fuel damage events).

#### Line 3 — PPO loss + Flags

| שדה | מהי |
|---|---|
| `π` | policy_loss |
| `V` | value_loss |
| `H` | entropy |
| `Flags(window)` | מילון של !FLAG=count בחלון |

##### מה אתה רוצה לראות לאורך הריצה

| Metric | תחילת הריצה | סוף הריצה |
|---|---|---|
| `π` | קרוב ל-0 (sometimes negative) | קרוב ל-0 |
| `V` | גבוה (5-10) | יורד (1-3) |
| `H` | ~1.6 (max) | יורד ל-0.5-1.0 |
| `!Flags` | ייתכנו `TIMEOUT` | פוחתים, אולי 0 |

ה-`H` (entropy) הוא ה-בעל משמעות הגבוהה ביותר ל-monitoring. אם הוא יורד מהר מדי, ה-RL מתכנס מוקדם מדי. אם הוא לא יורד בכלל, ה-RL לא לומד.

### הגדרת `--progress-every` ל-5000

ברירת המחדל היא 50 אפיזודות. בריצת 5000:
- `--progress-every 50` → 100 בלוקים, אפשר לעקוב טרנדים בדיוק
- `--progress-every 100` → 50 בלוקים, נוח לסקור
- `--progress-every 250` → 20 בלוקים, אופייני מחקר ארוך

**ההמלצה: 100.** מספיק רזולוציה לראות שינויים, לא יותר מדי רעש.

### הערות מעשיות לקראת ריצת 5000

##### 1. זה ה-feed הראשי שלך לזיהוי אם ה-RL לומד

עקוב אחר ה-deltas:
- **אם `Δ Reward` חיובי לאורך זמן** → המודל משתפר
- **אם `Δ Utility` חיובי** → ההשפעה אמיתית
- **אם שניהם שליליים בריצוף** → משהו לא תקין

##### 2. ה-anchor לחיפוש
```bash
grep "Progress @" training.log    # כל ה-progress blocks
```

---

## תחנה 11: Training Complete

זה הבלוק שמופיע **פעם אחת בסוף הריצה**. הוא מסכם את כל הריצה ב-10 שורות INFO.

### מתי זה מופיע

- **על הקונסול:** תמיד, פעם אחת בסוף.
- **ב-`training.log`:** תמיד.
- **ב-`episode_NNNN.log`:** לא — זה אחרי האפיזודה האחרונה.

### מה תראה (מ-`run_capture.log`)

```
======================================================================
Training Complete!
======================================================================
Total episodes:      5
Total PPO updates:   5
Avg policy loss:     -0.0055
Avg value loss:      2.5751
Avg reward (last 10): 1.90
Avg accuracy (last 10): 46.7%
Avg utility ratio (last 10): 30.0%
Run summary written to: training_output\logs\run_summary.txt
Highlights written to:  training_output\logs\highlights.txt

Outputs saved to: C:\...\training_output
  Logs:       training_output\logs/
  Recordings: training_output\recordings/
  Models:     training_output\models/
  Scenarios:  training_output\scenarios/
```

מ-`train_full.py:2813-2862`.

### פירוק

#### חלק 1 — Banner וסטטיסטיקה

```
Training Complete!
Total episodes:      5
Total PPO updates:   5
Avg policy loss:     -0.0055
Avg value loss:      2.5751
```

| שדה | משמעות |
|---|---|
| `Total episodes` | כמה אפיזודות הושלמו (כולל מדולגות?) |
| `Total PPO updates` | כמה updates עשה ה-PPO trainer |
| `Avg policy loss` | ממוצע ה-policy loss על פני הריצה |
| `Avg value loss` | ממוצע ה-value loss |

#### חלק 2 — סטטיסטיקות אחרונות

```
Avg reward (last 10): 1.90
Avg accuracy (last 10): 46.7%
Avg utility ratio (last 10): 30.0%
```

ממוצע על **10 האפיזודות האחרונות**. בריצת 5 כל הריצה. בריצת 5000 זה הסיגנל ל"איך הסתיים ה-training" — אם רוב הלמידה קרתה ב-1000 הראשונים והסתיימה משם — `last 10` יתפוס את המצב הסופי.

#### חלק 3 — הקבצים החיוניים

```
Run summary written to: training_output\logs\run_summary.txt
Highlights written to:  training_output\logs\highlights.txt
```

שני קבצים שנכתבים בסוף הריצה. הם מסבירים את **כל מה שקרה בריצה** ומשמשים אינדקס. נראה אותם בתחנה 12.

#### חלק 4 — סיכום נתיבים

```
Outputs saved to: C:\...\training_output
  Logs:       training_output\logs/
  Recordings: training_output\recordings/
  Models:     training_output\models/
  Scenarios:  training_output\scenarios/
```

תזכורת איפה כל סוג קובץ.

### הערות מעשיות לקראת ריצת 5000

##### 1. זה הסיגנל ש"הסיום הצליח"

אם תראה את `Training Complete!`, הריצה הסתיימה תקין. אם הריצה נופלת באמצע (CRASH, kill), השורה הזאת לא תופיע.

##### 2. ה-`last 10` הוא ה-snapshot של ה"מצב הסופי" של הלמידה

**זה לא** ממוצע על כל הריצה. זה רק על 10 האחרונים — כדי לתפוס את המצב לאחר ההתכנסות.

---

## תחנה 12: קבצי הפלט של הריצה

זה לא תחנה לפי סדר הריצה — אלא **סקירה של כל הקבצים שנשמרו**, מה הם מכילים, ואיך להשתמש בהם.

### מבנה התיקיות

```
training_output/
├── logs/
│   ├── training.log              ← לוג ראשי לכל הריצה
│   ├── episode_0001.log          ← פר-אפיזודה DEBUG firehose
│   ├── episode_0002.log
│   ├── ...
│   ├── episode_NNNN.log
│   ├── run_summary.txt           ← סיכום מובנה לכל הריצה
│   └── highlights.txt            ← אינדקס "ראה ב-Panopticon"
├── models/
│   ├── checkpoint_ep1.pt         ← checkpoints לפי --save-freq
│   ├── checkpoint_ep100.pt
│   ├── ...
│   ├── final_model.pt            ← המודל הסופי
│   └── actor_critic_final.pt     ← כפילות, אותו תוכן
├── recordings/
│   ├── ep001_validation Recording...jsonl   ← אם validation+record
│   ├── ep001_rl Recording...jsonl           ← אם record
│   ├── ep0042_flagged_TIMEOUT_rl Recording...jsonl   ← flagged replay
│   └── ...
└── scenarios/
    ├── episode_0000_scenario.json    ← תרחיש לכל אפיזודה (0-indexed!)
    ├── episode_0001_scenario.json
    └── ...
```

### `training.log` — לוג ראשי

**הקובץ הכי שימושי.** מכיל:
- כל ההודעות INFO+ של הריצה (banner, episode summaries, progress blocks, training complete)
- אם רץ עם `--verbose`: גם DEBUG (Pyomo, RL DECISION, וכו')

**גודל מצופה ב-5000:** ~5MB ללא verbose, ~150MB עם verbose.

#### חיפושים שימושיים

```bash
grep "Training Complete" training.log         # אם הריצה הסתיימה
grep "^!" training.log                        # כל האפיזודות עם flag
grep "Progress @" training.log                # כל progress blocks
grep "ANOMALY" training.log                   # תקלות validation
grep -E "ep[0-9]{4} \[VAL\]" training.log     # רק validation episodes
```

### `episode_NNNN.log` — פר-אפיזודה DEBUG

**ה-firehose** של אפיזודה אחת. תמיד DEBUG מלא, גם בלי `--verbose`.

מכיל:
- Banner של תחילת אפיזודה
- כל היצירת התרחיש (עם רעש Pyomo)
- כל ה-validation phase (אם הייתה)
- כל ה-RL phase (כולל RL DECISIONs, fuel damage, אקטיבציות)
- ה-utility computation
- ה-PPO update
- **לא** מכיל את שורות הסיכום (handler נסגר לפני)

**מתי לפתוח את זה:**
- אפיזודה ספציפית עם flag — לראות מה השתבש
- לדבג בעיה ב-RL decision
- לראות בדיוק מה קרה ב-tick מסוים

**גודל מצופה ב-5000:** ~50KB-200KB אפיזודה רגילה, עד ~4MB עם `!TIMEOUT`. סך הכל **~1-2 GB ב-5000 אפיזודות.**

### `run_summary.txt` — סיכום מובנה

קובץ טקסט מאורגן עם:
- הגדרות הריצה (config dump)
- סטטיסטיקה כללית
- **רשימה של כל האפיזודות עם flags** (אינדקס לעיון)
- היסטוגרמת flags

זה ה-"go-to" שלך אחרי הריצה. אם רצית לדעת "איזה אפיזודות היו בעייתיות" — פותח את זה.

### `highlights.txt` — אינדקס Panopticon

**הקובץ הכי שימושי לעיון ויזואלי.**

מכיל רשימות אפיזודות "מעניינות" עם **שמות הקבצי-recording המלאים**, מאורגנות לפי קטגוריות:
- `Perfect-match RL episodes` — אפיזודות שבהן ה-RL הסכים עם האוראקל בכל ההחלטות
- `Significant-mismatch episodes` — אפיזודות עם הרבה Match=✗
- `Learning-trend samples` — דגימות מאורך הריצה (1, 1000, 2500, 5000) להשוות
- `Flagged episode index` — לפי קטגוריית flag

הרעיון: אתה פותח את הקובץ הזה, מעתיק שם של recording, ופותח אותו ב-Panopticon. בלי לחפש.

### `models/` — checkpoints

```
checkpoint_ep1.pt        ~50KB
checkpoint_ep100.pt
...
checkpoint_ep5000.pt
final_model.pt           ~50KB (זהה ל-actor_critic_final.pt)
```

נשמרים כל `--save-freq` אפיזודות (ברירת מחדל: 50).

**גודל ב-5000 עם save-freq=100:** 50 checkpoints × 50KB = **2.5MB.** קטן.

### `recordings/` — קבצי Panopticon

קבצי JSONL ש-Panopticon יודעת לפתוח. כל אחד מתאר את כל המצב של אפיזודה לאורך כל ה-ticks.

**גודל מצופה:** 1-12MB לכל recording.

ב-5000 עם:
- `--validate-every 50` ו-`--record-every 50` → 100 RL recordings + 100 validation = **~1GB**
- `--validate-every 50` ו-`--record-every 1` → 5000 RL + 100 validation = **~25GB**

### `scenarios/` — תרחישים שנוצרו

JSON של כל תרחיש שנוצר. **0-indexed!** (`episode_0000_scenario.json` = "Episode 1/N").

**גודל מצופה ב-5000:** 5000 × 12KB = **~60MB.**

### תקציר טבלת גדלים ב-5000

| תיקייה/קובץ | גודל מצופה | הערות |
|---|---|---|
| `training.log` | 5MB-150MB | תלוי ב-verbose |
| `episode_NNNN.log` (5000 קבצים) | 1-2GB | DEBUG firehose |
| `run_summary.txt` | <1MB | מובנה |
| `highlights.txt` | <1MB | קצר |
| `models/` | ~2.5MB | 50 checkpoints |
| `recordings/` | 1GB-25GB | תלוי ב-record-every |
| `scenarios/` | ~60MB | 5000 קבצים |

### הערות מעשיות

##### 1. `recordings/` הוא הצרכן הגדול ביותר של מקום
שקול את `--record-every` בקפידה. אפילו 50 שווה את הזמן.

##### 2. שיטת העבודה המומלצת אחרי הריצה
1. פתח `run_summary.txt` — סקור flags כלליים
2. פתח `highlights.txt` — בחר 5-10 אפיזודות מעניינות
3. פתח את ה-recordings שלהן ב-Panopticon
4. אם משהו צריך פירוט יותר — פתח את `episode_NNNN.log` המתאים
5. אם רוצה להשוות ביצוע ל-RL לאוראקל באותה אפיזודה — פתח גם את `validation` recording של אותה אפיזודה

##### 3. גיבוי אחרי הריצה
ה-`models/final_model.pt` הוא ה-policy שאומן. **תגבה אותו** לפני שתריץ ריצה חדשה (כי `models/` לא מנוקה אבל יכול להידרס בריצה הבאה).

---

## סיכום: ה-flow המלא של ריצת 5000

עברת על 12 תחנות. הנה תזכורת קצרה של מה תראה בריצה ממש ארוכה ללא verbose:

1. **Startup & Cleanup** — שקט (אין פלט)
2. **Run-init Banner** — 30 שורות INFO, פעם אחת
3. **Starting Training** — 3 שורות INFO, פעם אחת
4. **Episode Separator** — שקט (DEBUG)
5. **Scenario Generation** — שקט (DEBUG, אלא אם WARNING)
6. **Validation Phase** — אם רץ: 8 שורות INFO של audit block
7. **RL Setup** — שקט (DEBUG)
8. **RL Simulation Loop** — שקט (DEBUG)
9. **Episode End** — **2 שורות INFO** (הסיכום הקומפקטי) לכל אפיזודה
10. **Progress Block** — **5 שורות INFO** כל `--progress-every` אפיזודות
11. **Training Complete** — 15 שורות INFO, פעם אחת
12. **קבצי הפלט** — הקבצים הסופיים

### זמן ריצה משוער ל-5000

עם `--validate-every 50`, `--record-every 50`:
- 5000 × ~10 שניות אפיזודה = **~14 שעות** של אימון נטו
- + 100 validations × ~12 שניות = **~20 דקות**
- + 5000 PPO updates (כלולים בזמן האפיזודה)
- **סה"כ: כ-15 שעות**

עם פחות validation (`--validate-every 100`): ~14:10 שעות.

### מה לוודא לפני ריצת 5000

- [ ] להחליט על `--validate-every` (50 או 100)
- [ ] להחליט על `--record-every` (50 או 100, לא 1)
- [ ] **לא** להעביר `--debug-force-flags` (זה מזהם את הסטטיסטיקה)
- [ ] **לא** להעביר `--verbose` (יציף את הקונסול)
- [ ] לוודא שיש מקום פנוי בדיסק (~3GB מינימום, ~10GB עם recordings מלאים)
- [ ] לגבות את `models/final_model.pt` של ריצות קודמות אם הן חשובות

### כעת מוכן לריצה

המסמך הזה הוא ה-reference שלך לכל פלט. אם תראה משהו שלא ברור בריצה — חזור לתחנה הרלוונטית.

בהצלחה!

