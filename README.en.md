# DaynAC

------------------------------------------------------------------------

> \[!CAUTION\]
>
> ## Warning
>
> The project is currently in early alpha testing.

**A neural-network anti-cheat for Minecraft PvP servers.** Detects
external aim assistance (AimBot / KillAura) through behavioral hit
patterns --- without speed-hack checks, packet sniffers, or other
conventional techniques. The model is trained on datasets that you
collect yourself on your server.

The target use case is **1.9+ no-cooldown PvP**, where legitimately high
CPS is not evidence of cheating: the distinction between legitimate play
and cheating is based not on click speed, but on aiming geometry and
timing.

------------------------------------------------------------------------

## How It Works

    player hits ──► SnapshotTracker ──► HitData (21 features per hit)
                         │
                         ▼
                  FeatureExtractor: window of 16 hits
                  16×21 + 7 aggregates = 343 features
                         │
                         ▼
                  NeuralNetwork (343 → 128 → 64 → 32 → 1)
                  pure Kotlin, no ML libraries
                         │
                         ▼
                  cheat probability (0..1) ──► server response

**Features for each hit:** aim angle, distance, interval between hits
and its variation, rotation entropy (abs/signed, yaw/pitch), jitter, XZ
velocity, rotation smoothness, snap factor (sudden final adjustment of
the aim), micro-corrections, jerk, rotation linearity.

**Window aggregates:** mean angle, coefficients of variation,
angle-distribution entropy, lag-1 autocorrelations (mechanical
predictability of timing and aim), R² of the angle's linear trend
(monotonic final adjustment of the aim).

**Probability-threshold responses** (configurable in `config.yml`):

  -----------------------------------------------------------------------
  Range                               Action
  ----------------------------------- -----------------------------------
  \< flag (0.65)                      full trust

  ≥ flag (0.65)                       notify admins in chat

  ≥ reduce (0.70)                     gradually reduce damage

  ≥ cancel (0.90)                     cancel damage completely

  ≥ punish (0.95) for                 kick / ban (by UUID; IP ban is
  `punish-confirm-windows` (2)        optional)
  consecutive independent windows     
  -----------------------------------------------------------------------

False-positive safeguards: the damage multiplier is calculated from
**previous** predictions (the first hit always deals full damage).
Automatic punishment requires the threshold to be exceeded in several
**independent** (non-overlapping) windows in a row --- two by default,
or approximately 32 hits. Windows are recalculated on every hit and
overlap by 15 out of 16 hits, so "the average of the last N predictions"
does not count as confirmation. Inference runs asynchronously --- the
server's main tick is not blocked.

------------------------------------------------------------------------

## Installation

### Requirements

-   **Paper 1.21.11** server (or a newer version of the same major
    release)
-   Java **21+**

### Installing a Prebuilt JAR

1.  Copy `DaynAC-<version>-all.jar` (the `shadowJar` artifact) into the
    server's `plugins/` folder.
2.  Start the server --- the plugin will create `plugins/DaynAC/` with
    `config.yml`.
3.  Configure the thresholds for your server (see the table above) and
    restart.

**Detection is disabled until the model has been trained** --- until you
collect a dataset and run `/daynac train`, the plugin works only as a
data collector.

### Building from Source

``` bash
./gradlew build          # resulting JAR: build/libs/DaynAC-<version>-all.jar
```

To launch a local test server directly from the project:

``` bash
./gradlew runServer      # launches Paper 1.21.11 with the plugin (2 GB RAM)
```

------------------------------------------------------------------------

## Usage

All commands require the `daynac.admin` permission (granted to ops by
default).

### Collecting a Dataset

1.  Ask a trusted player (or connect a second account) to participate in
    a PvP duel.

2.  Enable recording:

        /daynac record <player> legit    # player plays legitimately
        /daynac record <player> cheat    # player uses cheats
        /daynac record <player> off      # stop recording

3.  A feature vector is recorded every 16 hits (non-overlapping windows)
    in `plugins/DaynAC/legit_vectors.csv` and `cheat_vectors.csv`.
    Players in recording mode are not checked by detection --- they are
    exempt while data is being collected.

    Two metadata columns are appended to each vector: `session_id` (one
    recording session, from `record <player> legit|cheat` until `off`)
    and `player_hash` (a pseudonym for the player: SHA-256 of the UUID
    with the salt from `dataset.salt`; the UUID itself is not written to
    the CSV). These columns are not used for training; they are used to
    split the dataset. Files in the old format are upgraded
    automatically, with a `*.bak` backup created first; old rows remain
    ungrouped. Do not lose or publish `dataset.salt`: without it, hashes
    for new recordings will not match the old ones.

For a high-quality model, you need **both classes**, preferably with
comparable numbers of examples in each (hundreds of vectors per class).
A variety of situations (distances, movement, different targets) matters
more than sheer quantity. Record **several separate sessions** for each
class (preferably with different players): at least 3 sessions per class
are needed for a reliable evaluation.

### Training

    /daynac train

-   Splits the dataset 60/20/20 into train/validation/test sets
    (stratified by class) and normalizes features using the training
    portion only. If there are ≥ 3 sessions per class, it splits by
    **whole sessions** (`training.split-by: session`, or `player` to
    split by player), so windows from the same player do not appear in
    both training and testing. If there are fewer sessions, the split is
    random by window, and the plugin warns that the metrics are likely
    inflated.
-   Trains with early stopping based on validation loss and restores the
    best weights. The test set is not used for training or epoch
    selection --- it is used to calculate final metrics and for
    `/daynac eval`.
-   At least 5 vectors per class are required (in practice, hundreds are
    recommended).
-   Progress is shown in real time: a live progress bar in the action
    bar (for the player) and progress updates in chat every 10% of the
    epochs.
-   On completion: a report with train/validation metrics and the gap
    (overfitting), the model is saved to `model.dat`, and detection is
    enabled automatically.

The training result indicates dataset quality: a large gap between train
and validation accuracy means the data is insufficient or too uniform.

### Monitoring

-   `/daynac ver <player>` --- the player's latest and average score,
    and hit count.
-   `/daynac allinfo` --- GUI (54-slot chest): all observed players,
    scores, verdicts, and recent-hit trends. Left-click the compass to
    teleport into spectator mode. Punished players automatically
    disappear from the list --- case closed.
-   `/daynac info` --- dataset statistics, model status, and current
    thresholds.
-   `/daynac eval` --- metrics on the test set across a range of
    thresholds (precision / recall / F1 / FPR), with a 95% Wilson
    confidence interval. A threshold is recommended only if the upper
    bound of FPR is ≤ 1%: even with zero errors, this requires ≥ 381
    legitimate windows in the test set (approximately 1,900 legitimate
    vectors in the full dataset).
-   `/daynac debug` --- toggle logging of every hit to operators' chat.
-   `/daynac reload` --- reload `config.yml` (thresholds take effect
    after a restart).

### Example Chat Alert

    [DaynAC] Steve: unusual hit 0.871 > threshold 0.65 — recent hits: 0.31 0.44 0.61 0.87 (average: 0.54)

------------------------------------------------------------------------

## Architecture

    src/main/kotlin/zov/grusha/daynAC/
    ├── DaynAC.kt              # entry point: events, commands, dataset recording
    ├── tracking/
    │   ├── CombatMath.kt      # geometry: aim angle, distance
    │   ├── HitData.kt         # feature snapshot for a single hit
    │   ├── HitTracker.kt      # sliding window of hit history
    │   ├── PlayerSnapshot.kt  # rotation/velocity deltas per tick
    │   └── SnapshotTracker.kt # per-tick player snapshots, rolling metrics
    ├── ml/
    │   ├── FeatureExtractor.kt# hit window → 343-feature vector
    │   ├── DatasetManager.kt  # legitimate/cheat CSV datasets
    │   └── NeuralNetwork.kt   # MLP + backprop, lock-free inference
    ├── detection/
    │   └── DetectionEngine.kt # asynchronous inference, thresholds, punishments
    └── gui/
        └── AllInfoGUI.kt      # player monitoring GUI

Key engineering decisions:

-   **Dependency-free neural network** --- MLP 343→128→64→32→1
    implemented in pure Kotlin (BCE + SGD + He initialization + L2 +
    gradient clipping). No PyTorch runtime is bundled in production, and
    Python is not required.
-   **Lock-free inference** --- weights live in an immutable snapshot
    behind an `AtomicReference`: `predict()` does not acquire locks, and
    retraining on a live server does not block detection even for a
    second.
-   **Atomic model publishing** --- `train`/`loadFrom` are serialized
    with a mutex, and the model file is written via tmp+rename so a
    server crash does not corrupt `model.dat`.
-   **Data-leakage protection** --- non-overlapping windows when
    recording the dataset, stratified splitting, and normalization using
    the training portion only.
-   **No telemetry** --- everything stays local: datasets, model, and
    logs.

## Configuration

`plugins/DaynAC/config.yml`:

``` yaml
hit-logging: false          # log every hit

detection:
  flag-threshold: 0.65      # threshold for admin alerts
  reduce-threshold: 0.70    # start reducing damage
  cancel-threshold: 0.90    # cancel damage completely
  punish-threshold: 0.95    # automatic kick/ban
  punish-confirm-windows: 2 # number of consecutive independent windows (16 hits each) that must be >= punish-threshold
  punish-enabled: true      # false = monitoring mode (alerts only)
  punish-action: kick       # none | kick | ban
  ban-with-ip: false        # ban: also ban by IP (NAT networks may affect innocent players — UUID-only bans by default)
  punish-reason: "KillAura (detected by DaynAC neural network)"
  max-recent-predictions: 10 # averaging window for damage and punishment
  inactivity-reset-seconds: 30 # inactivity: after this long without hits, flags are cleared and the window starts over (0 = disabled)
```

Threshold ordering is validated at startup:
`flag ≤ reduce < cancel ≤ punish`.

The inactivity reset (`inactivity-reset-seconds`) is checked once per
second: if a player has not hit anyone for longer than the configured
time, their flags are cleared --- active predictions are cleared, damage
is restored to full, and the hit window starts over. The next 16 hits
fill a new window, and only then will a new prediction be produced.

The log is not cleared in the process --- the player remains in
`/daynac allinfo` with "Recent Hits" and their all-time maximum. The
card shows progress for the new window (`Hits: n/16`) and the verdict
"Flags cleared after inactivity".

The reset affects only online players --- flags for players who have
left are retained until TTL cleanup (1 hour), giving admins time to
review them.

------------------------------------------------------------------------

## License

MIT
