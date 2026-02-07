from kinematic_features import compute_all_kinematic_features, detect_hsr_sprint_events
from kloppy import skillcorner
import config
import polars as pl

print("Loading dataset via Kloppy SkillCorner open data")
dataset = skillcorner.load_open_data(match_id="1925299")

# 1) Get a wide pandas DataFrame from Kloppy, then to polars
print("Converting Kloppy TrackingDataset to wide DataFrame")
pdf_wide = dataset.to_df()   # per Kloppy docs
df_wide = pl.from_pandas(pdf_wide)

# 2) Check timestamp format
print(f"Sample timestamps: {df_wide['timestamp'].head(5).to_list()}")
print(f"Timestamp dtype: {df_wide['timestamp'].dtype}")

# Convert timedelta to seconds (float) with microsecond precision
print("Converting timedelta timestamps to seconds")
df_wide = df_wide.with_columns([
    (pl.col("timestamp").dt.total_microseconds() / 1_000_000.0).alias("timestamp")
])

print(f"Converted timestamps (seconds): {df_wide['timestamp'].head(5).to_list()}")

# 3) Identify player columns (all columns ending with '_x' that are NOT ball_x)
player_x_cols = [
    col for col in df_wide.columns
    if col.endswith("_x") and col not in ("ball_x",)
]

# Derive player identifiers (the prefix before '_x')
player_ids = [col[:-2] for col in player_x_cols]  # strip '_x'

print(f"Found {len(player_ids)} players from wide columns")

# 4) Reshape wide → long: one row per player per frame
long_dfs = []
for pid in player_ids:
    # Expect columns like f"{pid}_x", f"{pid}_y"
    x_col = f"{pid}_x"
    y_col = f"{pid}_y"

    if x_col not in df_wide.columns or y_col not in df_wide.columns:
        # Defensive: skip if for some reason this player doesn't have both coords
        continue

    p_df = df_wide.select(
        [
            "period_id",
            "timestamp",      # seconds in match time (SkillCorner / Kloppy)
            "frame_id",
            pl.lit(pid).alias("player_id"),
            pl.col(x_col).alias("x"),
            pl.col(y_col).alias("y"),
        ]
    ).filter(
        pl.col("x").is_not_null() & pl.col("y").is_not_null()
    )

    long_dfs.append(p_df)

df = pl.concat(long_dfs, how="vertical")
print(f"Reshaped to long format: {df.height} rows, {len(player_ids)} players")

# 5) Check if coordinates are normalized and scale to meters
print("\n--- Coordinate Diagnostics ---")
x_min, x_max = df['x'].min(), df['x'].max()
y_min, y_max = df['y'].min(), df['y'].max()
print(f"X range: {x_min:.2f} to {x_max:.2f}")
print(f"Y range: {y_min:.2f} to {y_max:.2f}")

# Standard pitch dimensions
PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters

# If coordinates are normalized (between 0 and 1), scale them to meters
if x_max <= 1.1 and y_max <= 1.1:  # Allow small margin for edge cases
    print("⚠ Coordinates are NORMALIZED - scaling to meters")
    print(f"Scaling: X by {PITCH_LENGTH}m, Y by {PITCH_WIDTH}m")
    
    df = df.with_columns([
        (pl.col("x") * PITCH_LENGTH).alias("x"),
        (pl.col("y") * PITCH_WIDTH).alias("y")
    ])
    
    print(f"After scaling - X range: {df['x'].min():.2f} to {df['x'].max():.2f} m")
    print(f"After scaling - Y range: {df['y'].min():.2f} to {df['y'].max():.2f} m")
else:
    print("✓ Coordinates appear to be in meters already")

print(f"\nSample coordinates (first player, first 10 frames):")
first_player = df['player_id'][0]
print(df.filter(pl.col('player_id') == first_player).head(10).select(['timestamp', 'x', 'y']))

# 6) Sort by player then timestamp so any time-based operations behave correctly
df = df.sort(["player_id", "timestamp"])

# 7) Compute time_gap: the time difference between consecutive frames for each player
print("\nComputing time_gap column")
df = df.with_columns([
    (pl.col("timestamp") - pl.col("timestamp").shift(1))
    .over("player_id")
    .alias("time_gap")
])

# Fill the first time_gap for each player with the expected frame interval
# For 10 Hz data, this should be 0.1 seconds
df = df.with_columns([
    pl.when(pl.col("time_gap").is_null())
    .then(0.1)
    .otherwise(pl.col("time_gap"))
    .alias("time_gap")
])

print(f"Time gap stats: mean={df['time_gap'].mean():.3f}s, median={df['time_gap'].median():.3f}s, min={df['time_gap'].min():.3f}s, max={df['time_gap'].max():.3f}s")

# 8) Add has_gap_before column (flag for detecting interruptions in tracking)
expected_interval = 0.1  # 10 Hz = 0.1 seconds
gap_threshold = expected_interval * 2  # 0.2 seconds

df = df.with_columns([
    (pl.col("time_gap") > gap_threshold).alias("has_gap_before")
])

print(f"Frames with gaps: {df['has_gap_before'].sum()}")

print("\nInitializing FatigueConfig from config.py")
cfg = config.FatigueConfig()

# 9) Populate config using documented metadata where possible
md = dataset.metadata

# Frame rate — documented as metadata.frame_rate for TrackingDataset
if cfg.frame_rate is None and hasattr(md, "frame_rate"):
    cfg.frame_rate = md.frame_rate
if cfg.frame_rate is None:
    cfg.frame_rate = 10.0  # SkillCorner open data is 10 fps per their GitHub README

# Pitch dimensions
if cfg.pitch_length is None:
    cfg.pitch_length = PITCH_LENGTH
if cfg.pitch_width is None:
    cfg.pitch_width = PITCH_WIDTH

print(
    f"Config resolved: pitch_length={cfg.pitch_length} m, "
    f"pitch_width={cfg.pitch_width} m, frame_rate={cfg.frame_rate} Hz"
)

print("\nComputing kinematic features on long-format player tracking")
df = compute_all_kinematic_features(df, cfg)
print(f"✓ Kinematic features computed: {df.shape[1]} columns")

# Debug: Check speed calculations
print("\n--- Speed Diagnostics ---")
print(f"Speed (v) range: {df['v'].min():.2f} to {df['v'].max():.2f} m/s")
print(f"Speed (v) mean: {df['v'].mean():.2f} m/s")
print(f"Speed (v) 95th percentile: {df['v'].quantile(0.95):.2f} m/s")
print(f"Speed (v) 99th percentile: {df['v'].quantile(0.99):.2f} m/s")
print(f"\nHSR threshold (config): {cfg.hsr_threshold} m/s")
print(f"Sprint threshold (config): {cfg.sprint_threshold} m/s")
print(f"\nFrames above HSR threshold: {(df['v'] >= cfg.hsr_threshold).sum()}")
print(f"Frames above sprint threshold: {(df['v'] >= cfg.sprint_threshold).sum()}")

print("\nDetecting HSR/sprint events")
df = detect_hsr_sprint_events(df, cfg)
print(f"✓ HSR/sprint events detected")

# 10) Summary statistics
print("\n" + "="*60)
print("FATIGUE ESTIMATION RESULTS")
print("="*60)

print(f"\nDataset: {df.height:,} tracking frames across {df['player_id'].n_unique()} players")
print(f"Match duration: {df['timestamp'].min():.1f}s to {df['timestamp'].max():.1f}s ({(df['timestamp'].max() - df['timestamp'].min())/60:.1f} minutes)")

print(f"\n--- High-Speed Running (HSR) ---")
print(f"HSR frames: {df['is_hsr_event'].sum():,} ({100*df['is_hsr_event'].sum()/df.height:.2f}%)")
if df['is_hsr_event'].sum() > 0:
    print(f"Total HSR distance: {df.filter(pl.col('is_hsr_event'))['distance_per_frame'].sum():.1f} m")
else:
    print(f"Total HSR distance: 0.0 m")

print(f"\n--- Sprinting ---")
print(f"Sprint frames: {df['is_sprint_event'].sum():,} ({100*df['is_sprint_event'].sum()/df.height:.2f}%)")
if df['is_sprint_event'].sum() > 0:
    print(f"Total sprint distance: {df.filter(pl.col('is_sprint_event'))['distance_per_frame'].sum():.1f} m")
else:
    print(f"Total sprint distance: 0.0 m")

print(f"\n--- Acceleration/Deceleration ---")
print(f"High acceleration frames: {df['is_high_accel'].sum():,}")
print(f"High deceleration frames: {df['is_high_decel'].sum():,}")

print(f"\n--- Recovery ---")
print(f"Frames in recovery: {df['is_recovered'].sum():,} ({100*df['is_recovered'].sum()/df.height:.2f}%)")
recovery_mean = df.filter(pl.col('recovery_time_sec') > 0)['recovery_time_sec'].mean()
if recovery_mean is not None:
    print(f"Average recovery time: {recovery_mean:.1f}s")
else:
    print(f"Average recovery time: N/A (no recovery events)")

# 11) Per-player summary
print("\n" + "="*60)
print("PER-PLAYER SUMMARY (Top 5 by total distance)")
print("="*60)

player_summary = df.group_by("player_id").agg([
    pl.col("distance_per_frame").sum().alias("total_distance"),
    pl.col("distance_per_frame").filter(pl.col("is_hsr_event")).sum().alias("hsr_distance"),
    pl.col("distance_per_frame").filter(pl.col("is_sprint_event")).sum().alias("sprint_distance"),
    pl.col("is_hsr_event").sum().alias("hsr_count"),
    pl.col("is_sprint_event").sum().alias("sprint_count"),
    pl.col("v").max().alias("max_speed"),
    pl.col("a").max().alias("max_accel"),
]).sort("total_distance", descending=True)

print(player_summary.head(5))

# 12) Sample detailed view for one player
print("\n" + "="*60)
print(f"DETAILED VIEW: Player {player_summary['player_id'][0]} (first 50 frames)")
print("="*60)

sample_player = player_summary['player_id'][0]
sample = df.filter(pl.col("player_id") == sample_player).head(50)

print(sample.select([
    "timestamp", 
    "x", 
    "y", 
    "v", 
    "a",
    "is_hsr_event", 
    "is_sprint_event",
    "hsr_count_60s",
    "sprint_count_60s",
    "is_recovered"
]))

print("\n✓ Test completed successfully!")