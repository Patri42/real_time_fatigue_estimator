"""
Test script for kinematic_features.py
Tests speed, acceleration, and rolling distance computation.
"""

import polars as pl
import logging
from kinematic_features import compute_speed_acceleration, compute_rolling_distance
from config import FatigueConfig
from kloppy import skillcorner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def prepare_tracking_data(dataset, config: FatigueConfig) -> pl.DataFrame:
    """
    Convert Kloppy dataset to Polars DataFrame with required columns.
    
    Required columns for kinematic functions:
    - player_id: str
    - x, y: float (coordinates in meters)
    - timestamp: float (seconds)
    - time_gap: float (time since last frame)
    - has_gap_before: bool (indicates tracking gaps)
    """
    logger.info("Converting Kloppy dataset to Polars DataFrame...")
    
    # Get pitch dimensions for coordinate conversion
    pitch_length = config.pitch_length if config.pitch_length else 105.0
    pitch_width = config.pitch_width if config.pitch_width else 68.0
    
    records = []
    for frame in dataset.frames:
        timestamp = frame.timestamp
        
        # players_data is a dict where key=Player object, value=PlayerData
        for player, player_data in frame.players_data.items():
            if player_data and player_data.coordinates:
                # Kloppy coordinates are normalized (0-1), convert to meters
                x_meters = player_data.coordinates.x * pitch_length
                y_meters = player_data.coordinates.y * pitch_width
                
                records.append({
                    'player_id': str(player.player_id),
                    'timestamp': timestamp,
                    'x': x_meters,
                    'y': y_meters,
                })
    
    if not records:
        raise ValueError("No tracking data found in dataset!")
    
    df = pl.DataFrame(records)
    
    # Check coordinate ranges
    logger.info(f"X range: {df['x'].min():.2f} - {df['x'].max():.2f} m")
    logger.info(f"Y range: {df['y'].min():.2f} - {df['y'].max():.2f} m")
    logger.info(f"✓ Prepared {len(df)} tracking records for {df['player_id'].n_unique()} players")
    
    # Sort by player and time
    df = df.sort(['player_id', 'timestamp'])
    
    # Compute time gaps in SECONDS (convert from microseconds)
    df = df.with_columns([
        ((pl.col('timestamp') - pl.col('timestamp').shift(1))
         .over('player_id')
         .dt.total_microseconds() / 1_000_000.0)  # Convert microseconds to seconds
        .alias('time_gap')
    ])
    
    # Detect gaps (> 2x expected frame time)
    expected_dt = 1.0 / config.frame_rate
    df = df.with_columns([
        ((pl.col('time_gap').is_null()) | (pl.col('time_gap') > 2 * expected_dt))
        .alias('has_gap_before')
    ])
    
    logger.info(f"✓ Time gaps computed, unique players: {df['player_id'].n_unique()}")
    
    return df


def main():
    """Run kinematic feature tests."""
    
    # Load configuration
    config = FatigueConfig()
    logger.info(f"Loaded config: match_id={config.match_id}")
    
    # Load SkillCorner data
    logger.info("Loading SkillCorner data...")
    dataset = skillcorner.load_open_data(match_id=config.match_id)
    
    # Auto-detect pitch dimensions and frame rate from dataset
    if dataset.metadata.pitch_dimensions:
        # Kloppy pitch dimensions are in normalized units, use standard pitch size
        config.pitch_length = 105.0  # Standard pitch length
        config.pitch_width = 68.0    # Standard pitch width
        logger.info(f"✓ Pitch: {config.pitch_length}m × {config.pitch_width}m")
    
    if dataset.metadata.frame_rate:
        config.frame_rate = dataset.metadata.frame_rate
        logger.info(f"✓ Frame rate: {config.frame_rate} Hz")
    
    # Prepare tracking data
    df = prepare_tracking_data(dataset, config)
    
    # Test 1: Compute speed and acceleration
    logger.info("\n=== TEST 1: Speed & Acceleration ===")
    df = compute_speed_acceleration(df, config)
    
    print(f"\nSpeed statistics:")
    print(f"  Min: {df['v'].min():.2f} m/s")
    print(f"  Mean: {df['v'].mean():.2f} m/s")
    print(f"  Max: {df['v'].max():.2f} m/s")
    print(f"  P95: {df['v'].quantile(0.95):.2f} m/s")
    
    print(f"\nAcceleration statistics:")
    print(f"  Min: {df['a'].min():.2f} m/s²")
    print(f"  Mean: {df['a'].mean():.2f} m/s²")
    print(f"  Max: {df['a'].max():.2f} m/s²")
    
    # Test 2: Compute rolling distance
    logger.info("\n=== TEST 2: Rolling Distance ===")
    df = compute_rolling_distance(df, config)
    
    print(f"\nRolling distance (1 min window):")
    print(f"  Min: {df['rolling_distance_1min'].min():.1f} m")
    print(f"  Mean: {df['rolling_distance_1min'].mean():.1f} m")
    print(f"  Max: {df['rolling_distance_1min'].max():.1f} m")
    
    # Sample output
    print(f"\n=== Sample Data (first player, first 10 frames) ===")
    sample = df.filter(
        pl.col('player_id') == df['player_id'][0]
    ).head(10).select(['player_id', 'timestamp', 'x', 'y', 'v', 'a', 'rolling_distance_1min'])
    print(sample)
    
    # High-speed running check
    hsr_count = df.filter(pl.col('v') >= config.hsr_threshold).height
    sprint_count = df.filter(pl.col('v') >= config.sprint_threshold).height
    
    print(f"\n=== Speed Thresholds ===")
    print(f"HSR frames (≥{config.hsr_threshold} m/s): {hsr_count} ({100*hsr_count/len(df):.1f}%)")
    print(f"Sprint frames (≥{config.sprint_threshold} m/s): {sprint_count} ({100*sprint_count/len(df):.1f}%)")
    
    logger.info("\n✓ All tests completed successfully!")


if __name__ == '__main__':
    main()