"""
Kinematic feature computation module.
Computes speed, acceleration, distance, HSR, sprint events, and recovery metrics.
"""

import polars as pl
import logging
from config import FatigueConfig

logger = logging.getLogger(__name__)


def smooth_series(series: pl.Series, window: int) -> pl.Series:
    """Apply moving average smoothing to a series."""
    return series.rolling_mean(window_size=window, center=True).fill_null(strategy='forward').fill_null(strategy='backward')


def compute_speed_acceleration(df: pl.DataFrame, config: FatigueConfig) -> pl.DataFrame:
    """
    Compute instantaneous speed and acceleration with smoothing.
    
    Args:
        df: Tracking DataFrame with x, y coordinates
        config: FatigueConfig instance
        
    Returns:
        DataFrame with v (speed) and a (acceleration) columns
    """
    logger.info("Computing speed and acceleration...")
    
    dt = 1.0 / config.frame_rate
    
    # Compute displacement per frame
    df = df.with_columns([
        (pl.col('x') - pl.col('x').shift(1)).over('player_id').alias('dx'),
        (pl.col('y') - pl.col('y').shift(1)).over('player_id').alias('dy'),
        pl.col('time_gap').alias('dt')
    ])
    
    # Compute raw speed (m/s)
    # Use safe division to avoid inf values
    df = df.with_columns([
        pl.when(
            (pl.col('dt').is_null()) | 
            (pl.col('dt') <= 0) | 
            (pl.col('has_gap_before')) | 
            (pl.col('dt') > 2 * dt)
        )
        .then(0.0)
        .otherwise(
            (pl.col('dx').pow(2) + pl.col('dy').pow(2)).sqrt() / pl.col('dt')
        )
        .alias('v_raw')
    ])
    
    # Cap unrealistic speeds (> 15 m/s = 54 km/h)
    df = df.with_columns([
        pl.col('v_raw').clip(0.0, 15.0).alias('v_raw')
    ])
    
    # Apply smoothing
    df = df.with_columns([
        pl.col('v_raw').rolling_mean(
            window_size=config.speed_smoothing_window,
            center=True
        ).over('player_id').fill_null(strategy='forward').fill_null(strategy='backward').alias('v')
    ])
    
    # Compute acceleration (m/s²)
    df = df.with_columns([
        pl.when(
            (pl.col('dt').is_null()) | 
            (pl.col('dt') <= 0) | 
            (pl.col('has_gap_before')) | 
            (pl.col('dt') > 2 * dt)
        )
        .then(0.0)
        .otherwise(
            (pl.col('v') - pl.col('v').shift(1)).over('player_id') / pl.col('dt')
        )
        .alias('a_raw')
    ])
    
    # Cap unrealistic accelerations (±10 m/s²)
    df = df.with_columns([
        pl.col('a_raw').clip(-10.0, 10.0).alias('a_raw')
    ])
    
    # Apply smoothing to acceleration
    df = df.with_columns([
        pl.col('a_raw').rolling_mean(
            window_size=config.acceleration_smoothing_window,
            center=True
        ).over('player_id').fill_null(strategy='forward').fill_null(strategy='backward').alias('a')
    ])
    
    logger.info(f"✓ Speed range: {df['v'].min():.2f} - {df['v'].max():.2f} m/s")
    logger.info(f"✓ Acceleration range: {df['a'].min():.2f} - {df['a'].max():.2f} m/s²")
    
    return df


def compute_rolling_distance(df: pl.DataFrame, config: FatigueConfig) -> pl.DataFrame:
    """
    Compute rolling distance covered per minute.
    
    Args:
        df: DataFrame with speed column
        config: FatigueConfig instance
        
    Returns:
        DataFrame with rolling distance columns
    """
    logger.info("Computing rolling distance...")
    
    dt = 1.0 / config.frame_rate
    
    # Distance per frame (meters)
    df = df.with_columns([
        (pl.col('v') * dt).alias('distance_per_frame')
    ])
    
    # Rolling distance over specified window (e.g., 60 seconds)
    window_frames = int(config.rolling_distance_window * config.frame_rate)
    
    df = df.with_columns([
        pl.col('distance_per_frame')
        .rolling_sum(window_size=window_frames, min_periods=1)
        .over('player_id')
        .alias('rolling_distance_1min')
    ])
    
    logger.info("✓ Rolling distance computed")
    
    return df