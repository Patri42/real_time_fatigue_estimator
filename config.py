"""
Configuration file for fatigue estimation system.

==============================================================================
WHAT YOU NEED TO CHANGE vs. WHAT'S AUTOMATIC
==============================================================================

✅ CHANGE THESE (Data-specific):
    - match_id: Your match ID from SkillCorner, Opta, StatsBomb, etc.

⚙️ AUTO-DETECTED (No need to change):
    - pitch_length: Automatically detected from data (typical: 100-110m)
    - pitch_width: Automatically detected from data (typical: 64-75m)
    - frame_rate: Automatically detected from data (typical: 10-25 Hz)

🔬 RESEARCH-BASED DEFAULTS (Work for any match):
    - hsr_threshold: 5.5 m/s (Carling et al. 2008, industry standard)
    - sprint_threshold: 7.0 m/s (Bradley et al. 2013, industry standard)
    - high_acceleration_threshold: 3.0 m/s² (Akenhead et al. 2016)
    - baseline_end_minutes: 10.0 (Mohr et al. 2003)
    - weight_*: Component weights (Martín-García et al. 2018)

🔧 OPTIONAL TUNING (Advanced users):
    - Smoothing windows (for noisy tracking data)
    - Rolling windows (for different temporal scales)
    - Percentiles (for different sensitivity)

==============================================================================
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class FatigueConfig:
    """
    Configuration for fatigue estimation system.
    
    Most parameters are research-based defaults that work for any match.
    Pitch dimensions and frame rate are AUTO-DETECTED from data.
    """
    
    # === Data Loading ===
    match_id: str = '1925299'  # SkillCorner match ID (change to your match)
    coordinate_system: str = 'metrica'  # Kloppy coordinate system
    only_alive: bool = True  # Only include ball-in-play frames
    
    # === Pitch Dimensions (AUTO-DETECTED) ===
    # These are set automatically from Kloppy metadata.
    # Only override if using custom data without metadata.
    pitch_length: Optional[float] = None  # meters (auto-detected, typical: 100-110m)
    pitch_width: Optional[float] = None   # meters (auto-detected, typical: 64-75m)
    
    # === Temporal Parameters (AUTO-DETECTED) ===
    frame_rate: Optional[float] = None  # Hz (auto-detected from data, typical: 10-25 Hz)
    
    # === Smoothing Parameters ===
    speed_smoothing_window: int = 7  # frames (0.7s at 10Hz)
    acceleration_smoothing_window: int = 5  # frames (0.5s at 10Hz)
    
    # === Speed Thresholds (UNIVERSAL - Same for all matches) ===
    # These are research-based and used by 80-95% of elite clubs worldwide.
    # Based on metabolic cost and glycogen depletion patterns.
    # 
    # Sources: Carling et al. (2008) Sports Medicine, Bradley et al. (2013) JSS
    # Only change if analyzing youth/women's soccer or custom requirements.
    hsr_threshold: float = 5.5  # m/s (19.8 km/h) - High-Speed Running
    sprint_threshold: float = 7.0  # m/s (25.2 km/h) - Sprint threshold
    
    # === Acceleration Thresholds (UNIVERSAL) ===
    # Based on neuromuscular load and injury risk research.
    # Source: Akenhead & Nassis (2016) IJSPP
    high_acceleration_threshold: float = 3.0  # m/s² - High acceleration
    high_deceleration_threshold: float = -3.0  # m/s² - High deceleration
    
    # === Minimum Duration for Events (seconds) ===
    hsr_min_duration: float = 1.0  # seconds
    sprint_min_duration: float = 1.0  # seconds
    
    # === Rolling Windows (UNIVERSAL - Recommended by research) ===
    # These define how we aggregate data over time.
    rolling_distance_window: float = 60.0  # seconds - For distance per minute
    fatigue_window_lookback: float = 180.0  # seconds - Current performance window (3 min)
    fatigue_window_step: float = 1.0  # seconds - Update frequency (real-time: 1s)
    
    # === Baseline Window (UNIVERSAL - Industry standard) ===
    # Baseline = "fresh" performance to compare against.
    # First 5-10 minutes of each half is standard (Mohr et al. 2003).
    # Most fatigue occurs after this window.
    baseline_start_minutes: float = 0.0  # Start of baseline (from period start)
    baseline_end_minutes: float = 10.0  # End of baseline (10 min = industry standard)
    baseline_per_half: bool = True  # True = fresh baseline per half (accounts for half-time recovery)
    
    # === Recovery Time ===
    recovery_threshold_percentile: float = 50  # % of baseline median speed
    
    # === Fatigue Score Components (RESEARCH-BASED WEIGHTS) ===
    # These weights combine 4 fatigue indicators into one score.
    # 
    # MUST sum to 1.0 (validation enforced in __post_init__).
    # 
    # Rationale:
    # - Speed/Burst: Core performance metrics (50% combined)
    # - Recent Load: Strongest predictor - cumulative effect (30%)
    # - Recovery: Leading indicator but newer research (20%)
    weight_speed_drop: float = 0.25         # Weight for speed decline
    weight_burst_drop: float = 0.25         # Weight for HSR/sprint frequency decline
    weight_recovery_increase: float = 0.20  # Weight for recovery time increase
    weight_recent_load: float = 0.30        # Weight for cumulative load (highest)
    
    # === Fatigue Score Percentiles ===
    speed_percentile: int = 90  # Use P90 speed for comparison
    
    # === Load Accumulation Window ===
    load_accumulation_window: float = 600.0  # seconds (10 minutes)
    
    # === Confidence Parameters ===
    min_frames_for_confidence: int = 10  # minimum frames in window
    max_gap_tolerance: float = 5.0  # seconds (max gap in tracking)
    
    # === Output ===
    output_dir: str = '/home/ubuntu/fatigue_output'
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hsr_threshold < self.sprint_threshold, "HSR threshold must be < Sprint threshold"
        assert self.baseline_end_minutes > self.baseline_start_minutes, "Invalid baseline window"
        assert 0 <= self.weight_speed_drop <= 1, "Weights must be in [0, 1]"
        total_weight = (self.weight_speed_drop + self.weight_burst_drop + 
                       self.weight_recovery_increase + self.weight_recent_load)
        assert 0.99 <= total_weight <= 1.01, f"Weights must sum to 1.0, got {total_weight}"


# Default configuration instance
DEFAULT_CONFIG = FatigueConfig()