# First version
import polars as pl
from kloppy import skillcorner

# Speed = displacement / time between frames

# Need smoothing to reduce GPS/tracking noise

# Hardcode pitch dimensions for now (105m x 68m)

def compute_speed(df: pl.DataFrame, frame_rate: float):
    dt = 1.0 / frame_rate
    df = df.with_columns([
        (pl.col('x') - pl.col('x').shift(1)).over('player_id').alias('dx'),
        (pl.col('y') - pl.col('y').shift(1)).over('player_id').alias('dy'),
    ])
    df = df.with_columns([
        (pl.col('dx') ** 2 + pl.col('dy') ** 2).sqrt().alias('speed')
    ])
    return df

# Load Skillcorner dataset
print("Loading dataset")
dataset = skillcorner.load_open_data(match_id='1925299')
print(f"Frames: {len(dataset.records)}")

# Convert tracking data to DataFrame
data = []
for frame in dataset.records:
    for player_id, player_data in frame.players_data.items():
        data.append({
            'frame_id': frame.frame_id,
            'player_id': str(player_id),
            'x': player_data.coordinates.x,
            'y': player_data.coordinates.y,
        })

df = pl.DataFrame(data)

# Test compute_speed
frame_rate = 10.0 
df_with_speed = compute_speed(df, frame_rate)
print(df_with_speed.head(20))

# Show data for a single player to see speed values
print ("\nFirst 20 rows for one player:")
print (df_with_speed.filter(pl.col('player_id') == "Walid Shour").head(20))

# Show some stats
print ("\nSpeed stats (m/s):")
print (df_with_speed.select('speed').describe())
