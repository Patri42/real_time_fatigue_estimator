from kloppy import skillcorner
import polars as pl

# Load Skillcorner Opendata
print ("Loading dataset")
dataset = skillcorner.load_open_data(match_id='1925299')
print (f"Frames: {len(dataset.records)}")

# Convert to Dataframe and explore
rows = []
for frame in dataset.records[:1000]: # Sample first 1000
    for player, data in frame.players_data.items():
        rows.append({
            'timestamp': frame.timestamp.total_seconds(),
            'player_id': player.player_id,
            'x': data.coordinates.x,
            'y': data.coordinates.y
        })
df = pl.DataFrame(rows)
print (df.head())