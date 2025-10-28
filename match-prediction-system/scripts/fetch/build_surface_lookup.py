#!/usr/bin/env python3
"""
Extract tournament -> surface mapping from existing match_info.csv data.
This gives us a lookup table for tournaments we already know about.
"""

import pandas as pd
import json

# Load existing match data
print("="*80)
print("BUILDING TOURNAMENT -> SURFACE MAPPING FROM EXISTING DATA")
print("="*80)

match_info_path = 'data/processed/match_info.csv'

try:
    df = pd.read_csv(match_info_path)
    print(f"\n✅ Loaded {len(df)} existing matches")
    
    # Check what columns we have
    print(f"Columns: {list(df.columns)}")
    
    # If we have surface and tournament/description, we can build mapping
    if 'surface' in df.columns and 'description' in df.columns:
        # Group by tournament description and get the surface
        # Assuming tournament name is consistent for same venue
        tournament_surface = df.groupby('description')['surface'].first().to_dict()
        
        print(f"\n✅ Extracted {len(tournament_surface)} tournament -> surface mappings")
        
        # Show some examples
        print("\nSample mappings:")
        for i, (tournament, surface) in enumerate(list(tournament_surface.items())[:10]):
            print(f"  {tournament}: {surface}")
        
        # Save to JSON file
        output_file = 'data/processed/tournament_surface_mapping.json'
        with open(output_file, 'w') as f:
            json.dump(tournament_surface, f, indent=2)
        
        print(f"\n✅ Saved mapping to {output_file}")
        
        # Also create a mapping by common tournament names (cleaned)
        # For example: "Australian Open" appears in many years
        tournament_base_names = {}
        for desc, surface in tournament_surface.items():
            # Extract base tournament name (before year/round details)
            # This is a simple heuristic - tournament name is usually first part
            parts = desc.split(',')[0].split('-')[0].strip()
            if parts:
                tournament_base_names[parts] = surface
        
        print(f"\n✅ Created {len(tournament_base_names)} base tournament name mappings")
        print("\nSample base name mappings:")
        for i, (name, surface) in enumerate(list(tournament_base_names.items())[:10]):
            print(f"  {name}: {surface}")
        
        # Save base names mapping
        base_output_file = 'data/processed/tournament_base_surface_mapping.json'
        with open(base_output_file, 'w') as f:
            json.dump(tournament_base_names, f, indent=2)
        
        print(f"\n✅ Saved base mapping to {base_output_file}")
        
    else:
        print("\n⚠️  Surface column not found in match_info.csv")
        print(f"Available columns: {list(df.columns)}")
        
except FileNotFoundError:
    print(f"\n❌ File not found: {match_info_path}")
    print("Make sure you're running this from the match-prediction-system directory")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")

print("\n" + "="*80)
print("DONE")
print("="*80)
