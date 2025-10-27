"""
Create visualizations comparing ATP Rankings vs Elo Ratings for December 8, 2020.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_comparison_data():
    """Load the comparison data."""
    df = pd.read_csv('models/saved_models/atp_vs_elo_comparison_2020.csv')
    return df

def create_visualizations(df):
    """Create comprehensive comparison visualizations."""
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('ATP Rankings vs Elo Ratings - December 8, 2020', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # ========================================
    # Plot 1: Top 20 Comparison
    # ========================================
    ax1 = axes[0, 0]
    top20 = df.head(20)
    
    x = np.arange(len(top20))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, top20['atp_rank'], width, 
                     label='ATP Rank', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.barh(x + width/2, top20['elo_rank'], width, 
                     label='Elo Rank', color='#4ECDC4', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(top20['player_name'], fontsize=9)
    ax1.set_xlabel('Rank Position (lower is better)', fontsize=11, fontweight='bold')
    ax1.set_title('Top 20 Players - ATP vs Elo Rankings', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.invert_xaxis()  # Lower rank is better
    ax1.invert_yaxis()  # Top players at top
    ax1.grid(axis='x', alpha=0.3)
    
    # ========================================
    # Plot 2: Scatter Plot - ATP Rank vs Elo Rank
    # ========================================
    ax2 = axes[0, 1]
    
    # Color by difference
    colors = ['green' if abs(d) < 10 else 'orange' if abs(d) < 30 else 'red' 
              for d in df['rank_diff']]
    
    ax2.scatter(df['atp_rank'], df['elo_rank'], c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect agreement)
    max_rank = max(df['atp_rank'].max(), df['elo_rank'].max())
    ax2.plot([0, max_rank], [0, max_rank], 'k--', linewidth=2, label='Perfect Agreement', alpha=0.5)
    
    # Annotate some interesting players
    outliers = df[abs(df['rank_diff']) > 100].head(3)
    for _, player in outliers.iterrows():
        ax2.annotate(player['player_name'], 
                    (player['atp_rank'], player['elo_rank']),
                    fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('ATP Rank', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Elo Rank', fontsize=11, fontweight='bold')
    ax2.set_title('ATP vs Elo Rank Correlation', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add correlation text
    correlation = df['atp_rank'].corr(df['elo_rank'])
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax2.transAxes, fontsize=11, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========================================
    # Plot 3: Rank Difference Distribution
    # ========================================
    ax3 = axes[1, 0]
    
    # Histogram of rank differences
    rank_diffs = df['rank_diff']
    ax3.hist(rank_diffs, bins=30, color='#95E1D3', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference')
    ax3.axvline(x=rank_diffs.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {rank_diffs.mean():.1f}')
    ax3.axvline(x=rank_diffs.median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {rank_diffs.median():.1f}')
    
    ax3.set_xlabel('Rank Difference (Elo - ATP)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Players', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Rank Differences', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add statistics box
    stats_text = f'Mean: {rank_diffs.mean():.1f}\nMedian: {rank_diffs.median():.1f}\nStd: {rank_diffs.std():.1f}'
    ax3.text(0.75, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ========================================
    # Plot 4: Top 10 + Biggest Differences
    # ========================================
    ax4 = axes[1, 1]
    
    # Get top 10 and biggest differences
    top10 = df.head(10).copy()
    top10['category'] = 'Top 10'
    
    # Biggest positive differences (ATP ranks higher)
    overrated = df.nlargest(5, 'rank_diff').copy()
    overrated['category'] = 'ATP >> Elo'
    
    # Biggest negative differences (Elo ranks higher)
    underrated = df.nsmallest(5, 'rank_diff').copy()
    underrated['category'] = 'Elo >> ATP'
    
    # Combine
    interesting = pd.concat([top10, overrated, underrated])
    interesting = interesting.sort_values('rank_diff')
    
    # Plot
    colors_map = {'Top 10': '#FFD93D', 'ATP >> Elo': '#FF6B6B', 'Elo >> ATP': '#6BCB77'}
    colors = [colors_map[cat] for cat in interesting['category']]
    
    bars = ax4.barh(range(len(interesting)), interesting['rank_diff'], color=colors, alpha=0.8)
    ax4.set_yticks(range(len(interesting)))
    ax4.set_yticklabels(interesting['player_name'], fontsize=8)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Rank Difference (Elo Rank - ATP Rank)', fontsize=11, fontweight='bold')
    ax4.set_title('Notable Players: Top 10 + Biggest Disagreements', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[cat], alpha=0.8, label=cat) 
                      for cat in colors_map.keys()]
    ax4.legend(handles=legend_elements, fontsize=9, loc='lower right')
    
    # Add annotations for interesting cases
    ax4.text(0.02, 0.98, 'Negative = Elo ranks higher\nPositive = ATP ranks higher',
            transform=ax4.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'models/saved_models/atp_vs_elo_comparison_2020.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    return output_path

def create_summary_table(df):
    """Create a summary table image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare top 20 data
    top20 = df.head(20).copy()
    top20['Rank Diff'] = top20['rank_diff'].apply(lambda x: f"+{int(x)}" if x > 0 else str(int(x)))
    
    # Create table data
    table_data = []
    table_data.append(['ATP', 'Player', 'ATP Pts', 'Elo Rank', 'Elo Rating', 'Diff'])
    
    for _, row in top20.iterrows():
        table_data.append([
            int(row['atp_rank']),
            row['player_name'],
            int(row['atp_points']),
            int(row['elo_rank']),
            f"{row['elo_rating']:.1f}",
            row['Rank Diff']
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.08, 0.35, 0.15, 0.12, 0.15, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code differences
    for i in range(1, len(table_data)):
        diff_val = top20.iloc[i-1]['rank_diff']
        
        # Color the difference column
        cell = table[(i, 5)]
        if abs(diff_val) < 5:
            cell.set_facecolor('#C8E6C9')  # Light green
        elif abs(diff_val) < 15:
            cell.set_facecolor('#FFE082')  # Light yellow
        else:
            cell.set_facecolor('#FFAB91')  # Light red
        
        # Alternate row colors
        if i % 2 == 0:
            for j in range(6):
                if j != 5:  # Don't override diff column
                    table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title('ATP vs Elo Rankings - Top 20 Players (December 8, 2020)', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Save table
    table_path = 'models/saved_models/atp_vs_elo_table_2020.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"✓ Table saved to: {table_path}")
    
    plt.close()
    return table_path

def main():
    """Main visualization pipeline."""
    print("="*80)
    print("📊 CREATING ATP vs ELO VISUALIZATIONS")
    print("="*80)
    
    # Load data
    df = load_comparison_data()
    print(f"\nLoaded comparison data for {len(df)} players")
    
    # Create main visualization
    viz_path = create_visualizations(df)
    
    # Create summary table
    table_path = create_summary_table(df)
    
    # Show where graphs are stored
    print("\n" + "="*80)
    print("📁 GRAPH STORAGE LOCATIONS")
    print("="*80)
    print("\nAll graphs and visualizations are stored in:")
    print("  📂 models/saved_models/")
    print("\nFiles created in this session:")
    print(f"  📊 {viz_path}")
    print(f"  📋 {table_path}")
    print("\nOther graphs in this directory:")
    print("  📊 calibration_plot.png - Probability calibration curves")
    print("  📊 elo_difference_analysis.png - Prediction accuracy by Elo difference")
    print("\nYou can view any of these files by opening them from Finder or using:")
    print("  open models/saved_models/<filename>.png")
    
    print("\n✓ VISUALIZATION COMPLETE!")

if __name__ == '__main__':
    main()
