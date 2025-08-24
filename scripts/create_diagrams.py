#!/usr/bin/env python3
"""
Create architecture diagrams for transformer vs baseline models
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_architecture_diagram():
    """Create visual comparison of model architectures"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Architecture Comparison: Transformers vs Baselines', fontsize=16, fontweight='bold')
    
    # Colors
    colors = {
        'input': '#e8f4fd',
        'embed': '#b3d9ff',
        'attention': '#66c2ff',
        'mlp': '#1a8cff',
        'output': '#0066cc',
        'baseline': '#ff9999',
        'good': '#90EE90',
        'bad': '#FFB6C1'
    }
    
    # 1. Full Transformer
    ax1.set_title('Full Transformer\n(331K params, F1=0.060)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 15)
    
    # Input
    ax1.add_patch(patches.Rectangle((1, 13), 8, 1, facecolor=colors['input'], edgecolor='black'))
    ax1.text(5, 13.5, 'Gene Expression (2700×500)', ha='center', va='center', fontsize=10)
    
    # Gene Embedding
    ax1.add_patch(patches.Rectangle((1, 11.5), 8, 1, facecolor=colors['embed'], edgecolor='black'))
    ax1.text(5, 12, 'Gene Embedding (64K params)', ha='center', va='center', fontsize=10)
    
    # Transformer Blocks
    for i, block_y in enumerate([9, 6.5]):
        ax1.add_patch(patches.Rectangle((1, block_y), 8, 2, facecolor=colors['attention'], edgecolor='black'))
        ax1.text(5, block_y+1.5, f'TransformerBlock {i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(5, block_y+1, '• LocalGraphAttention (65K params)', ha='center', va='center', fontsize=8)
        ax1.text(5, block_y+0.5, '• MLP (65K params)', ha='center', va='center', fontsize=8)
    
    # Output
    ax1.add_patch(patches.Rectangle((1, 4.5), 8, 1, facecolor=colors['output'], edgecolor='black'))
    ax1.text(5, 5, 'Classification Head → 9 classes', ha='center', va='center', fontsize=10)
    
    # Result
    ax1.add_patch(patches.Rectangle((1, 2.5), 8, 1.5, facecolor=colors['bad'], edgecolor='red', linewidth=2))
    ax1.text(5, 3.5, '❌ OVERFITS', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(5, 3, 'Needs 5M+ samples', ha='center', va='center', fontsize=10)
    ax1.text(5, 2.7, 'Has only 2.7K samples', ha='center', va='center', fontsize=10)
    
    # Arrows
    for y in [12.5, 10.5, 8, 5.5, 4]:
        ax1.arrow(5, y, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    
    # 2. Tiny Transformer
    ax2.set_title('Tiny Transformer\n(4.6K params, F1=0.060)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    
    # Input
    ax2.add_patch(patches.Rectangle((1, 10.5), 8, 1, facecolor=colors['input'], edgecolor='black'))
    ax2.text(5, 11, 'Gene Expression (2700×500)', ha='center', va='center', fontsize=10)
    
    # Gene Embedding
    ax2.add_patch(patches.Rectangle((1, 9), 8, 1, facecolor=colors['embed'], edgecolor='black'))
    ax2.text(5, 9.5, 'Gene Embedding (4K params)', ha='center', va='center', fontsize=10)
    
    # Single Transformer Block
    ax2.add_patch(patches.Rectangle((1, 6.5), 8, 2, facecolor=colors['attention'], edgecolor='black'))
    ax2.text(5, 7.8, 'TransformerBlock (tiny)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.text(5, 7.3, '• LocalGraphAttention (300 params)', ha='center', va='center', fontsize=8)
    ax2.text(5, 6.8, '• MLP (130 params)', ha='center', va='center', fontsize=8)
    
    # Output
    ax2.add_patch(patches.Rectangle((1, 5), 8, 1, facecolor=colors['output'], edgecolor='black'))
    ax2.text(5, 5.5, 'Classification Head → 9 classes', ha='center', va='center', fontsize=10)
    
    # Result
    ax2.add_patch(patches.Rectangle((1, 3), 8, 1.5, facecolor=colors['bad'], edgecolor='red', linewidth=2))
    ax2.text(5, 4, '❌ STILL OVERFITS', ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(5, 3.5, 'Needs 69K+ samples', ha='center', va='center', fontsize=10)
    ax2.text(5, 3.2, 'Has only 2.7K samples', ha='center', va='center', fontsize=10)
    
    # Arrows
    for y in [10, 8.5, 6, 4.5]:
        ax2.arrow(5, y, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    
    # 3. Logistic Regression
    ax3.set_title('Logistic Regression\n(4.5K params, F1=0.290)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Input
    ax3.add_patch(patches.Rectangle((1, 8.5), 8, 1, facecolor=colors['input'], edgecolor='black'))
    ax3.text(5, 9, 'Gene Expression (2700×500)', ha='center', va='center', fontsize=10)
    
    # Preprocessing
    ax3.add_patch(patches.Rectangle((1, 7), 8, 1, facecolor=colors['baseline'], edgecolor='black'))
    ax3.text(5, 7.5, 'StandardScaler (mean=0, std=1)', ha='center', va='center', fontsize=10)
    
    # Linear Layer
    ax3.add_patch(patches.Rectangle((1, 5.5), 8, 1, facecolor=colors['baseline'], edgecolor='black'))
    ax3.text(5, 6, 'Linear: 500 genes → 9 classes (4.5K params)', ha='center', va='center', fontsize=10)
    
    # Output
    ax3.add_patch(patches.Rectangle((1, 4), 8, 1, facecolor=colors['output'], edgecolor='black'))
    ax3.text(5, 4.5, 'Softmax → Probabilities', ha='center', va='center', fontsize=10)
    
    # Result
    ax3.add_patch(patches.Rectangle((1, 2), 8, 1.5, facecolor=colors['good'], edgecolor='green', linewidth=2))
    ax3.text(5, 3, '✅ WORKS WELL!', ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.text(5, 2.5, '5x Better F1 Score', ha='center', va='center', fontsize=10)
    ax3.text(5, 2.2, 'Handles class imbalance', ha='center', va='center', fontsize=10)
    
    # Arrows
    for y in [8, 6.5, 5, 3.5]:
        ax3.arrow(5, y, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('off')
    
    # 4. Performance Comparison
    ax4.set_title('Performance vs Model Complexity', fontsize=12, fontweight='bold')
    
    models = ['Logistic\nRegression', 'Tiny\nTransformer', 'Full\nTransformer']
    params = [4.5, 4.6, 331]
    f1_scores = [0.290, 0.060, 0.060]
    colors_bar = ['green', 'orange', 'red']
    
    # Bar plot for F1 scores
    bars = ax4.bar(models, f1_scores, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('F1 Score', fontsize=11)
    ax4.set_ylim(0, 0.35)
    
    # Add parameter counts as text
    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'F1: {height:.3f}\n{param}K params',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line for random baseline
    ax4.axhline(y=0.111, color='gray', linestyle='--', alpha=0.7)
    ax4.text(2.5, 0.12, 'Random Baseline (0.111)', ha='right', va='bottom', fontsize=9, style='italic')
    
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Model Type', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('docs/architecture_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Architecture diagram saved to docs/architecture_comparison.png")

def create_data_requirement_plot():
    """Create plot showing data requirements vs model complexity"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Model data
    models = ['Linear\nModels', 'Small\nTransformer', 'Medium\nTransformer', 'Large\nTransformer']
    params = [5, 50, 500, 5000]  # in thousands
    data_needed = [p * 15 for p in params]  # 15x rule
    current_data = 2.7  # Current dataset size in thousands
    
    # Create scatter plot
    colors = ['green', 'orange', 'red', 'darkred']
    sizes = [100, 150, 200, 250]
    
    for i, (model, param, data, color, size) in enumerate(zip(models, params, data_needed, colors, sizes)):
        ax.scatter(param, data, c=color, s=size, alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(model, (param, data), xytext=(10, 10), 
                   textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Add current data line
    ax.axhline(y=current_data, color='blue', linestyle='-', linewidth=3, alpha=0.8)
    ax.text(1000, current_data + 500, 'Current Dataset (2.7K samples)', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
    
    # Add feasible region
    ax.fill_between([0, 10000], [0, 0], [current_data, current_data], 
                   alpha=0.2, color='green', label='Feasible Region')
    
    ax.set_xlabel('Model Parameters (thousands)', fontsize=12)
    ax.set_ylabel('Training Samples Needed (thousands)', fontsize=12)
    ax.set_title('Data Requirements vs Model Complexity\n(15x Parameter Rule)', fontsize=14, fontweight='bold')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations
    ax.annotate('✅ Feasible with current data', xy=(5, 50), xytext=(50, 20),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold')
    
    ax.annotate('❌ Need 100x more data', xy=(500, 7500), xytext=(200, 20000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/data_requirements.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Data requirements plot saved to docs/data_requirements.png")

if __name__ == "__main__":
    import os
    os.makedirs('docs', exist_ok=True)
    
    create_architecture_diagram()
    create_data_requirement_plot()
    
    print("\n📊 Diagrams created successfully!")
    print("View them in the docs/ folder:")
    print("  - docs/architecture_comparison.png")
    print("  - docs/data_requirements.png")
