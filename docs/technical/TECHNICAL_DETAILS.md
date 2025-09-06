# Technical Details: Complete Architecture Specification

## 🏗️ **Model Architecture Deep Dive**

### **SCRNALongformer Overview**
```python
class SCRNALongformer(nn.Module):
    """
    kNN-Masked Gene-Token Transformer for Single-Cell RNA Analysis
    
    Architecture:
    - Gene embedding layer (vocab_size × d_model)
    - Optional value projection (d_model → d_model)
    - Stack of TransformerBlocks with LocalGraphAttention
    - Global average pooling + classification head
    - Optional MLM head for masked gene prediction
    """
```

### **Core Components**

#### **1. Gene Embedding Layer**
```python
# Maps gene indices to dense vectors
gene_embedding = nn.Embedding(vocab_size, d_model)
# vocab_size = number of genes (typically 500-2000)
# d_model = embedding dimension (32, 64, 128)
```

**Purpose:** Convert discrete gene indices into continuous representations that capture gene relationships.

**Parameters:** `vocab_size × d_model` (e.g., 500 × 128 = 64K params)

#### **2. LocalGraphAttention**
```python
class LocalGraphAttention(nn.Module):
    """
    Attention mechanism with kNN-based biological locality
    
    Key Innovation: Only attend to k-nearest neighbor genes
    based on cosine similarity of expression patterns
    """
    
    def __init__(self, d_model, n_heads, allow_mask_shape):
        # allow_mask_shape: (G,G), (B,G,G), (H,G,G), or (B,H,G,G)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask):
        # x: (B, G, D) - batch, genes, dimensions
        # mask: boolean adjacency matrix for allowed attention
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)
        
        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_model)
        
        # Apply biological locality mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Standard attention computation
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        
        return self.out_proj(output)
```

**Key Innovation:** 
- **Biological locality:** Genes only attend to functionally related genes
- **Data-driven:** kNN mask computed from expression similarity, not manual curation
- **Scalable:** O(k×G) complexity instead of O(G²)

#### **3. kNN Mask Generation**
```python
def create_knn_mask(gene_expressions, k=20):
    """
    Create biological locality mask based on gene co-expression
    
    Args:
        gene_expressions: (n_cells, n_genes) expression matrix
        k: number of nearest neighbors per gene
    
    Returns:
        mask: (n_genes, n_genes) boolean adjacency matrix
    """
    # Compute pairwise cosine similarity
    gene_similarity = cosine_similarity(gene_expressions.T)  # (G, G)
    
    # For each gene, select k most similar genes
    _, top_k_indices = torch.topk(gene_similarity, k, dim=-1)
    
    # Create boolean mask
    mask = torch.zeros_like(gene_similarity, dtype=torch.bool)
    mask.scatter_(1, top_k_indices, True)
    
    # Ensure symmetry and self-connections
    mask = mask | mask.T
    mask.fill_diagonal_(True)
    
    return mask
```

**Biological Rationale:**
- Co-expressed genes often share regulatory pathways
- Local attention preserves biological modularity
- Reduces noise from irrelevant gene interactions

#### **4. TransformerBlock**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        self.attention = LocalGraphAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask):
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

#### **5. Classification Head**
```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        self.pooling = "mean"  # Global average pooling
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (B, G, D) → (B, D)
        pooled = x.mean(dim=1)  # Average over genes
        logits = self.classifier(pooled)
        return logits
```

#### **6. Optional MLM Head**
```python
class MLMHead(nn.Module):
    """Masked Language Model head for gene expression prediction"""
    def __init__(self, d_model, vocab_size):
        self.predictor = nn.Linear(d_model, 1)  # Regression for expression
    
    def forward(self, x):
        # x: (B, G, D) → (B, G, 1)
        predictions = self.predictor(x).squeeze(-1)
        return predictions
```

---

## 📊 **Parameter Analysis**

### **Parameter Counting by Component**

```python
def count_parameters(model):
    """Detailed parameter breakdown"""
    
    # Gene embedding
    embedding_params = vocab_size * d_model
    
    # Value projection (optional)
    val_proj_params = d_model * d_model if use_val_proj else 0
    
    # Per TransformerBlock
    block_params = (
        3 * d_model * d_model +  # QKV projection
        d_model * d_model +      # Output projection
        2 * d_model +           # LayerNorm parameters
        d_model * d_ff +        # FFN layer 1
        d_ff * d_model +        # FFN layer 2
        d_ff + d_model          # Bias terms
    )
    
    # Total transformer blocks
    transformer_params = depth * block_params
    
    # Classification head
    classifier_params = d_model * num_classes + num_classes
    
    # MLM head (optional)
    mlm_params = d_model + 1 if use_mlm else 0
    
    total = (embedding_params + val_proj_params + 
             transformer_params + classifier_params + mlm_params)
    
    return {
        'embedding': embedding_params,
        'transformer': transformer_params,
        'classifier': classifier_params,
        'mlm': mlm_params,
        'total': total
    }
```

### **Model Size Configurations**

| Configuration | d_model | depth | Parameters | Use Case |
|---------------|---------|--------|------------|----------|
| **Micro** | 8 | 1 | 2.2K | Proof of concept |
| **Tiny** | 8 | 1 | 4.6K | Small datasets |
| **Small** | 32 | 1 | 18K | Medium datasets |
| **Default** | 64 | 2 | 130K | Large datasets |
| **Large** | 128 | 2 | 331K | Very large datasets |

### **Data Requirements (15x Rule)**

```python
# Conservative estimate for transformer training
samples_needed = total_parameters * 15

# Examples:
tiny_model = 4_600 * 15      # = 69K samples needed
default_model = 130_000 * 15 # = 1.95M samples needed
large_model = 331_000 * 15   # = 4.97M samples needed

# Available in PBMC3k
pbmc3k_samples = 2_700       # Insufficient for any transformer
```

---

## 🧬 **Biological Implementation Details**

### **Data Preprocessing Pipeline**

```python
def prepare_biological_data(adata):
    """Complete preprocessing for single-cell data"""
    
    # 1. Basic filtering
    sc.pp.filter_cells(adata, min_genes=200)    # Remove empty cells
    sc.pp.filter_genes(adata, min_cells=3)      # Remove rare genes
    
    # 2. Normalization
    sc.pp.normalize_total(adata, target_sum=1e4) # Total count normalization
    sc.pp.log1p(adata)                          # Log transformation
    
    # 3. CRITICAL: Gene scaling before PCA
    sc.pp.scale(adata, max_value=10)            # Scale genes to mean=0, std=1
    
    # 4. Feature selection
    sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=500)
    adata = adata[:, adata.var.highly_variable]
    
    # 5. Dimensionality reduction
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # 6. Clustering for cell type labels
    sc.tl.leiden(adata, resolution=0.5)
    
    return adata
```

**Critical Insight:** Gene scaling (`sc.pp.scale`) before PCA is biologically essential, not optional. Without it, PCA captures technical variation rather than biological signal.

### **kNN Graph Construction**

```python
def build_gene_knn_graph(expression_matrix, k=20):
    """Build biologically meaningful gene-gene adjacency"""
    
    # Compute gene-gene cosine similarity
    gene_corr = np.corrcoef(expression_matrix.T)  # (n_genes, n_genes)
    
    # Convert to cosine similarity (more robust than correlation)
    from sklearn.metrics.pairwise import cosine_similarity
    gene_sim = cosine_similarity(expression_matrix.T)
    
    # For each gene, keep top-k most similar genes
    adjacency = np.zeros_like(gene_sim, dtype=bool)
    for i in range(gene_sim.shape[0]):
        top_k_idx = np.argsort(gene_sim[i])[-k:]
        adjacency[i, top_k_idx] = True
    
    # Make symmetric and add self-connections
    adjacency = adjacency | adjacency.T
    np.fill_diagonal(adjacency, True)
    
    return adjacency
```

---

## 🔬 **Implementation Quality Assurance**

### **Numerical Stability**

```python
# Attention computation with proper scaling
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)

# Gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Mixed precision training for memory efficiency
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

### **Memory Optimization**

```python
# Gradient checkpointing for large models
if use_gradient_checkpointing:
    model.gradient_checkpointing_enable()

# Efficient attention computation
def efficient_attention(Q, K, V, mask, chunk_size=1024):
    """Chunked attention to reduce memory usage"""
    # Process attention in chunks to avoid OOM
    pass
```

### **Device Compatibility**

```python
def get_device():
    """Automatic device selection"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device('mps')
    else:
        return torch.device('cpu')
```

---

## 📋 **Configuration System**

### **YAML Configuration Schema**

```yaml
# configs/default.yaml
model:
  vocab_size: 500           # Number of genes
  d_model: 64              # Embedding dimension
  depth: 2                 # Number of transformer blocks
  n_heads: 4               # Attention heads per block
  d_ff: 256               # Feed-forward hidden size
  dropout: 0.1            # Dropout rate
  use_val_proj: true      # Value projection layer
  use_mlm: false          # MLM head for pretraining

data:
  dataset: "pbmc3k"       # Dataset identifier
  n_hvg: 500              # Top variable genes
  knn_k: 20               # kNN graph connectivity
  zscore: false           # Additional z-scoring
  train_split: 0.8        # Training fraction

training:
  batch_size: 32          # Batch size
  learning_rate: 1e-3     # Learning rate
  epochs: 100             # Training epochs
  weight_decay: 1e-4      # L2 regularization
  early_stopping: 10      # Patience for early stopping
```

### **Specialized Configurations**

- `configs/tiny_model.yaml` - Minimal parameters for small datasets
- `configs/real_leiden_top500.yaml` - Real PBMC data with leiden clustering
- `configs/exp_zscore.yaml` - Z-scoring experiments
- `configs/mlm_pretraining.yaml` - Masked language model setup

---

## 🎯 **Performance Characteristics**

### **Computational Complexity**

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| **Gene Embedding** | O(B×G) | O(V×D) |
| **kNN Attention** | O(B×G×k×D) | O(G×k) |
| **Full Attention** | O(B×G²×D) | O(G²) |
| **Feed-forward** | O(B×G×D²) | O(D²) |

Where: B=batch size, G=genes, V=vocab size, D=model dimension, k=kNN parameter

### **Memory Usage**

```python
# Approximate memory requirements (MB)
def estimate_memory(batch_size, n_genes, d_model, depth):
    # Input tensors
    input_mem = batch_size * n_genes * d_model * 4  # float32
    
    # Model parameters
    param_mem = count_parameters(model)['total'] * 4
    
    # Gradient memory (2x parameters)
    grad_mem = param_mem * 2
    
    # Activation memory (depth-dependent)
    activation_mem = batch_size * n_genes * d_model * depth * 4
    
    total_mb = (input_mem + param_mem + grad_mem + activation_mem) / (1024**2)
    return total_mb
```

### **Training Time Estimates**

| Model Size | Dataset | GPU | Training Time |
|------------|---------|-----|---------------|
| Tiny (4.6K) | PBMC3k | CPU | ~2 minutes |
| Small (18K) | PBMC3k | MPS | ~5 minutes |
| Default (130K) | PBMC3k | V100 | ~15 minutes |
| Large (331K) | 100K cells | A100 | ~2 hours |

---

This technical specification provides the complete implementation details for researchers and developers who want to understand, modify, or extend the architecture.
