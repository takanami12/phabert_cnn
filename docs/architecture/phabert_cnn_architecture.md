# PhaBERT-CNN_GeneGated — Architecture

```mermaid
flowchart TD
    %% ============ Input ============
    INPUT["DNA contig<br/>(100–1800 bp)<br/><i>ATCG...</i>"]:::input

    %% ============ Backbone ============
    subgraph BACKBONE["DNABERT-2 backbone — 117M params · 768-d hidden"]
        direction TB
        TOK["BPE tokenizer<br/>[B, T]"]:::seq
        BERT["12× Transformer encoder layers<br/>(ALiBi, no flash-attn)<br/><b>Phase 1: frozen</b><br/><b>Phase 2: fine-tune (full / LoRA)</b>"]:::seq
        TOK --> BERT
    end
    INPUT --> TOK
    BERT --> H["hidden_states<br/>[B, T, 768]"]:::tensor

    %% ============ Gene-feature branch ============
    subgraph GENE["Gene-feature branch — HMM annotation (parallel)"]
        direction TB
        HMM["HMM activations [B, 26]<br/>+ gene_stats [B, 4]"]:::geneIO
        ACTENC["ActivationEncoder<br/>log1p → LayerNorm → MLP"]:::gene
        ZCOND["z_cond [B, 128]"]:::tensor

        FAMEMB["family_emb [26, 768]<br/>+ activation projections"]:::geneIO
        FAMAGG["LearnableFamilyAggregator<br/>Transformer encoder<br/>+ Attention pooling<br/>(zero-activation families masked)"]:::gene
        FAMOUT["fam_agg [B, 128]"]:::tensor

        HMM --> ACTENC --> ZCOND
        FAMEMB --> FAMAGG --> FAMOUT
    end

    %% ============ FiLM + cross-attn ============
    H --> FILM
    ZCOND --> FILM

    FILM["<b>FiLM modulation</b><br/>h = h · (1 + γ) + β<br/><i>init: γ = β = 0 → identity day-1</i>"]:::gene
    H_FILM["h_modulated<br/>[B, T, 768]"]:::tensor
    FILM --> H_FILM

    H_FILM --> XATTN
    FAMOUT -.family tokens.-> XATTN

    XATTN["<b>FamilyCrossAttention</b><br/>DNA q ⟷ family k,v<br/><i>residual_scale = 0 → identity day-1</i>"]:::gene
    H_X["h_xattn [B, T, 768]"]:::tensor
    XATTN --> H_X

    %% ============ Parallel heads ============
    H_X --> CNN
    H_X --> ATTNPOOL

    subgraph HEADS["Parallel heads"]
        direction LR
        CNN["MultiScale CNN<br/>3× Conv1D, k=3,5,7<br/>each 128-d, GroupNorm, GELU<br/>concat → [B, 384]"]:::seq
        ATTNPOOL["AttentionPooling<br/>[B, 768] → Linear → [B, 128]"]:::seq
    end

    %% Codon branch (independent input)
    CODONIN["codon_features [B, 65]<br/>(64 RSCU + GC3)"]:::codonIO
    CODON["<b>CodonBranch</b><br/>LayerNorm → Linear(65→128)<br/>→ GELU → Dropout<br/>→ Linear → LayerNorm<br/>→ [B, 128]"]:::codon
    CODONIN --> CODON

    %% Stats norm passthrough
    GS["gene_stats_norm [B, 4]"]:::geneIO

    %% ============ Fusion ============
    CNN --> CONCAT
    ATTNPOOL --> CONCAT
    GS --> CONCAT
    FAMOUT --> CONCAT
    CODON --> CONCAT

    CONCAT["<b>Concat</b><br/>cnn(384) ⊕ attn(128) ⊕<br/>gs(4) ⊕ fam(128) ⊕ codon(128)<br/>→ [B, 772]"]:::fusion

    %% ============ Classifier ============
    CONCAT --> HEAD
    HEAD["<b>Classifier head</b><br/>LayerNorm → Dropout<br/>→ Linear(772→256) → ReLU → Dropout<br/>→ Linear(256→2)"]:::fusion

    HEAD --> LOGITS["logits [B, 2]"]:::tensor
    LOGITS --> SOFTMAX["softmax"]:::fusion
    SOFTMAX --> OUT["{Temperate=0,<br/>Virulent=1}"]:::output

    %% ============ Styles ============
    classDef input fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#000
    classDef output fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#000
    classDef seq fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.5px,color:#000
    classDef gene fill:#d1fae5,stroke:#047857,stroke-width:1.5px,color:#000
    classDef geneIO fill:#ecfdf5,stroke:#059669,stroke-width:1px,color:#000
    classDef codon fill:#ffedd5,stroke:#c2410c,stroke-width:1.5px,color:#000
    classDef codonIO fill:#fff7ed,stroke:#ea580c,stroke-width:1px,color:#000
    classDef fusion fill:#ede9fe,stroke:#6d28d9,stroke-width:1.5px,color:#000
    classDef tensor fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#000,font-style:italic
```

## Color legend

| Color | Module type |
|-------|-------------|
| 🔵 Blue | Sequence ops (DNABERT-2, Multi-scale CNN, Attention Pooling) |
| 🟢 Green | Gene features (HMM, ActivationEncoder, FamilyAggregator, FiLM, Cross-Attention) |
| 🟠 Orange | Codon features (CodonBranch — RSCU + GC3) |
| 🟣 Purple | Fusion + Classifier head |
| 🟡 Yellow | Input |
| 🔴 Red | Output |

## Key design choices

- **Day-1 identity**: FiLM (`γ = β = 0`) and cross-attention (`residual_scale = 0`) start as identity → backbone distribution preserved on Phase 1.
- **Two-phase training**: Phase 1 frozen backbone task warmup → Phase 2 unfreeze (full or LoRA).
- **Per-contig features**: HMM activations + RSCU computed strictly within contig window — no full-genome leak.
- **Zero-activation masking**: `LearnableFamilyAggregator` masks families with zero activation → only present families contribute.
- **Multi-modal fusion**: concatenation of 5 streams (CNN, attention pool, gene stats, family aggregate, codon) → 772-d → classifier.

## Render

- **GitHub / VS Code / IDEs hỗ trợ Mermaid**: file `.md` này render trực tiếp.
- **Export PNG/SVG**: dùng [mermaid.live](https://mermaid.live) (paste code block) hoặc:
  ```bash
  npx -p @mermaid-js/mermaid-cli mmdc -i phabert_cnn_architecture.md -o phabert_cnn_architecture.svg
  ```
