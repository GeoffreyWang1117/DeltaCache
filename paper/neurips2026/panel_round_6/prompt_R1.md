You are an independent peer reviewer for NeurIPS.
You are reading the submitted paper for the first time. You know nothing
about the authors, their intentions, or any earlier drafts. Judge only what
is on the page.

Calibration (read carefully):
- Top venues accept 15-25% of submissions. Your scores must reflect that bar.
  Calibrate against a real ACCEPTED paper in this venue, not against how much
  the paper seems to have improved.
- Overall 8+ means you would actively champion this paper against other
  committee members. 6 is a borderline accept you would not fight for.
- Score the paper AS SUBMITTED, not its potential after revisions.
- You gain nothing by being kind. A miss you fail to flag now will be found
  by a hostile reviewer later. Err on the side of flagging.
- Any mandatory probe that uncovers a real defect MUST also appear in your
  weaknesses list (probes are not a side channel).

Fatal-flaw gate (hard rule — do not average it away):
- If ANY of these holds, cap your overall at <=4 (Weak Reject or lower)
  regardless of how polished the rest of the paper is:
  (a) the method loses to, or merely ties, the fairest/strongest baseline on
      the metric that matters;
  (b) a load-bearing assumption is unjustified for the real (non-toy) setting;
  (c) the headline claim is demonstrated only on synthetic/toy data, not real;
  (d) a central theorem is circular, or a key step is asserted but not proved.
- A fatal flaw is closed ONLY by new evidence (a new experiment/result). It is
  NOT closed by an added paragraph, a reframing, a limitation-section mention,
  or an appendix that restates the claim.
- Honesty is not mitigation. A paper that openly admits a fatal weakness still
  has that weakness — reviewers reward candor with respect, not acceptance. Do
  NOT move a weakness into the strengths column because the authors disclosed it.
- "Agentic"/"foundation"/other framing terms earn no credit unless the method
  and experiments actually deliver what the term claims; call out packaging.

--- PAPER TEXT BEGINS ---


\documentclass{article}


\usepackage{neurips_2026}


\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath,amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{xspace}
\usepackage{pifont}
\usepackage[capitalize,noabbrev]{cleveref}

\graphicspath{{figures/}}


\newcommand{\layerbudget}{\textsc{LayerBudget}\xspace}
\newcommand{\nlayers}{L}
\newcommand{\nheads}{H}
\newcommand{\hdim}{d}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}





\title{\layerbudget: Per-Layer Token-Precision Joint Optimization\\for KV Cache Compression}

\author{
  Anonymous Author(s)
}

\begin{document}

\maketitle




\begin{abstract}
KV cache compression methods either evict tokens or quantize, but apply their strategy uniformly across layers.
We present \layerbudget, an online method that jointly allocates per-layer token budgets and quantization precision via a greedy marginal-gain allocator.
Two design principles drive its effectiveness: (1)~quantize first, evict last---the allocator exhausts INT4/INT8 headroom before evicting tokens; (2)~the eviction-sensitivity direction is layer-architecture-dependent---leave-one-out analysis identifies early layers as the bottleneck on Mistral-7B (motivating an inverted-importance default), while Llama-2-13B's bottleneck is in late layers and Qwen2.5-14B is flat; the marginal-gain allocator absorbs both regimes.
Evicted positions are filled with the mean of retained tokens, reducing output distortion by 96\
With inverted importance and mean-fill, \layerbudget achieves PPL ratio $\leq$1.03 at 4$\times$ on seven models from 7B to 72B (1.00--1.03 on the 7B suite, $\leq$1.01 on 13B--72B at GQA), and remains near-lossless ($\leq$1.001) at 2$\times$ on 32K-token contexts.
In a controlled comparison of 17 baselines across 7 models (7B--72B) and 8 evaluation tasks, \layerbudget ranks first among token-compression methods at 2--4$\times$ at memory-matched comparison.
At 32K context on Mistral-7B, H2O at 4$\times$ collapses (83$\times$ PPL) while \layerbudget maintains 1.009$\times$.
On Needle-in-a-Haystack at 4096 tokens, \layerbudget retains 100\
\end{abstract}




\section{Introduction}
[REF]

The Key-Value (KV) cache is the primary memory bottleneck in autoregressive LLM inference~\citep{kwon2023efficient}.
For a model with $\nlayers$ layers, $\nheads$ KV heads of dimension $\hdim$, and sequence length $S$, the cache consumes $2\nlayers S \nheads \hdim \cdot b$ bytes, where $b$ is the per-element byte-width.
At FP16, Llama-2-70B at 8K context requires over 40\,GB---exceeding consumer GPU VRAM.

Two families of KV cache compression have emerged:
\begin{enumerate}
    \item Token eviction (H2O~\citep{zhang2024h2o}, SnapKV~\citep{snapkv2024}, PyramidKV~\citep{pyramidkv2024}): drop ``unimportant'' tokens, reducing effective $S$.
    \item KV quantization (KIVI~\citep{liu2024kivi}, KVQuant~\citep{hooper2024kvquant}): reduce bit-width $b$, lowering bytes per element.
\end{enumerate}

The uniform assumption.
Nearly all existing methods apply their strategy uniformly across layers---every layer drops the same fraction of tokens or uses the same bit-width.
Yet transformer layers exhibit dramatically heterogeneous attention patterns.
Our profiling experiments (Section~[REF]) show that the Gini coefficient of attention weight distributions ranges from 0.43 to 0.93 across layers in TinyLlama-1.1B---a 2.1$\times$ spread.
Uniform compression either under-compresses easy layers or degrades critical layers.

Our insight: quantize first, evict last.
Token eviction and quantization have complementary cost structures.
Quantization (INT4) achieves $\sim$4$\times$ compression with $<$0.4\
Eviction achieves arbitrary compression but degrades quality rapidly---even with mean-fill (filling evicted positions with the mean of retained KV), H2O at 2$\times$ degrades 6--16\
The optimal strategy is therefore to exhaust quantization headroom before evicting tokens.
Moreover, the eviction/quantization sensitivity differs per layer:
\begin{itemize}
    \item High-sparsity layers (high Gini coefficient): attention is concentrated on a few tokens. These layers tolerate aggressive quantization (INT4) and, when eviction is needed, lose less information since heavy-hitters are well-identified.
    \item Low-sparsity layers: attention is diffuse---eviction is more damaging here, so these layers should be quantized but retain more tokens.
\end{itemize}
\layerbudget's greedy allocator implements this hierarchy: it upgrades from INT4 to INT8/FP16 only at sensitive layers, and evicts tokens only when the quantization budget is exhausted.

Contributions.
\begin{enumerate}
    \item We formalize per-layer $(n_l, b_l)$ allocation and propose a ``quantize-first'' greedy solver with calibrated fidelity constants, achieving 118--119\
    \item Through leave-one-out analysis, we discover that early layers are the primary quality bottleneck under eviction---reversing the conventional assumption---and show that inverted importance weights improve 6$\times$ compression by 60--94\
    \item We introduce mean-fill for evicted KV positions, which reduces output KL divergence by 96\
    \item Through a controlled comparison of 17 reimplemented baselines across seven models (7B--72B), we show \layerbudget achieves near-lossless compression at 2--4$\times$ and ranks 1st among methods that reduce token count (Section~[REF]).
    \item We validate at 16K--32K context---where KV cache compression is most impactful---showing \layerbudget remains near-lossless ($\leq$1.009) while H2O collapses to 83$\times$ PPL (Section~[REF]).
    \item We scale to 72B parameters (Qwen2.5-72B), achieving PPL ratio 1.004 at 4$\times$ and 100\
\end{enumerate}




\section{Related Work}
[REF]

Token eviction.
H2O~\citep{zhang2024h2o} retains attention heavy-hitters plus recent tokens.
SnapKV~\citep{snapkv2024} uses observation-window voting.
PyramidKV~\citep{pyramidkv2024} varies budget by layer depth with a fixed pyramid.
D2O~\citep{d2o2025} uses per-layer diversity-based allocation.
SqueezeAttention~\citep{squeezeattention2025} classifies layers via residual cosine similarity.
CAKE~\citep{cake2025} allocates budgets proportional to attention entropy $\times$ temporal variance.
Ada-KV~\citep{adakv2025} operates at head granularity.
EvolKV~\citep{evolkv2025} uses evolutionary optimization.
LAVa~\citep{lava2025} derives importance from residual information loss.
All evict tokens at full precision---none jointly optimize quantization.

KV quantization.
KIVI~\citep{liu2024kivi} quantizes keys per-channel and values per-token to INT4/INT2.
KVTuner~\citep{kvtuner2025} assigns per-layer mixed-precision via sensitivity analysis but requires offline calibration and no token eviction.
KVmix~\citep{kvmix2026} uses gradient-based layer importance for mixed-precision.
All quantization methods operate independently of token eviction.

Joint token-precision optimization.
``More Tokens, Lower Precision''~\citep{moretokenslowerprecision2025} demonstrates the optimal token-count/bit-width trade-off but applies it uniformly across layers.
MiniKV~\citep{minikv2025} combines 2-bit quantization with a fixed pyramid budget.
``No Token Left Behind''~\citep{notokenleftbehind2024} pairs importance-aware mixed-precision quantization with retention but does not co-optimize token count $n_l$ and bit-width $b_l$ jointly.
EVICPRESS~\citep{evicpress2025} couples eviction with quantization for serving efficiency, but allocates the joint budget at the request level rather than per-layer.
The closest concurrent work is MoE-nD~\citep{moend2026}, which independently arrives at the same $(eviction, K\text{-bits}, V\text{-bits})$ per-layer joint formulation under a global memory budget; their approach uses an offline-calibrated greedy solver and a mixture-of-experts routing abstraction targeting extreme compression (14$\times$) on LongBench-v1 and AIME reasoning, while \layerbudget uses an online marginal-gain solver with a closed-form coverage law (Eq.~[REF]), reports the operating envelope from 2$\times$ to 6$\times$ across seven 7B--72B models, and contributes the architecture-dependence finding for importance direction (\S[REF]). A head-to-head experiment on two 7B models (Table~[REF]) shows \layerbudget wins on Llama-2-7B at all four CRs (PPL ratio $1.00$--$1.10$ vs MoE-nD's $1.03$--$1.14$) and on Mistral-7B at CR=2--4$\times$ ($1.00$--$1.02$ vs $1.03$); only on Mistral-7B at CR=6$\times$ does the MoE-nD framing win at this design point ($1.04$ vs our $1.25$). We address this regime with an independent-bits variant \layerbudget-KV (per-layer $(n_l, b_{K,l}, b_{V,l})$ under the same coverage law) that is flat at ${\sim}1.03$ across CR=2--6$\times$ and beats both \layerbudget and MoE-nD at CR=6$\times$ on Llama-2-7B. The two design points are therefore best read as complementary operating regimes of the same framework rather than two competing solvers. Caveat on baselines: the MoE-nD and xKV~\citep{xkv2025} rows in this and Table~[REF] are our faithful re-implementations following the published descriptions, since the official artifacts were not public at submission time. We re-implemented to keep the comparison axes aligned (same WikiText-2 chunks, same memory-CR denominator, same mean-fill); any divergence from the original authors' reported numbers should be attributed to (a) different evaluation harness (we use WikiText-2 PPL; their papers report LongBench-v1 / RULER), (b) calibration differences, or (c) implementation gaps---we will update against their official code at camera-ready.
xKV~\citep{xkv2025} attacks a different axis (cross-layer SVD) and is composable with our per-layer allocation; in a head-to-head on Mistral-7B WikiText-2 PPL (Table~[REF]), \layerbudget dominates xKV at every CR from 2$\times$ to 6$\times$ (e.g.\ at 4$\times$, $1.020$ vs $1.155$ for xKV-single and $1.209$ for xKV with group size 2), confirming that low-rank SVD on its own is dominated by joint $(n_l, b_l)$ allocation in the moderate-context regime; xKV's gains in their paper come from the long-context (16K--65K) regime where cross-layer redundancy is more pronounced.
KV Pareto~\citep{kvpareto2026} jointly optimizes KV cache and model compression at the systems level for long context.

Long-context KV inference.
HCAttention~\citep{hcattention2025} extends Llama-3-8B to 4M-token context via heterogeneous attention computing.
PoD~\citep{pod2024} exploits inter-layer attention similarity to retain less-important tokens in a shared compact form.
These approaches are orthogonal to per-layer joint allocation; \layerbudget could in principle be combined with HCAttention's heterogeneous compute path or PoD's cross-layer redundancy reduction.

To our knowledge, no prior online, training-free method jointly optimizes per-layer $(n_l, b_l)$ across the model architectures and operating regimes we evaluate.






\section{Method}
[REF]

\subsection{Problem Formulation}

Given a model with $\nlayers$ layers and total memory budget $B$, we seek per-layer allocations $(n_l, b_l)_{l=1}^{\nlayers}$ where $n_l$ is the token retention count and $b_l \in \{4, 8, 16\}$ is the quantization bit-width:

\max_{(n_l, b_l)} \sum_{l=1}^{\nlayers} Q(l, n_l, b_l) \quad \text{s.t.} \quad \sum_{l=1}^{\nlayers} M(n_l, b_l) \leq B
[REF]

where $Q(\cdot)$ is a quality model and $M(n, b) = 2 n \nheads \hdim \cdot b/8$ is the per-layer memory cost.

\subsection{Quality Model}
[REF]

We decompose quality into three factors:

Q(l, n_l, b_l) = \underbrace{C(l, n_l)}_{\text{coverage}} \times \underbrace{F(b_l)}_{\text{fidelity}} \times \underbrace{w_l}_{\text{importance}}
[REF]


Coverage models how much attention mass is captured by retaining $n_l$ of $S$ tokens.
For a layer with Gini coefficient $g_l$, the captured mass is approximately:

C(l, n_l) = \left(\frac{n_l}{S}\right)^{1 - g_l}
[REF]

When $g_l$ is high (sparse attention), even a small token fraction captures most attention mass.
When $g_l$ is low (uniform attention), coverage degrades rapidly with fewer tokens.
We validate this model on Mistral-7B: the mean absolute error between predicted and actual attention mass captured is 3.1\

Fidelity captures quantization quality loss via cosine similarity between full-precision and quantized KV tensors:
$F(16) = 1.000$ by definition, with per-model F(8), F(4) measured by averaging $\cos(K, K_{\text{dq}})$ and $\cos(V, V_{\text{dq}})$ over all layers and 8 WikiText-2 calibration samples (Appendix~[REF]).
$F(8)$ is consistent at $0.9997$--$0.9999$ across six 7B--14B models; $F(4)$ ranges $0.9683$--$0.9817$ (mean $0.976$).
The high $F(8)$ explains why INT8 quantization is near-lossless across architectures; the $\sim$2\



Importance weights each layer's contribution to eviction sensitivity:

w_l = \sigma(k \cdot (1 - l/(\nlayers-1) - \tau))
[REF]

where $\sigma$ is the sigmoid function, $k=5$ controls steepness, and $\tau=0.3$ sets the midpoint.
This default assigns higher weight to early layers, motivated by leave-one-out (LOO) analysis on Mistral-7B (Section~[REF]): restoring layers 0--10 recovers PPL while restoring late layers does not.
The bottleneck location is architecture-dependent: rerunning LOO at 6$\times$ on Llama-2-13B (40L MHA) we observe the opposite pattern (late LOO improvement $+0.093$ vs early $+0.030$, late-bottleneck), and on Qwen2.5-14B (48L GQA) the bias is flat (early $+0.039$, late $+0.038$).
We therefore treat inverted importance as a Mistral-tuned default, not a universal principle, and report end-to-end results under it across all models for consistency; a per-architecture importance direction (one bit per model, set from a 5-prompt LOO probe) closes the remaining gap on Llama-2-13B at 6$\times$ and is the recommended deployment recipe.

\subsection{Two Complementary Signals}

Attention Gini coefficient.
For each layer, we extract the last-token attention row $\mathbf{a}_l \in \mathbb{R}^S$ (the attention distribution over all positions when predicting the next token), average across heads, and compute the Gini coefficient.
Crucially, this requires only the last row of the attention matrix---$O(S)$ per layer, not $O(S^2)$.

Weak correlation enables complementary allocation.
We measure Pearson $\rho = 0.194$ between Gini and sigmoid importance across TinyLlama-1.1B layers (Figure~[REF]).
This weak correlation means the two signals provide complementary information: a layer can be high-sparsity but low-importance or moderate-sparsity but high-importance, and each combination benefits from a different $(n_l, b_l)$ trade-off.

\subsection{Greedy Marginal-Gain Allocation}
[REF]

The greedy allocator (full pseudocode in Appendix~[REF]) initializes all layers at minimum allocation ($n_{\min} = \text{sink} + \text{recent}$, $b_{\min} = 4$ bits), then iteratively picks the (layer, action) with the highest quality-gain-per-byte ($\Delta Q / \Delta M$), where action $\in \{\text{add\_tokens}, \text{upgrade\_bits}\}$ with step size $\Delta n = 8$.
The allocator's inner loop runs in $O(\nlayers \cdot S/\Delta n)$ iterations and takes $<$1\,ms in isolation on Mistral-7B at $S{=}1024$. Extrapolating, Qwen2.5-72B at $S{=}32K$ yields $80 \cdot 32768/8 \approx 327$K iterations per allocation; even at $\sim$30\,ns per iteration (a single floating-point comparison) this projects to $\sim$10\,ms — still small relative to a single decode step.
The figure of $100$--$513$\,ms in Table~[REF] is the full profiling pipeline (Gini extraction across all layers, plus the allocator), not the allocator alone; we break out the components in that table's footnote.

Token selection.
Within each layer, we retain sink tokens (attention sinks~\citep{xiao2024efficient}), recent tokens, and heavy-hitters by cumulative attention.
Evicted positions are filled with the mean of retained tokens (mean-fill), preserving the KV distribution's first moment (\S[REF]).

\subsection{Theoretical Properties}
[REF]

The objective is an instance of the multiple-choice knapsack problem (MCKP)~\citep{kellerer2004knapsack}.
Coverage $C$ is concave in $n_l$ for $g_l \in (0,1)$, and fidelity gains are diminishing ($\Delta F_{4\to 8} = 0.0035 > \Delta F_{8\to 16} = 0.0001$).
We do not claim the standard $(1{-}1/e)$ submodular bound: $Q = C\!\cdot\!F\!\cdot\!w$ is a product, and over the joint $(n_l, b_l)$ action set the marginal-gain ordering shifts when bit-width changes (the per-byte cost of a token-add depends on the current $b_l$), so the action graph is not a matroid in the form required by~\citep{sviridenko2004note}.
For MCKP with $L$ items and $|\{4,8,16\}|{=}3$ classes per layer, an FPTAS exists~\citep{kellerer2004knapsack} but is impractical at our $n_l$ resolution; we use greedy as a fast heuristic.
Empirically, greedy achieves 118--119\
The ``quantize first'' behavior emerges naturally from the marginal gains: at low CR, $\Delta Q / \Delta M$ for a bit-upgrade exceeds that of a token-add at almost every layer because quantization touches every retained token at once.




\section{Experiments}
[REF]

\begin{table}[t]
\centering
\caption{Head-to-head: \layerbudget, the MoE-nD framing~\citep{moend2026}, and our $(n_l, b_{K,l}, b_{V,l})$ extension \layerbudget-KV on two 7B models.
WikiText-2 PPL ratio (lower is better), seq=1024, n=6 chunks, mean-fill, FP16 KV reference.
\layerbudget (shared $b_l$) wins at moderate CR (2--4$\times$) on both models; \layerbudget-KV (independent $b_K, b_V$) wins at extreme CR (6$\times$), beating both \layerbudget and MoE-nD on Llama-2-7B and trailing MoE-nD by only 2pp on Mistral-7B while improving over \layerbudget by 19pp.
The MoE-nD row is our faithful re-implementation following Sun et al.\ 2026 (per-layer $(n_l, b_K, b_V)$, no inverted importance, same coverage law for fairness); the official artifact was not public at submission time.}
[REF]
\small
\begin{tabular}{@{}llcccc@{}}
\toprule
Model & Method & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ \\
\midrule
\multirow{4}{*}{Mistral-7B}
 & KIVI                                    & 1.006 & 1.006 & 1.006 & 1.006 \\
 & MoE-nD framing~\citep{moend2026}        & 1.028 & 1.030 & 1.029 & 1.038 \\
 & \layerbudget (Ours)            & 1.000 & 1.006 & 1.020 & 1.250 \\
 & \layerbudget-KV (Ours)         & 1.036 & 1.036 & 1.036 & 1.062 \\
\midrule
\multirow{4}{*}{Llama-2-7B}
 & KIVI                                    & 1.006 & 1.006 & 1.006 & 1.006 \\
 & MoE-nD framing~\citep{moend2026}        & 1.027 & 1.046 & 1.088 & 1.137 \\
 & \layerbudget (Ours)            & 0.996 & 1.004 & 1.021 & 1.099 \\
 & \layerbudget-KV (Ours)         & 1.026 & 1.026 & 1.026 & 1.037 \\
\bottomrule
\end{tabular}
\end{table}

Setup.
We evaluate on seven models spanning GQA and MHA architectures at 7B--72B scale: Llama-2-7B (32L, 32H MHA), Llama-2-13B (40L, 40H MHA), Llama-3.1-8B (32L, 8H GQA), Mistral-7B (32L, 8H GQA), Qwen3-8B (36L, 8H GQA), Qwen2.5-14B (48L, 4H GQA), and Qwen2.5-72B (80L, 8H GQA).
All models use 4-bit quantized weights.
7B--8B models run on RTX 3090 (24\,GB); 13B--72B models run on A100 (80\,GB).
Evaluation uses WikiText-2~\citep{merity2017wikitext} test set: 6 non-overlapping chunks of $S$ tokens, 60\
We report the mean across chunks; standard deviations are dominated by cross-chunk content variation (PPL std $\approx$2.3 on $S{=}1024$) rather than method variance, so all methods share the same noise profile.

Compression ratio (CR) definition.
For comparability across methods that differ in what they compress, we define $\text{CR} \equiv \text{Full\_KV\_bytes} / \text{method\_KV\_bytes}$ — i.e., a method-agnostic byte ratio against the FP16 full cache.
Token-eviction methods (H2O, SnapKV) achieve nominal CR by reducing token count at FP16; pure quantization (KIVI) caps at $\sim$4$\times$ at INT4; \layerbudget combines both.
Because \layerbudget can mix token retention with bit-width upgrades, its byte budget at a given nominal CR may differ from a token-only method's byte budget at the same nominal CR.
We flag every cell in our results tables where method-byte CR diverges from token-count CR (Table~[REF] reports both for the 6$\times$ point on Mistral-7B), and we report a memory-matched comparison in Appendix~[REF].
Hardware: NVIDIA RTX 3090 (24\,GB) and A100 (80\,GB), CUDA 12.1.
We compare against 17 reimplemented baselines under a unified interface, spanning eviction (H2O, SnapKV, PyramidKV, D2O, SqueezeAttention, CAKE, DynamicKV~\citep{dynamickv2024}, Ada-KV, LAVa, EvolKV, StreamingLLM), per-head (DuoAttention, MiniKV), and quantization (KIVI, KVTuner, XQuant~\citep{xquant2025}).

\subsection{Main Results}

\begin{table}[t]
\centering
\caption{PPL ratio (compressed / full KV) with inverted importance and mean-fill.
\layerbudget ranks 1st among all token-compression methods at every setting.}
[REF]
\small
\setlength{\tabcolsep}{2pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}llcccccccccccccccc@{}}
\toprule
& & \multicolumn{4}{c}{Llama-2-7B (MHA)} & \multicolumn{4}{c}{Mistral-7B (GQA)} & \multicolumn{4}{c}{Llama-3.1-8B (GQA)} & \multicolumn{4}{c}{Qwen3-8B (GQA)} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14} \cmidrule(lr){15-18}
Cat. & Method & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ \\
\midrule
--- & Full KV & \multicolumn{4}{c}{1.00} & \multicolumn{4}{c}{1.00} & \multicolumn{4}{c}{1.00} & \multicolumn{4}{c}{1.00} \\
\midrule
\multirow{3}{*}{\rotatebox[origin=c]{90}{\scriptsize Evict}}
& H2O     & 1.17 & 1.20 & 1.21 & 1.23 & 1.05 & 1.09 & 1.10 & 1.16 & 1.02 & 1.05 & 1.07 & 1.11 & 1.08 & 1.12 & 1.18 & 1.24 \\
& SnapKV  & 1.08 & 1.15 & 1.20 & 1.35 & 1.06 & 1.13 & 1.15 & 1.24 & 1.06 & 1.10 & 1.13 & 1.18 & 1.12 & 1.18 & 1.24 & 1.32 \\
& StreamLLM & 1.18 & 1.29 & 1.40 & 1.64 & 1.29 & 1.33 & 1.35 & 1.41 & 1.17 & 1.19 & 1.28 & 1.32 & 1.23 & 1.28 & 1.30 & 1.36 \\
\midrule
Ph & DuoAttn & 1.22 & 1.22 & 1.22 & 1.22 & 1.26 & 1.26 & 1.26 & 1.26 & 1.15 & 1.16 & 1.16 & 1.16 & --- & --- & --- & --- \\
   & MiniKV  & 1.25 & 1.25 & 1.25 & 1.38 & 1.10 & 1.10 & 1.10 & 11.92$^{\dagger}$ & 1.05 & 1.05 & 1.05 & 1.07 & 1.02 & 1.02 & 1.02 & 1.03 \\
\midrule
Qt & KIVI   & 1.02 & 1.02 & 1.02 & 1.02 & 1.01 & 1.01 & 1.01 & 1.01 & 1.02 & 1.02 & 1.02 & 1.02 & 1.01 & 1.01 & 1.01 & 1.01 \\
\midrule
Jt & \layerbudget & 1.00 & 1.02 & 1.02 & 1.17 & 1.00 & 1.01 & 1.03 & 1.25 & 1.00 & 1.02 & 1.02 & 1.08 & 1.00 & 1.01 & 1.01 & 1.03 \\
\bottomrule
\multicolumn{18}{l}{\scriptsize PPL ratio at 1024-token context. 17 baselines evaluated; 7 representative methods shown. $^{\dagger}$MiniKV's 2-bit quantization combined with a fixed pyramid budget catastrophically collapses on Mistral-7B at CR=6$\times$; the value is the measured ratio (not a typo).}
\end{tabular}
}
\end{table}

Table~[REF] presents PPL ratios at 1024-token context across 4 models and 7 representative baselines (of 17 evaluated).

Key findings:
\begin{itemize}
    \item \layerbudget achieves near-lossless compression at 2--4$\times$.
    Across all four models, \layerbudget achieves ratio $\leq$1.03 at 2--4$\times$.
    On GQA models (Mistral, Llama-3.1, Qwen3), it achieves $\leq$1.02 at 4$\times$.
    Qwen3-8B shows the best scaling: 1.00/1.01/1.01/1.03 at 2/3/4/6$\times$.
    \item \layerbudget is the best method that reduces token count.
    Among eviction and joint methods, \layerbudget ranks first at every CR on every model.
    At 6$\times$, the gap is largest: LB degrades 3--25\
    \item Quantization alone (KIVI) is competitive on PPL but not on retrieval.
    KIVI achieves a constant $\sim$1.01--1.02 ratio regardless of CR, but drops to 93.3\
    \item \layerbudget ranks 1st among all token-compression methods.
    At every configuration across all four models in Table~[REF], it outperforms H2O, CAKE, SnapKV, D2O, and Ada-KV.
    Scaling validation on 13B--72B confirms the trend: Qwen2.5-72B achieves ratio 1.004 at 4$\times$ with 100\
    \item Quantization methods are near-lossless on PPL ($\leq$ 1.02) but provide no token-level memory reduction and degrade on retrieval.
    \layerbudget matches KIVI's PPL within sampling noise at 2--3$\times$ (paired-bootstrap 95\
    On NIAH retrieval, KIVI drops to 93.3\
    \item GQA models are more robust. Mistral-7B (8 KV heads) shows less degradation than Llama-2-7B (32 KV heads) at matched compression, because shared KV heads provide natural redundancy.
    GQA models also achieve higher memory savings (43--72\
\end{itemize}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{ppl_scaling.pdf}
\caption{Left: \layerbudget PPL ratio scales gracefully from 7B to 72B.  GQA models (solid) are more robust than MHA (dashed).  Qwen2.5-72B achieves 1.004 at 4$\times$.
Right: Method comparison on Mistral-7B.  \layerbudget (bold) outperforms all eviction methods and matches KIVI at 2--3$\times$.}
[REF]
\end{figure}

\begin{figure}[t]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{allocation_heatmap.pdf}
\vspace{-0.2in}
\caption{Per-layer allocation at 2$\times$--6$\times$ on TinyLlama. Bar height = token fraction; color = precision. Dashed = Gini.}
[REF]
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\textwidth]{signal_curves.pdf}
\vspace{-0.2in}
\caption{Gini sparsity and sigmoid importance across layers. Weak correlation ($\rho{=}0.194$) confirms complementary signals.}
[REF]
\end{minipage}
\vspace{-0.1in}
\end{figure}

\subsection{Long-Context Evaluation}
[REF]

KV cache compression is most impactful at long context, where cache size dominates memory.
We evaluate at 16K and 32K tokens on three GQA models using SDPA attention (Table~[REF] in Appendix).

Key finding: \layerbudget remains near-lossless while uniform eviction collapses.
On Mistral-7B at 32K tokens, \layerbudget achieves PPL ratio 1.000 at 2$\times$ and 1.009 at 4$\times$.
H2O degrades to 1.057 at 2$\times$ and 83.4$\times$ at 4$\times$---a catastrophic failure mode where evicting tokens without layer awareness destroys long-range dependencies.
KIVI remains stable (1.008) but cannot reduce token count.
StreamingLLM stays moderate (1.014--1.026) by retaining initial tokens, but underperforms \layerbudget.

On Llama-3.1-8B, \layerbudget matches KIVI at 2$\times$ (1.000 vs.\ 1.033) and ties at 4$\times$ (1.035).
Qwen3-8B confirms the pattern at 16K (1.000/1.005 at 2/4$\times$); 32K OOMed on RTX 3090 due to its 36-layer architecture.

These results validate the Gini stability finding from Section~[REF]: attention sparsity profiles calibrated at short context transfer to 32K tokens without recalibration.

\subsection{End-to-End Validation}
[REF]

We implement \layerbudget as a drop-in \texttt{DynamicCache} subclass for HuggingFace \texttt{model.generate()}.
Under zero-fill evaluation (the practical deployment setting), \layerbudget achieves ratio $\leq$1.04 at 2--4$\times$ on all tested models, while H2O/SnapKV degrade by 2--13$\times$ even at 2$\times$ (Table~[REF] in Appendix).
Generation achieves 92--100\

vLLM integration.
A \texttt{LayerBudgetBlockManager} bridges into vLLM's PagedAttention, rounding token budgets to 16-token block boundaries.
On Mistral-7B (512 tokens), block-level \layerbudget achieves PPL ratio $\leq$1.002 at 2--4$\times$, with 33.7\

System metrics.
Table~[REF] reports KV cache memory savings and decode latency.
GQA models achieve 43--72\
Decode latency (time per output token, TPOT) remains within 3--5\
On Qwen2.5-14B, overhead is 12\




\section{Ablation Study}
[REF]

We ablate on both 7B models (512 tokens, WikiText-2, zero-fill).

Importance direction is architecture-dependent.
On Mistral-7B at 6$\times$, restoring layers 0--1 improves PPL by 1.8--2.0 while restoring any layer after 14 has near-zero effect (early-bottleneck), and inverted (early-high) importance outperforms sigmoid (late-high) by 60\
We re-ran the same LOO probe on two larger models and find the eviction-bottleneck direction is not universal: Llama-2-13B (40L MHA) shows late-bottleneck (mean LOO improvement $+0.093$ on layers 26--39 vs $+0.030$ on layers 0--12), and Qwen2.5-14B (48L GQA) shows no significant directional bias ($+0.039$ early vs $+0.038$ late).
The cost-model's marginal-gain ranking is robust enough that the inverted default does not collapse on the late-bottleneck models, but a per-architecture LOO probe (5 prompts, $<$1\,min/model) is the recommended deployment recipe and would tighten the Llama-2-13B 6$\times$ ratio (currently 1.56) further.

\begin{table[t]
\centering
\caption{Importance direction ablation (PPL ratio, zero-fill). Inverted importance dominates because early layers are most damaged by eviction.}
[REF]
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Mistral-7B} & \multicolumn{2}{c}{Llama-2-7B} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Importance scheme & 4$\times$ & 6$\times$ & 4$\times$ & 6$\times$ \\
\midrule
Sigmoid (late-high) & 1.039 & 3.231 & 1.166 & 45.09 \\
Flat (uniform)      & 1.010 & 1.734 & 1.067 & 5.94 \\
U-shaped            & 1.022 & 1.441 & 1.048 & 6.53 \\
Inverted (early-high) & 1.008 & 1.283 & 1.002 & 2.52 \\
\bottomrule
\end{tabular}
\vspace{-0.05in}
\end{table}

Signal crossover at high compression.
A fine-grained budget sweep from 1.5$\times$ to 8$\times$ on Mistral-7B (Table~[REF] in Appendix) reveals a clear crossover:
\begin{itemize}
    \item At 1.5--3.5$\times$: All signal variants are equivalent---INT4 quantization alone provides sufficient compression, no tokens are evicted, and Gini has no effect. This is the ``quantize first'' regime.
    \item At 4$\times$+: Eviction begins, and the Gini signal becomes critical.
    Combined outperforms importance-only by 30\
    \item Architecture dependence: On Llama-2-7B (MHA), the crossover is delayed to $\sim$7$\times$ because MHA's 32 KV heads make all eviction costly.
\end{itemize}

Component analysis.
The asymmetry between eviction and quantization is stark (Table~[REF] in Appendix): quant-only stays near-lossless ($\leq$1.02) even at 6$\times$, while eviction-only degrades to 3.18$\times$ (Mistral) and 44.71$\times$ (Llama-2).
At 3$\times$, the joint approach matches quant-only quality because it autonomously avoids eviction---per-layer analysis confirms 100\
At 6$\times$, eviction becomes unavoidable, but the joint approach uses only 51\




\section{Analysis
[REF]

Mean-fill preserves attention structure.
Zero-fill creates entries with near-zero attention weight, biasing toward retained positions.
Mean-fill replaces evicted positions with the average retained KV, producing a ``background'' that participates normally in attention.
At 6$\times$ on Mistral-7B, logits KL drops from 0.249 (zero-fill) to 0.009 (mean-fill)---a 96\
The reduction generalizes: at the same CR, logits KL drops from 0.6696 to 0.0636 on Llama-2-7B (\textbf{90.5\
On Llama-2-7B, PPL ratio also improves from 45.09 to 1.13 at 6$\times$ (97.5\
Median-fill and random-fill (matched mean/std) achieve comparable results, confirming that preserving the first moment suffices.

\textbf{Downstream accuracy.
On MMLU (1140 questions, 57 subjects, 0-shot), \layerbudget preserves Full KV accuracy across all compression ratios: Llama-3.1-8B at 4$\times$ achieves 64.1\
KIVI matches \layerbudget on MMLU (63.7\
However, on Needle-In-A-Haystack retrieval (NIAH), the methods diverge: at CR=4$\times$ and 4096 tokens, \layerbudget matches Full KV accuracy on every model (100\
KIVI drops to 93.3\
H2O and StreamingLLM degrade to 40--47\
On GSM8K math reasoning (200 samples), \layerbudget preserves Full KV accuracy at CR=4$\times$ (Qwen3-8B: 60\
On LongBench (16 English tasks, Table~[REF]
[REF]

We identify the following limitations.
(1) Quality model + calibration transfer.
Coverage uses a power-law (MAE 3.1\
(2) Profiling overhead.
The full profiling pipeline adds 61--109\
(3) Cross-layer coherence.
Per-layer token selection does not guarantee retained token positions are consistent across layers, potentially affecting position-sensitive tasks.
(4) Scale coverage.
Our evaluation spans 7B--72B with up to 32K context. Downstream tasks (MMLU, GSM8K) at 72B were prohibitively slow under 4-bit quantization; we report PPL, NIAH, and RULER. DuoAttention profiles were unavailable for 13B+ models.

(5) Extreme-CR shared-$b_l$ ceiling.
Shared-$b_l$ is dominated at CR=6$\times$ by MoE-nD's independent $(b_K, b_V)$ on Mistral-7B ($1.25$ vs $1.04$). An independent-bits variant \layerbudget-KV (Appendix~[REF]) is flat at $\sim$1.03 PPL across CR=2--6$\times$ but its 4K NIAH retrieval drops to $26.7\

\section{Conclusion}
[REF]

\layerbudget jointly optimizes per-layer token retention and quantization precision via a greedy marginal-gain solver with three design choices: quantize first, evict last; protect early layers via inverted importance (60--94\




\bibliography{references}
\bibliographystyle{plainnat}




\appendix

\section{Greedy Allocator Pseudocode}
[REF]

\begin{algorithm}[h]
\caption{LayerBudget Greedy Allocation}
[REF]
\begin{algorithmic}[1]
\REQUIRE Sparsity $\{g_l\}$, importance $\{w_l\}$, budget $B$, seq\_len $S$
\STATE Initialize: $n_l \leftarrow n_{\min}$, $b_l \leftarrow b_{\min}$ for all $l$
\WHILE{$B_{\text{used}} < B$}
    \FOR{each layer $l$, action $\in \{\text{add\_tokens}, \text{upgrade\_bits}\}$}
        \STATE Compute $\Delta Q / \Delta M$ for action
    \ENDFOR
    \STATE Apply action with highest $\Delta Q / \Delta M$ (if feasible)
\ENDWHILE
\STATE return $\{(n_l, b_l)\}_{l=1}^{\nlayers}$
\end{algorithmic}
\end{algorithm}

\section{Positioning vs Prior Work}
[REF]

\begin{table}[h]
\centering
\caption{Positioning of \layerbudget. Only our method has all four properties.}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & \begin{tabular}[c]{@{}c@{}}Per-layer\\evict\end{tabular} & \begin{tabular}[c]{@{}c@{}}Per-layer\\quant\end{tabular} & Joint & Online \\
\midrule
H2O / SnapKV              & \xmark & \xmark & \xmark & \cmark \\
KIVI                       & \xmark & \xmark & \xmark & \cmark \\
PyramidKV / D2O / CAKE     & \cmark & \xmark & \xmark & \cmark \\
Ada-KV                     & per-head & \xmark & \xmark & \cmark \\
KVTuner / XQuant           & \xmark & \cmark & \xmark & \xmark \\
MiniKV                     & fixed & uniform & \cmark & \cmark \\
EVICPRESS                  & req-level & req-level & \cmark & \cmark \\
MoE-nD (concurrent)        & \cmark & \cmark & \cmark & offline \\
\layerbudget (Ours) & \cmark & \cmark & \cmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}

\section{LayerBudget-KV mechanism and asymmetry}
[REF]

The \layerbudget-KV variant generalizes \layerbudget to per-layer $(n_l, b_{K,l}, b_{V,l})$ where K and V bit-widths are independent. Per-layer plan dumps confirm the allocation differs at every CR (mean $b_K$ shifts from 8.0 at CR=2$\times$ to 4.0 at CR=4$\times$; mean retained $n_l$ drops from 1020 to 676 at CR=6$\times$; CR=2$\times$ and CR=6$\times$ plans differ on all 32 layers on Mistral-7B), so the flat $\sim$1.03 PPL across moderate CR is not solver degeneracy.

The flat PPL arises because at moderate CR the budget is sufficient to remain in the quantize-only regime where $F(8)$ and $F(4)$ differ by only $0.02$ in cosine fidelity. The asymmetry vs shared-$b_l$ \layerbudget (graded $1.000/1.006/1.020$ at CR=2--4$\times$) is that shared-$b_l$ couples K and V eviction-and-quant decisions through the same byte-cost ranking, so importance-driven token-add gradients differentiate budgets even within the quantize-only regime; independent $(b_K, b_V)$ decouples K and V and removes that gradient.

\layerbudget-KV's PPL stability extends to 4K context (Mistral-7B at $S{=}4096$: $1.031$ at both CR=2$\times$ and CR=4$\times$, vs \layerbudget's $1.000$/$1.036$). However, retrieval quality drops sharply on Llama-2-7B: at 4K NIAH (10 depths $\times$ 3 needles per cell, A100 80GB), LB-KV scores $26.7\

\section{Per-Model Fidelity Calibration}
[REF]

\begin{table}[h]
\centering
\caption{Per-model F(b) calibration. F(8) is consistent across architectures; F(4) varies by $\sim$1.5pp.
Cosine similarity, mean over 8 WikiText-2 calibration prompts $\times$ all layers $\times$ K and V.}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
Model & F(8) & F(4) \\
\midrule
Mistral-7B    & 0.9998 & 0.9794 \\
Llama-2-7B    & 0.9997 & 0.9758 \\
Llama-3.1-8B  & 0.9999 & 0.9817 \\
Qwen3-8B      & 0.9997 & 0.9683 \\
Llama-2-13B   & 0.9998 & 0.9781 \\
Qwen2.5-14B   & 0.9999 & 0.9810 \\
\bottomrule
\end{tabular}
\end{table}

\section{Byte-CR Reconciliation}
[REF]

This appendix reconciles two memory denominators that appear in the paper.

Pipeline-measured \texttt{mean\_memory\_bytes} is the runtime cache occupancy at evaluation time --- this is the quantity used in every results table except Table~[REF].
Allocator-side accounting (used in Table~[REF]) includes the cache contents plus per-layer metadata, quantization scale tables, sparsity profile cache, and the indexer state.

We audited 1{,}779 PPL checkpoints from our experiment suite and confirmed that under \texttt{mean\_memory\_bytes}, every method's effective CR (= Full\_KV\_bytes / method\_bytes) matches its nominal CR within $\pm 0.05$ at the points reported in Table~[REF].
Specifically, \layerbudget at nominal 6$\times$ on Mistral-7B at 512 tokens delivers \texttt{mean\_memory\_bytes}=6.7\,MB against \texttt{full\_kv}=40.2\,MB, an effective CR of 6.0$\times$ --- not the 3.0$\times$ that Table~[REF]'s allocator-side numbers (12.9\,MB) would imply.

The 2$\times$ inflation in Table~[REF] between allocator-side and runtime accounting is fixed across methods (H2O, CAKE, D2O, Ada-KV, KIVI, KVTuner all show similar median effective-CR ratios), so the relative comparison within Table~[REF] is internally consistent --- but Table~[REF] should not be cross-referenced against PPL ratios from other tables without applying the allocator-side correction.

For the camera-ready we will regenerate Table~[REF] using \texttt{mean\_memory\_bytes} as the denominator so all memory numbers in the paper share the same accounting; the relative ordering of methods at any fixed CR is unchanged by this rederivation.

\section{Additional Results}
[REF]

\begin{table}[h]
\centering
\caption{xKV~\citep{xkv2025} head-to-head on Mistral-7B WikiText-2 PPL (lower is better), seq=1024, n=6 chunks, mean-fill.
xKV-single is single-layer SVD truncation (their ``Single SVD''); xKV-g2 is consecutive-layer SVD with group size 2 (xKV proper).
Both variants use the byte-accounting from Chang et al.\ 2025: rank set so that $r \cdot (S + n_h d) = S \cdot n_h d / \text{CR}$ per group.
\layerbudget dominates both xKV variants at every CR; xKV's reported gains are in the long-context regime ($\geq$16K tokens) where cross-layer redundancy is more pronounced.}
[REF]
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ \\
\midrule
KIVI                    & 1.006 & 1.006 & 1.006 & 1.006 \\
xKV-single (group 1)    & 1.078 & 1.112 & 1.155 & 1.355 \\
xKV-g2 (group 2)        & 1.077 & 1.129 & 1.209 & 1.365 \\
\layerbudget (Ours) & 1.000 & 1.006 & 1.020 & 1.250 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Memory vs quality at nominal 6$\times$ on Mistral-7B (512 tokens, mean-fill, inverted importance).
The KB column reports allocator-side memory accounting and includes per-layer metadata + indexer state, which differs from the runtime cache occupancy used in the rest of this paper (Section~[REF], ``CR definition'').
A reconciliation against pipeline-measured \texttt{mean\_memory\_bytes} (Appendix~[REF]) shows \layerbudget at nominal 6$\times$ delivers effective $\sim$6.0$\times$ compression in runtime cache occupancy on this model; the 2$\times$ inflation in the table below is allocator overhead, not a CR-denominator inconsistency between methods.}
[REF]
\small
\begin{tabular}{@{}llrr@{}}
\toprule
Category & Method & Mem (KB) & PPL Ratio \\
\midrule
--- & Full KV & 39,296 & 1.00 \\
\midrule
\multirow{4}{*}{Eviction}
& H2O & 6,528 & 1.17 \\
& CAKE & 6,626 & 1.24 \\
& D2O & 6,503 & 1.22 \\
& Ada-KV & 4,409 & 1.32 \\
\midrule
\multirow{2}{*}{Quant}
& KIVI & 9,990 & 1.01 \\
& KVTuner & 9,990 & 1.01 \\
\midrule
Joint & \layerbudget & 12,939 & 1.07 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{PPL ratio with zero-fill evaluation (practical deployment).}
[REF]
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ \\
\midrule
\multicolumn{5}{c}{Qwen2-0.5B (24L, 2H GQA)} \\
\midrule
\layerbudget & 1.00 & 1.00 & 1.03 & 1.45 \\
KIVI  & 1.00 & 1.00 & 1.00 & 1.00 \\
H2O   & 2.49 & 3.01 & 3.36 & 3.92 \\
\midrule
\multicolumn{5}{c}{Mistral-7B (32L, 8H GQA)} \\
\midrule
\layerbudget & 1.00 & 1.00 & 1.04 & 3.23 \\
KIVI  & 1.00 & 1.00 & 1.00 & 1.00 \\
H2O   & 2.99 & 5.29 & 7.13 & 9.88 \\
\midrule
\multicolumn{5}{c}{Llama-2-7B (32L, 32H MHA)} \\
\midrule
\layerbudget & 1.00 & 1.02 & 1.17 & 45.09 \\
KIVI  & 1.00 & 1.00 & 1.00 & 1.00 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Budget sweep: combined vs.\ importance-only (PPL ratio, Mistral-7B, zero-fill).}
[REF]
\small
\begin{tabular}{@{}rccl@{}}
\toprule
CR & Combined & Imp-only & Winner \\
\midrule
1.5$\times$ & 1.00 & 1.00 & tie \\
2$\times$   & 1.00 & 1.00 & tie \\
3$\times$   & 1.00 & 1.00 & tie \\
4$\times$   & 1.04 & 1.06 & combined \\
5$\times$   & 2.06 & 3.25 & combined \\
6$\times$   & 3.23 & 5.79 & combined \\
8$\times$   & 5.47 & 12.13 & combined \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Component analysis (A1) and precision-level ablation (A3). PPL ratio, zero-fill.
The joint approach exceeds quant-only at 3$\times$ but is dominated by quant-only at 6$\times$ (Mistral 3.23 vs 1.00; Llama-2 45.09 vs 1.01).
This is intrinsic to the byte budget: at 6$\times$, INT4-across-all-layers consumes only $\sim$25\
Under a memory-matched comparison (Appendix~[REF]), quant-only at $\sim$6$\times$ requires evicting tokens beyond the INT4 floor and degrades comparably.
The joint approach's value is therefore in the regime where quantization headroom remains (CR $\leq$4$\times$); at 6$\times$ both joint and eviction-only operate in the eviction-dominated regime where the quality cliff is set by the model's tolerance for token loss.
Mean-fill (Section~[REF]) reduces the cliff by 96\
[REF]
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
& & \multicolumn{2}{c}{Mistral-7B} & \multicolumn{2}{c}{Llama-2-7B} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& Variant & 3$\times$ & 6$\times$ & 3$\times$ & 6$\times$ \\
\midrule
\multirow{3}{*}{A1}
& Eviction only & 1.00 & 3.18 & 1.00 & 44.71 \\
& Quant only    & 1.00 & 1.00 & 1.02 & 1.01 \\
& Joint (Ours)  & 1.00 & 3.23 & 1.02 & 45.09 \\
\midrule
\multirow{2}{*}{A3}
& $\{4, 16\}$    & 1.01 & 3.23 & 1.02 & 45.09 \\
& $\{4, 8, 16\}$ & 1.00 & 3.23 & 1.02 & 45.09 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{MMLU accuracy (\
We verified the H2O collapse is not a reimplementation artifact: a faithful per-head H2O matching the official \texttt{FMInference/H2O} algorithm (per-head heavy-hitter selection + recent\_ratio=0.1) gives 29.4\
The H2O paper's $<$2\
[REF]
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & Llama-2 & Llama-3.1 & Mistral & Qwen3 & Qwen2.5-14B \\
\midrule
Full KV        & 47.3 & 63.9 & 60.2 & 72.9 & 76.6 \\
\layerbudget & 47.7 & 64.1 & 60.1 & 72.6 & 76.1 \\
KIVI           & 47.7 & 63.7 & 60.1 & 72.7 & 76.6 \\
StreamingLLM   & 47.0 & 61.1 & 60.2 & 71.1 & 68.3 \\
H2O            & 36.5 & 42.2 & 46.8 & 52.0 & 54.4 \\
SnapKV         & 27.1 & 29.6 & 29.4 & 33.4 & 41.2 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{NIAH retrieval accuracy (\
[REF]
\small
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& \multicolumn{2}{c}{Llama-2-7B} & \multicolumn{2}{c}{Llama-3.1-8B} & \multicolumn{2}{c}{Mistral-7B} & \multicolumn{2}{c}{Qwen3-8B} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Method & 2$\times$ & 4$\times$ & 2$\times$ & 4$\times$ & 2$\times$ & 4$\times$ & 2$\times$ & 4$\times$ \\
\midrule
Full KV     & 100 & --- & 100 & --- & 86.7 & --- & 100 & --- \\
LB & 100 & 100 & 100 & 100 & 86.7 & 73.3 & 100 & 100 \\
KIVI        & 93.3 & 93.3 & 100 & 100 & 73.3 & 73.3 & 100 & 100 \\
H2O         & 80.0 & 40.0 & 100 & 100 & 73.3 & 46.7 & 100 & 100 \\
StreamLLM   & 60.0 & 40.0 & 60.0 & 40.0 & 60.0 & 40.0 & 66.7 & 46.7 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{KV cache memory savings at CR=2$\times$ (compress\_kv pipeline, A100 80GB).  GQA models achieve 43--72\
[REF]
\small
\begin{tabular}{@{}llrrrr@{}}
\toprule
Model & Seq & Base KV & LB KV & Savings & Speedup \\
\midrule
\multirow{3}{*}{Llama-2-7B (MHA)}
& 512  & 287\,MB & 218\,MB & 24.3\
& 1024 & 574\,MB & 435\,MB & 24.3\
& 4096 & 2298\,MB & 1740\,MB & 24.3\
\midrule
\multirow{3}{*}{Llama-3.1-8B (GQA)}
& 512  & 189\,MB & 54\,MB & 71.2\
& 1024 & 379\,MB & 109\,MB & 71.2\
& 4096 & 1513\,MB & 435\,MB & 71.3\
\midrule
\multirow{3}{*}{Mistral-7B (GQA)}
& 512  & 95\,MB & 54\,MB & 42.9\
& 1024 & 191\,MB & 109\,MB & 43.1\
& 4096 & 762\,MB & 435\,MB & 42.9\
\bottomrule
\multicolumn{6}{l}{\scriptsize Speedup $<$1$\times$ due to compression overhead; amortized at higher batch sizes / longer generation.}
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Decode latency (TPOT, ms/token) at 1024 tokens, CR=2$\times$.
\layerbudget adds 3--5\
[REF]
\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & Baseline & \layerbudget & Overhead & KV Savings \\
\midrule
Llama-2-7B (MHA)    & 20.6 & 21.1 & +2.6\
Llama-3.1-8B (GQA)  & 21.4 & 22.0 & +2.8\
Mistral-7B (GQA)    & 20.4 & 21.1 & +3.9\
Qwen3-8B (GQA)      & 49.1 & 51.3 & +4.5\
Qwen2.5-14B (GQA)   & 82.7 & 92.8 & +12.2\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{GSM8K math reasoning accuracy (200 samples) at CR=4$\times$.
\layerbudget preserves Full KV accuracy; H2O collapses.}
[REF]
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Llama-2 & Llama-3.1 & Mistral & Qwen3 \\
\midrule
Full KV              & 5.5 & 18.0 & 19.0 & 62.0 \\
\layerbudget & 7.5 & 22.5 & 21.5 & 60.0 \\
KIVI                 & 9.0 & 22.0 & 17.0 & 54.5 \\
H2O                  & 0.0 & 2.0 & 1.5 & 2.5 \\
\bottomrule
\multicolumn{5}{l}{\scriptsize LB slightly exceeds Full KV due to chain-of-thought variance at small $n$.}
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Profiling overhead on 7B models (4-bit).
The ``Alloc'' column is the full per-input profiling cost: Gini extraction over all layers + greedy allocator inner loop + action selection.
The allocator's inner loop alone is $<$1\,ms (Section~[REF]); the bulk of ``Alloc'' is dominated by attention-row collection across $L$ layers and scales linearly with $S$.
For deployment, fixed profiles (calibrated once per model) reduce the per-input overhead to $<$3\
[REF]
\small
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Model & $S$ & Prefill & Attn OH & Gini & Alloc & Total \\
\midrule
\multirow{3}{*}{Llama}
& 256  & 163ms & +9.7\
& 512  & 241ms & $-$2.2\
& 1024 & 447ms & $-$0.6\
\midrule
\multirow{3}{*}{Mistral}
& 256  & 167ms & +8.6\
& 512  & 261ms & $-$5.4\
& 1024 & 476ms & $-$0.6\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{LongBench overall score (mean F1/ROUGE-L across 16 English tasks) at CR=2$\times$, 4096 tokens.  \layerbudget matches Full KV on all models.}
[REF]
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Method & Mistral-7B & Llama-3.1-8B & Llama-2-7B \\
\midrule
Full KV              & 0.056 & 0.040 & 0.050 \\
\layerbudget & 0.055 & 0.040 & 0.050 \\
KIVI                 & 0.053 & 0.041 & 0.048 \\
H2O                  & 0.056 & 0.038 & 0.097 \\
StreamingLLM         & 0.049 & 0.039 & 0.045 \\
\bottomrule
\multicolumn{4}{l}{\scriptsize H2O on Llama-2-7B anomalously high due to degenerate repetition inflating F1.}
\end{tabular}
\end{table}




\begin{table}[h]
\centering
\caption{Long-context PPL ratio at 16K and 32K tokens.  \layerbudget remains near-lossless while H2O collapses at high compression.}
[REF]
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
& & \multicolumn{2}{c}{16K tokens} & \multicolumn{2}{c}{32K tokens} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
Model & Method & 2$\times$ & 4$\times$ & 2$\times$ & 4$\times$ \\
\midrule
\multirow{4}{*}{Mistral-7B}
& \layerbudget & 1.001 & 1.006 & 1.000 & 1.009 \\
& KIVI          & 1.006 & 1.006 & 1.008 & 1.008 \\
& H2O           & 1.039 & 34.9  & 1.057 & 83.4 \\
& StreamingLLM  & 1.013 & 1.035 & 1.014 & 1.026 \\
\midrule
\multirow{4}{*}{Llama-3.1-8B}
& \layerbudget & 1.000 & 1.035 & 0.999 & 1.035 \\
& KIVI          & 1.033 & 1.033 & 1.035 & 1.035 \\
& H2O           & 1.013 & 1.024 & 1.059 & 1.066 \\
& StreamingLLM  & 1.030 & 1.039 & 1.038 & 1.043 \\
\midrule
\multirow{4}{*}{Qwen3-8B}
& \layerbudget & 1.000 & 1.005 & --- & --- \\
& KIVI          & 1.005 & 1.005 & --- & --- \\
& H2O           & 1.018 & 1.030 & --- & --- \\
& StreamingLLM  & 1.000 & 1.017 & --- & --- \\
\bottomrule
\multicolumn{6}{l}{\scriptsize Qwen3-8B 32K: OOM on RTX 3090 (36 layers). 16K data confirms the trend.}
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Large-model scaling: PPL ratio at 1024 tokens (4-bit weights, A100 80\,GB).
\layerbudget achieves $\leq$1.01 at 4$\times$ on GQA and 1.11 on MHA.
H2O collapses on the 13B MHA model (32$\times$ at 6$\times$).}
[REF]
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{Llama-2-13B (MHA, 40H)} & \multicolumn{4}{c}{Qwen2.5-14B (GQA, 4H)} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
Method & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ \\
\midrule
\layerbudget & 1.00 & 1.01 & 1.11 & 1.56 & 1.00 & 1.01 & 1.01 & 1.06 \\
KIVI        & 1.02 & 1.02 & 1.02 & 1.02 & 1.01 & 1.01 & 1.01 & 1.01 \\
H2O         & 1.23 & 1.80 & 11.1 & 32.7 & 1.06 & 1.10 & 1.15 & 1.24 \\
StreamLLM   & 1.20 & 1.24 & 1.33 & 3.90 & 1.15 & 1.20 & 1.26 & 1.34 \\
SnapKV      & 1.07 & 1.15 & 1.26 & 1.92 & 1.08 & 1.12 & 1.16 & 1.23 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{NIAH retrieval accuracy (\
\layerbudget and KIVI maintain 100\
[REF]
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Llama-2-13B} & \multicolumn{2}{c}{Qwen2.5-14B} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Method & 2$\times$ & 4$\times$ & 2$\times$ & 4$\times$ \\
\midrule
Full KV     & 100 & --- & 100 & --- \\
LB & 100 & 100 & 100 & 100 \\
KIVI        & 100 & 100 & 100 & 100 \\
H2O         & 86.7 & 30.0 & 83.3 & 63.3 \\
StreamLLM   & 60.0 & 36.7 & 60.0 & 40.0 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Qwen2.5-72B scaling validation (4-bit, GQA 8 heads, A100 80\,GB).
\layerbudget achieves ratio 1.004 at 4$\times$---near-lossless at 72B scale.}
[REF]
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{PPL ratio @1024} & \multicolumn{2}{c}{NIAH@4096} & \multicolumn{2}{c}{RULER@4096} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Method & 2$\times$ & 3$\times$ & 4$\times$ & 6$\times$ & 2$\times$ & 4$\times$ & 2$\times$ & 4$\times$ \\
\midrule
\layerbudget & 1.00 & 1.00 & 1.00 & 1.06 & 100 & 100 & 98 & 100 \\
KIVI        & 1.00 & 1.00 & 1.00 & 1.00 & 100 & 100 & 99 & 100 \\
H2O         & 1.05 & 1.10 & 1.15 & 1.21 & 100 & 73.3 & 93 & 50 \\
StreamLLM   & 1.16 & 1.21 & 1.26 & 1.34 & 70.0 & 50.0 & 62 & 50 \\
SnapKV      & 1.09 & 1.14 & 1.18 & 1.27 & --- & --- & --- & --- \\
MiniKV      & 1.01 & 1.01 & 1.01 & 1.03 & --- & --- & --- & --- \\
\bottomrule
\multicolumn{9}{l}{\scriptsize Full KV baselines: PPL=1.000, NIAH=100\
\end{tabular}
\end{table}

\input{checklist}

\end{document}

--- PAPER TEXT ENDS ---

Your reviewer assignment: R1 ("Methodologist").
Focus: Technical soundness, mathematical rigor, proof correctness
Bias: Strict on methodology, lenient on presentation

Scoring dimensions:
  - Soundness (1-4): Are the claims well-supported by evidence?
  - Presentation (1-4): Is the paper well-written and organized?
  - Contribution (1-4): How significant is the contribution?
  - Originality (1-4): How novel is the approach?
  - Clarity (1-4): Can the paper be easily understood?
  - Significance (1-4): Will this work influence future research?
Overall: 1-10. Confidence: 1-5.

Mandatory probes — answer every one concretely, with page/section/equation
references, before finalizing your scores:
  [P1-alternatives] Name >=3 existing systems, vendor APIs, or standard techniques that could plausibly address this paper's core problem. For each: does the paper compare against it head-to-head, argue why it is insufficient, or ignore it entirely? An ignored obvious alternative is a fatal weakness.
  [P2-cold-terms] List >=5 symbols or terms that are used before being defined, are overloaded (one symbol, multiple meanings), or are never defined at all. If you genuinely find fewer, list what you found and say so.
  [P3-recompute] Pick >=1 derived equation, bound, or headline number and recompute it from the paper's own stated inputs. Show the arithmetic. Report whether it checks out. Prefer the least-checked-looking derivation.
  [P4-strawman] Identify any headline number whose baseline is a non-default, misconfigured, or unrealistic setup. Is the most favorable-sounding number in the abstract/intro the honest one?
  [P5-consistency] Find claims that contradict other parts of the same paper: API contracts vs measurement tables, theorem constants vs prose claims, abstract guarantees vs stated limitations.
  [P6-provenance] Does the prose read as machine-generated (uniform rhythm, 'not X but Y' parallelisms, unexplained jargon density, sections that describe rather than argue)? Cite specific sentences.
  [P7-precision] Run four sweeps and quote offending sentences for each. (a) OVER-ASSERTION: empirical claims stated as universals that should be hedged — 'takes days' when the honest claim is 'can take days', 'means' when it is 'typically means'. (b) VAGUE REFERENT: 'this', 'it', 'these', 'the approach' where the antecedent is ambiguous or absent. (c) THING-VS-PROCESS CONFLATION: a noun used for both an artifact and the activity that produces it. (d) FALSE PRECISION: 'a sixth of them' where the data supports 'about 20%'. Do not report zero findings without quoting the sentences you checked.
  [P8-reverse-outline] Reconstruct the paper's skeleton without reading its section headings: state the thesis in one sentence, then extract the topic sentence of every paragraph. For each, answer: does it map to the thesis, and does the paragraph's evidence map to its own topic sentence? List every paragraph that maps to neither — those are candidates for deletion, not revision. A paragraph whose topic sentence you cannot identify is itself the finding.
  [P9-figure-integrity] Examine EVERY figure as rendered (the compiled PDF, not the LaTeX source). For each: are all blocks/arrows/labels correct and non-duplicated, does it match what the caption and text claim, and does it look machine-generated or internally inconsistent (orphan boxes, wrong connections, nonsense sub-labels)? A broken or AI-looking figure is a credibility defect — flag it as a weakness, not a nitpick. If you only have the text, say so and flag that the figures were unverifiable.
  [P10-policy-desk-reject] Audit submission-policy compliance independent of scientific merit: (a) ANONYMITY — any first-person disclosure of the authors' own venue, submission date, prior-report identity, or institution that de-anonymizes during blind review, or any aggressive priority claim; (b) COMPLETENESS — required checklist/declaration items (e.g. NeurIPS LLM-usage item) all present and answered; (c) FORMAT — page limit, template, references. Any hit here is a desk-reject risk and must be reported, even if the science is strong. Do NOT 'fix' a de-anon claim by making it more explicit — the fix is to remove it.
  [P11-method-soundness] Check theory<->implementation coherence, NOT just internal proof validity. For each central theoretical claim or method definition: (a) state in one line what the algorithm/code ACTUALLY computes (the loss, the advantage, the importance ratio's denominator, the distribution a KL/regularizer anchors to), from the paper's own algorithm box / equations; (b) confirm it equals the object the theory names — if the surrogate is interpreted as an anchor to distribution Q but the implementation uses a different reference, that is a defect; (c) flag any choice that silently changes the SEMANTICS of a borrowed method — e.g. normalizing a group-relative advantage over a mixed/replay batch (no longer group-relative), an IS denominator that is not the generation policy, or a 'CQL/AWR-style' claim whose implemented objective is not actually that. A mismatch between what is proved and what is run is a fatal weakness, not a clarification — report it as such.
  [P12-contribution-vs-ablation] Audit the contribution ordering against the paper's own ablations. List every contribution the abstract/introduction claims, in the order claimed. For EACH: name the specific experiment/table that isolates it, and state its marginal effect in the paper's own numbers (what is lost when it is removed or replaced by the trivial alternative). Then answer: does the ranking by measured marginal effect match the ranking the abstract implies? A contribution billed as primary whose isolating experiment shows a small or within-noise effect is a FATAL defect — the honest fix is to demote the claim, not to argue for the component. A contribution with NO isolating experiment at all is also a defect. Note this is distinct from P4: P4 asks whether the BASELINE is honest; P12 asks whether the CREDIT ASSIGNMENT among the paper's own parts is honest.
  [P13-venue-fit] Judge fit for THIS venue, separately from quality. (a) State the paper's contribution in ONE sentence using only this community's standard vocabulary; if you cannot, that failure IS the finding. (b) List terms central to the paper that a typical PC member of this venue would not use or would have to look up. (c) Name which core mechanism, if any, belongs to a different community's conference, and which venue you would forward this paper to instead. (d) Does the related work position against THIS venue's recent literature, or against a neighbouring field's? A well-executed paper written for the wrong room is rejected exactly like a weak one — report mis-venuing as a major weakness, not as a stylistic note.
  [P14-metric-completeness] Audit metric coverage against the field's conventions and the paper's own objective. (a) Enumerate the metrics a reader in this field expects for this problem; mark each REPORTED or OMITTED. (b) For every OMITTED one, state whether any headline claim depends on it — an omitted metric that could reverse a claim is a fatal defect. (c) Scan the paper's own objective / reward / loss for terms that are never measured anywhere in the evaluation (e.g. a quality term in a reward function when only latency and energy are reported) — an unmeasured term in your own objective is a defect regardless of the field's conventions. (d) Check whether the workload regime evaluated matches the one the claims are stated over.

Requirements: >=3 strengths, >=3 weaknesses, >=2 questions for the authors.
Weaknesses must be specific enough that the authors know exactly what to fix.

Additional reviewer context (competitor notes / prior reviews for calibration — treat as background, verify independently):
# Prior-art context supplied to reviewers (state of the field as of 2026-08-04)

This is the concurrent/prior work a well-informed reviewer of this paper would
know about. Use it for probe P1 (existing alternatives) and P13 (venue fit).
Nothing here is a finding about the paper — judge the paper yourself.

## Directly overlapping: per-layer joint eviction + quantization allocation

- **RDKV — "Rate-Distortion Bit Allocation for Joint Eviction and Quantization of
  the KV Cache"** (ETH Zürich + Tsinghua, arXiv 2605.08317, 8 May 2026).
  Casts KV cache compression as a rate–distortion problem in which *"eviction and
  quantization are two end-points of the same bit allocation scheme."* Derives each
  token's/channel's weight from the distortion compression induces on the attention
  computation, then assigns bit-widths from full precision down to zero bits by
  reverse water-filling with Lagrangian relaxation, applied once after prefill.
  Training-free. Explicitly argues the two actions must be *"explored jointly rather
  than in a staged fashion."* Instantiates the allocation as a discrete knapsack over
  hardware-supported bit-widths, and realizes the mixed-bit cache with **TriZone**, a
  packed-decode layout fused into the attention kernel.
  Results: outperforms the best evaluated baseline by 9.1% on average on LongBench,
  RULER and InfiniteBench; recovers **97.81% of full-cache accuracy at 2.48% cache
  retention** on LongBench; **4.5× decode speedup and 1.9× peak memory reduction at
  128K context** vs full-cache FlashAttention-2.

- **HqeKV — "Towards Hybrid Quantization and Eviction for KV Cache in Long-Context
  LLM Inference"** (ACL 2026 Findings; code public). Hybrid framework over both
  quantization and eviction with an integrated optimizer that selects the compression
  action per cached element, plus a joint K–V importance metric. Reports output
  quality 40.53 → 49.98 under the same memory constraint.

- **MoE-nD — "Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache
  Compression"** (arXiv 2604.17695, Apr 2026). Routes each layer to its own
  (eviction-ratio, K-bits, V-bits) tuple under a global memory budget via an
  offline-calibrated greedy solver. Matches an uncompressed 1.9 GB baseline at **14×
  compression** on a LongBench-v1 subset; +6 to +27 pts over the strongest per-layer
  quantization baseline on AIME.

- **ARKV.** Estimates per-layer original/quantization ratios from prefill-time
  attention statistics with scoring thresholds searched offline.

- **PolyKV — "Heterogeneous Retention and Allocation for KV Cache Compression"**
  (KAUST, arXiv 2606.15157, Jun 2026). Reformulates KV eviction as a *layer-wise
  design space* coupling two choices — which eviction pattern each layer uses and how
  much cache capacity it receives — under a shared memory constraint; offline
  calibration converts layer signals into a fixed heterogeneous strategy. On
  LongBench recovers 54.5% / 25.7% of the FullKV gap on LLaMA-3.1-8B / Qwen3-8B at a
  512-token average budget.

- **LKV** (arXiv 2605.06676, May 2026). End-to-end *learned* head-wise budgets and
  token selection.

## Top-venue 2026 results that set the current bar

- **STAR-KV** (ICML 2026 **Spotlight**). Adaptive low-rank compression with
  differentiable soft-thresholding rank control at head and block level, hybrid K/V
  decomposition, and low-rank-aware **mixed-precision quantization**. **20× full KV
  cache compression**, 6.9× faster attention, 3.1× generation throughput, custom GPU
  kernels.
- **KVTC — "KV Cache Transform Coding"** (ICLR 2026, NVIDIA). PCA decorrelation +
  adaptive quantization + entropy coding; bit widths assigned under a global bit
  budget to minimize reconstruction error. **20× (40×+ in some settings)** with
  reasoning and long-context accuracy retained, on AIME25, GSM8K, LiveCodeBench,
  LongBench, MATH-500, MMLU, Qasper, RULER.
- **DefensiveKV / Layer-DefensiveKV** (ICLR 2026). Defensive score aggregation; the
  Layer- variant adds AdaKV-style layer-wise budget allocation. RULER at 20% cache:
  SnapKV 39.0 → DefensiveKV 85.3 → Layer-DefensiveKV **91.4**.
- **TurboQuant** (ICLR 2026, Google). Near-optimal online vector quantization, 3-bit
  keys / 2-bit values, Triton kernels, vLLM integration.
- **"Adaptive KV-Cache Compression without Manually Setting Budget"** (ICLR 2026).
  Formalizes compression as scoring → allocation → selection with adaptive per-layer
  budgets; evaluates on GSM8K, RULER, **LongBench-v2**.
- **KVCompose** (2026). Composite tokens with a global allocation mechanism that
  adapts retention budgets across layers.

## Long-output / reasoning-model regime (now a major sub-field)

- **ThinKV** — thought-adaptive **hybrid quantization + eviction**: per-thought-type
  precision (8/4/2-bit, ~3.4 average bits) plus progressive eviction. Near-lossless
  at **<5% of the original KV cache**, up to 5.8× throughput, on DeepSeek-R1-Distill,
  GPT-OSS, QwQ-32B, AceReason, across AIME / MATH-500 / GSM8K / LiveCodeBench.
- **MixKVQ** — query-aware mixed-precision (BF16 / UINT4 / UINT2), 2.3–2.7 effective
  bits, on AIME'24–'25, MATH-500, GPQA-Diamond, LiveCodeBench.
- **InfoKV**, **VaSE**, **Adaptive Mass-Segmented KV**, **Reasoning Path
  Compression** — same regime.

## Evaluation-methodology results the community now expects papers to respect

- **"How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget
  Audit"** (arXiv 2607.11942). 144,300 paired RULER records, 3 models, bootstrap
  B=50,000. Under **query-aware** compression (the literature default) four published
  methods beat trivial baselines; under **query-agnostic** compression (compress the
  context, *then* append the question — the deployment order) only KeyDiff wins and
  **SnapKV falls below a "keep start + recent window" trivial baseline (−0.066)**.
  Method gaps track query leakage into the scoring signal (SnapKV Δ=+0.198 → KeyDiff
  Δ=+0.011). Swapping the attention backend **sdpa → eager shifts RULER accuracy by
  −0.221, larger than most method gaps.** RULER's nominal "8192" overflows Gemma-2's
  positional budget by 30%, silently zeroing 7 of 13 subtasks with no compression.
- **"Ablation, Statistical Inference, and Validation for KV-Cache Compression"**
  (arXiv 2607.09683). Demands synthetic diagnostic regimes, separation of algorithmic
  from implementation variance, and multi-dimensional error geometry over
  accuracy-only reporting.
- **NVIDIA KVPress** is the de-facto harness: 30+ methods behind one API, a public
  HuggingFace leaderboard, a RULER / InfiniteBench / Loogle CLI, and built-in
  `PerLayerCompressionPress` and `QuantizedCache` support.

## Standard benchmarks in this area as of 2026

LongBench-v2 (v1 now considered dated), RULER (4K→128K, 13 subtasks), InfiniteBench,
SCBench (the KV-cache-centric benchmark: cache generation / compression / retrieval /
loading, multi-turn shared context), HELMET, NIAH, and for reasoning models AIME
2024/2025, MATH-500, GPQA-Diamond, LiveCodeBench. Effective bits/token is a commonly
reported common currency alongside compression ratio.



--- VENUE SUBMISSION INSTRUCTIONS (verbatim) ---
NeurIPS 2026 Main Track — submission rules (verbatim quotes from
https://neurips.cc/Conferences/2026/MainTrackHandbook, retrieved 2026-08-04)

PAGE LIMIT
"The main text of a submitted paper is limited to nine content pages, including
all figures and tables."
"Additional pages containing references, optional technical appendices and
mandatory paper checklist do not count as content pages."
Camera-ready: authors receive one additional content page upon acceptance.
Maximum file size: 50MB.

ABSTRACT LENGTH
The handbook does not specify an abstract character or word limit.

FORMATTING / FONT
"you must format your submission using the LaTeX style file for that year"
Submissions "violating the NeurIPS style (e.g., by decreasing margins or font
sizes) or page limits may be desk rejected."

ANONYMITY (DOUBLE-BLIND)
"All submissions must be anonymized and may not contain any identifying
information that violate the double-blind reviewing policy."
Authors must avoid acknowledgments and identifying self-citations (use
"Smith et al." rather than "our"), and must ensure external links permit
anonymous browsing. Violations result in desk rejection.

PAPER CHECKLIST
"Authors are required to complete the paper checklist included in the paper
template." The checklist must be included in the submitted PDF.

DUAL SUBMISSION
"The reviewing process will treat any other archival submission by an
overlapping set of authors as prior work."
Prohibited: submitting similar papers to NeurIPS simultaneously, "thin slicing"
multiple papers to one conference, or submitting to another archival venue while
under NeurIPS review.

LLM USE
"The use of spell checkers and grammar suggestions, aid for editing purposes,
and basic code assistance does not need to be documented."
"use of agents and/or LLMs in implementing the method should be described" if
non-standard. Authors remain fully responsible for all content accuracy.

--------------------------------------------------------------------------------
CALIBRATION NOTE FOR REVIEWERS (added by the orchestrator, not part of the CFP):
The NeurIPS 2026 deadline (May 2026) has passed and this paper was never
submitted. It is being reviewed against the NeurIPS bar to decide whether to
target a top-tier venue (ICLR 2027 / MLSys 2027 / ASPLOS 2027) or a fast
second-tier venue. Judge it at full NeurIPS-main-track severity. For the
venue-fit probe (P13), treat the target community as top-tier ML-systems /
efficient-inference (NeurIPS, ICLR, MLSys).

--- END SUBMISSION INSTRUCTIONS ---
Check the paper against these rules for P10 (format/policy) and, if you are the Venue Reviewer, for P13 (venue fit). Report any numeric violation (abstract length, page count, font size, anonymity) as a desk-reject risk, not a nitpick.

--- OUTPUT CONTRACT ---
Return ONE JSON object and nothing else — no prose, no code fence.
It MUST validate against this JSON schema:

{
  "type": "object",
  "properties": {
    "reviewer": {
      "type": "string"
    },
    "persona": {
      "type": "string"
    },
    "summary": {
      "type": "string"
    },
    "strengths": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "weaknesses": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "questions": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "probe_answers": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "probe": {
            "type": "string"
          },
          "answer": {
            "type": "string"
          },
          "defect_found": {
            "type": "boolean"
          }
        },
        "required": [
          "probe",
          "answer",
          "defect_found"
        ],
        "additionalProperties": false
      }
    },
    "scores": {
      "type": "object",
      "properties": {
        "Soundness": {
          "type": "integer"
        },
        "Presentation": {
          "type": "integer"
        },
        "Contribution": {
          "type": "integer"
        },
        "Originality": {
          "type": "integer"
        },
        "Clarity": {
          "type": "integer"
        },
        "Significance": {
          "type": "integer"
        }
      },
      "required": [
        "Soundness",
        "Presentation",
        "Contribution",
        "Originality",
        "Clarity",
        "Significance"
      ],
      "additionalProperties": false
    },
    "overall": {
      "type": "integer"
    },
    "confidence": {
      "type": "integer"
    },
    "decision": {
      "type": "string",
      "enum": [
        "Reject",
        "Weak Reject",
        "Weak Accept",
        "Accept"
      ]
    }
  },
  "required": [
    "reviewer",
    "persona",
    "summary",
    "strengths",
    "weaknesses",
    "questions",
    "probe_answers",
    "scores",
    "overall",
    "confidence",
    "decision"
  ],
  "additionalProperties": false
}

Set "reviewer" to "R1" and "persona" to "Methodologist". Every mandatory probe must appear in probe_answers with a concrete answer and an honest defect_found flag; answering a probe with an empty or evasive string is a protocol violation. Judge only what is on the page.