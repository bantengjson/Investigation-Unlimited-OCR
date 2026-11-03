<PAGE>
<BLOCK>header[117, 40, 396, 72]<FORK>
<BLOCK>title<LEVEL1>[348, 135, 653, 156]<FORK>Unlimited OCR Works 2
<BLOCK>text[201, 162, 797, 181]<FORK>Pioneering the OCR-3.0 Era: From Long- to Unlimited-Horizon Parsing
<BLOCK>text[418, 213, 581, 229]<FORK>Baidu ERNIE Team
<BLOCK>title<LEVEL2>[448, 274, 552, 292]<FORK>Abstract
<BLOCK>text[112, 298, 888, 572]<FORK>Unlimited OCR ushers the era of long-horizon parsing, substantially mitigating the growth of working memory as document lengthens. However, as document length scales to hundreds of pages, the visual context grows substantially, sharply increasing the computational cost of each decoding step and limiting the scalability of ultra-long document parsing. Inspired by human book transcription via localized focus without re-scanning preceding pages, we introduce Unlimited OCR 2, a document parsing model that looks only at what it transcribes. Building on Unlimited OCR, we upgrade all the decoder attention modules to Sparse Reference Sliding Window Attention (SR-SWA), bounding active reference tokens to a fixed capacity and thus sustaining constant computational costs across arbitrarily long documents. By combining SR-SWA with Unlimited-Horizon Decoding Paradigm, Unlimited OCR 2 enables document parsing over virtually unlimited lengths without prohibitive memory overhead. At the hundred-page scale, Unlimited OCR 2 achieves 7.0× and 5.6× per-step speedups over Full Attention and R-SWA, respectively. Furthermore, it delivers state-of-the-art end-to-end performance on both MPDocBench and MarathonDocBench, our newly proposed benchmark targeting hundred-page document parsing. Ultimately, the capacity for uninterrupted, one-shot document parsing signifies a fundamental paradigm shift into the OCR 3.0 era.
<BLOCK>image[202, 577, 469, 798]<FORK>
<BLOCK>image[510, 577, 798, 798]<FORK>
<BLOCK>image_caption[114, 809, 885, 859]<FORK>Figure 1 | R-SWA vs SR-SWA. R-SWA computes attention over all visual tokens across pages, inducing extreme visual redundancy; SR-SWA sparsely attends to relevant visual regions, aligning machine decoding with human reading intuition.
<PAGE>
<BLOCK>title<LEVEL2>[115, 100, 212, 118]<FORK>Contents
<BLOCK>text[112, 139, 887, 815]<FORK>1 Introduction 3
2 Related Works 4
2.1 VLM-based Document Parsing 4
2.2 Efficient Long-Context Modeling 4
3 Methodology 4
3.1 Problem Setup 4
3.1.1 Unlimited-horizon Parsing 4
3.1.2 MarathonDocBench 5
3.2 Architecture 5
3.3 Sparse Reference Sliding Window Attention 6
3.3.1 Page Gate 6
3.3.2 Token Sieve 6
3.4 Unlimited-Horizon Decoding Paradigm 7
4 Experimental Settings 8
4.1 Data Engine 8
4.2 Implementation Details 8
5 Evaluation 9
5.1 Benchmarks and Metrics 9
5.2 Main Results 10
5.3 Scaling with Document Length 11
5.3.1 Feasibility of Unlimited-horizon Parsing 11
5.3.2 Length-Invariant Accuracy 11
5.3.3 Constant Per-Step Decoding Latency 12
6 Limitation and Future Work 12
7 Conclusion 12
8 Author List 13
<BLOCK>page_number[493, 923, 506, 936]<FORK>2
<PAGE>
<BLOCK>title<LEVEL2>[116, 101, 274, 117]<FORK>1. Introduction
<BLOCK>text[160, 132, 837, 164]<FORK>“Pixels are my body, and tokens are my blood. I have parsed over a thousand pages. Unknown to KV cache. Nor known to context limits. So, as I pray: Unlimited OCR Works.”
<BLOCK>text[528, 169, 839, 183]<FORK>Snarcv on huggingface (Adapted from UBW)
<BLOCK>text[113, 188, 885, 382]<FORK>Document parsing recessing has undergone a major paradigm shift, transitioning from traditional multi-stage pipelines (OCR-1.0) [5, 6, 8, 20, 22, 26, 40] with handcrafted detection and recognition modules to end-to-end Vision-Language Models (OCR-2.0) [7, 18, 19, 21, 30, 31]. By formulating document understanding as a unified sequence-to-sequence transcription task, OCR-2.0 frameworks effectively eliminate cumulative pipeline errors. Our previous work, Unlimited OCR [37], pushed OCR-2.0 boundaries to tens of pages by using Reference Sliding Window Attention (R-SWA) to maintain a constant KV cache. Nevertheless, it still computes attention over all visual tokens across pages during decoding. As document length scales, this persistent attention to non-informative visual tokens severely escalates visual redundancy, compounding the difficulty of end-to-end inference. This visual attention bottleneck naturally prompts a fundamental question: how can we transition from length-limited parsing to an unlimited one-shot paradigm and genuinely inaugurate the OCR-3.0 era?
<BLOCK>text[113, 389, 885, 567]<FORK>To answer this question, human transcription of lengthy books provides a clear physical intuition for parsing scaling: as we copy text page by page, we maintain localized visual focus, casting only momentary glances back when cross-page references are needed, rather than keeping the entire volume active in mind. This cognitive mechanism reveals that visual dependencies in multi-page documents are inherently page-sparse yet intra-page concentrated. However, standard dense global attention completely overlooks this sparse dynamic, incurring redundant attention interactions across all pages. This not only incurs prohibitive computational overhead, but also leads to severe memory bottlenecks as document length scales. To break through these efficiency and memory bottlenecks, we align machine perception with human reading patterns and present Unlimited OCR 2, pioneering an unlimited-horizon parsing framework for the OCR-3.0 era. Our main contributions are as follows:
<BLOCK>text[139, 574, 885, 670]<FORK>- We introduce Sparse Reference Sliding Window Attention (SR-SWA), a hierarchical sparse attention mechanism designed for unlimited visual context. Specifically, Page Gate first narrows each query's visual reference scope from the entire document to relevant pages while robustly modeling multi-page positional relationships. Token Sieve selectively incorporates top-k remote token buckets to capture long-range visual dependencies, thereby drastically reducing attention overhead while preserving comprehensive context.
<BLOCK>text[139, 671, 884, 735]<FORK>- We introduce MarathonDocBench, a benchmark for hundred-page document parsing. It substantially extends existing evaluations in document length and cross-page structural complexity, enabling a comprehensive assessment of long-document parsing performance in the hundred-page regime.
<BLOCK>text[139, 735, 886, 833]<FORK>- Comprehensive evaluations in long-horizon settings demonstrate that our approach maintains constant per-step decoding latency regardless of document length while achieving state-of-the-art end-to-end performance on multi-page benchmarks. Building on these advances, we establish an efficient unlimited-horizon parsing paradigm by integrating SR-SWA with the Unlimited-Horizon Decoding Paradigm, enabling document parsing over virtually unlimited lengths.
<BLOCK>text[114, 839, 886, 904]<FORK>In summary, SR-SWA fundamentally breaks the computational and memory bottlenecks of ultra-long visual context. Operating at constant per-step decoding latency with length-invariant accuracy across hundreds of pages, Unlimited OCR 2 unlocks scalable long-document parsing and paves the way for the OCR-3.0 era of unlimited-horizon understanding.
<BLOCK>page_number[493, 924, 506, 936]<FORK>3
<PAGE>
<BLOCK>title<LEVEL2>[114, 101, 295, 117]<FORK>2. Related Works
<BLOCK>title<LEVEL3>[114, 133, 418, 149]<FORK>2.1. VLM-based Document Parsing
<BLOCK>text[113, 160, 886, 355]<FORK>Document parsing recessing has increasingly been reshaped by advances in vision-language models, evolving from pipeline-based OCR-1.0 systems [5, 6, 8, 20, 22, 26, 40], which combine specialized page analysis modules with VLM-based decoding, to unified VLM-based OCR-2.0 systems that directly generate structured representations from document images. OCR-2.0 encompasses both single-page parsers [7, 18, 19, 21, 30, 31] and Unlimited OCR [37], which extends end-to-end parsing to multi-page documents by jointly modeling their visual content within a shared context. However, although Unlimited OCR improves long-document efficiency with R-SWA, it still densely attends to all remote visual tokens, causing the visual context accessible to each decoding query to expand with document length and creating a fundamental scalability bottleneck at the hundred-page scale. Unlimited OCR 2 overcomes this limitation by explicitly bounding remote visual access, enabling attention-efficient end-to-end parsing of hundred-page documents and advancing document parsing into the OCR-3.0 era.
<BLOCK>title<LEVEL3>[114, 375, 434, 391]<FORK>2.2. Efficient Long-Context Modeling
<BLOCK>text[113, 401, 886, 484]<FORK>Efficient long-context modeling has been explored through both attention replacement and sparse attention. Linear or recurrent alternatives [11, 15, 17, 35, 36] reduce long-context cost by summarizing historical information into compact states. Document parsing, however, requires long-range memory to preserve fine-grained visual evidence for accurate recognition and cross-page retrieval, which may not be fully retained within such compact representations.
<BLOCK>text[113, 490, 887, 604]<FORK>Sparse attention retains direct access to selected historical tokens or KV blocks while avoiding dense all-to-all computation. Early methods use predefined patterns such as local windows and global tokens [2, 38], whereas recent approaches adopt query-dependent retrieval. Hard-selection methods, such as Compressed Sparse Attention(CSA) [33] and MiniMax Sparse Attention(MSA) [16], rely on discrete top-k routing and typically require auxiliary supervision or warmup. In contrast, DashAttention [13] and HiLS [12] incorporate differentiable or forward-integrated routing scores, enabling more direct end-to-end optimization.
<BLOCK>title<LEVEL2>[114, 625, 283, 643]<FORK>3. Methodology
<BLOCK>title<LEVEL3>[114, 658, 284, 674]<FORK>3.1. Problem Setup
<BLOCK>title<LEVEL4>[115, 685, 394, 701]<FORK>3.1.1. Unlimited-horizon Parsing
<BLOCK>text[114, 711, 885, 792]<FORK>Unlimited-horizon parsing aims to enable end-to-end processing of documents of arbitrary lengths, from single-page forms to massive multi-page documents, with bounded computational and memory complexities. Formally, multi-page document parsing generates a transcription  $ \mathcal{Y} = (y_1, \ldots, y_L) $ from an ordered document  $ \mathcal{D} = (P_1, \ldots, P_N) $ with visual tokens  $ \mathcal{V}_{1:N} $. At decoding step  $ t $, the model operates on:
<BLOCK>equation[375, 799, 621, 815]<FORK> $$ \mathcal{X}_{t}=\mathrm{C o n c a t}\left(\mathcal{V}_{1},\ldots,\mathcal{V}_{N},y_{<t}\right). $$ 
<BLOCK>equation[860, 800, 883, 815]<FORK>(1)
<BLOCK>text[114, 823, 886, 904]<FORK>While Unlimited OCR constrained text history to a fixed sliding window via R-SWA, it still requires attending over all preceding visual tokens  $ \mathcal{V}_{1:N} $. In the hundred-page regime, this page-dependent visual expansion becomes the primary bottleneck. Decoupling visual context from page count N to maintain a bounded computational and memory footprint directly motivates our proposed Sparse Reference Sliding Window Attention (SR-SWA).
<BLOCK>page_number[493, 925, 506, 936]<FORK>4
<PAGE>
<BLOCK>chart[127, 106, 868, 272]<FORK>
<BLOCK>image_caption[114, 282, 884, 331]<FORK>Figure 2 | MarathonDocBench versus MPDocBench. MarathonDocBench extends document length from tens to hundreds of pages (a), with substantially richer cross-page structures as shown in (b) and (c).
<BLOCK>title<LEVEL4>[115, 350, 338, 365]<FORK>3.1.2. MarathonDocBench
<BLOCK>text[113, 376, 885, 475]<FORK>Existing multi-page document parsing benchmarks neither extend to the hundred-page scale nor sufficiently capture the rich cross-page structures that emerge in long documents. To address this gap, we introduce MarathonDocBench, the first benchmark specifically designed for hundred-page document parsing. It contains 100 documents spanning 103–300 pages with 205k annotated layout elements, extending the document-length range by roughly an order of magnitude beyond MPDocBench [41].
<BLOCK>text[113, 481, 886, 644]<FORK>Beyond document length, MarathonDocBench contains substantially richer cross-page structures than MPDocBench, with cross-page text and tables appearing much more frequently and densely across documents, as summarized in Figure 2. This makes MarathonDocBench a substantially more challenging testbed for evaluating whether end-to-end parsers can preserve document-level structure while scaling from tens to hundreds of pages. To ensure reliable evaluation, structural annotations, including bounding boxes, block type, hierarchy, reading order, and cross-page continuations of text and tables, are initialized from a strong page-by-page parser and carefully refined by human annotators. Content transcriptions are initialized from the same parser outputs and verified through an iterative VLM-based refinement pipeline, with flagged disagreements adjudicated by human annotators.
<BLOCK>title<LEVEL3>[115, 665, 263, 680]<FORK>3.2. Architecture
<BLOCK>text[113, 691, 887, 854]<FORK>Unlimited OCR 2 inherits the fundamental backbone from Unlimited OCR, leveraging DeepSeek OCR [28] as its foundation. Specifically, it employs the DeepEncoder paired with a Mixture-of-Experts (MoE) decoder architecture featuring 3B total parameters and 500M activated parameters per token. Building upon this base, we replace the baseline R-SWA with our proposed SR-SWA as shown in Figure 3. While R-SWA bounds local interactions, its visual context scope still expands with growing document length, leading to noticeable latency accumulation over ultralong horizons. By introducing a selective remote visual routing mechanism, SR-SWA strictly bounds the active context scope, enabling Unlimited OCR 2 to maintain a constant KV cache footprint across both text and visual modalities during unlimited-horizon decoding. We detail the inner mechanics of SR-SWA in the following subsections.
<BLOCK>page_number[493, 923, 507, 936]<FORK>5
<PAGE>
<BLOCK>image[128, 102, 872, 322]<FORK>
<BLOCK>image_caption[114, 334, 885, 400]<FORK>Figure 3 | Inspired by the process of humans copying books, we propose Unlimited OCR 2. SR-SWA combines Page Gate and Token Sieve to construct a fixed-length set of working reference visual tokens, substantially reducing computational and memory overhead while keeping both independent of document length throughout generation.
<BLOCK>title<LEVEL3>[114, 423, 533, 440]<FORK>3.3. Sparse Reference Sliding Window Attention
<BLOCK>text[114, 450, 884, 514]<FORK>Following standard block-sparse attention [13, 16, 32, 33], we partition  $ \mathcal{X}_i $ into fixed-size blocks  $ \{\mathcal{B}_b\}_{b=1}^B $ and preserve a fixed local window  $ \mathcal{L}(q) $ for each query  $ q $. SR-SWA constructs  $ S(q) $ hierarchically, using Page Gate for page-level organization and Token Sieve for fixed-budget selection.
<BLOCK>title<LEVEL4>[114, 534, 260, 550]<FORK>3.3.1. Page Gate
<BLOCK>text[113, 560, 885, 610]<FORK>To eliminate irrelevant visual tokens and concentrate focus on the current page, Page Gate restricts the visual attention scope for a query q associated with page  $ g(q) $ by retaining only blocks from designated target pages:
<BLOCK>equation[372, 620, 625, 639]<FORK> $$ \mathcal{C}_{\mathrm{p a g e}}(q)=\left\{b\mid\mathcal{B}_{b}\cap\mathcal{V}_{g(q)}\neq\emptyset\right\}. $$ 
<BLOCK>equation[860, 622, 883, 637]<FORK>(2)
<BLOCK>text[113, 649, 885, 714]<FORK>Since block boundaries are independent of page boundaries, a candidate block may straddle adjacent pages. Meanwhile, to prevent positional explosion as page count scales, Page Gate factorizes each token position into a bounded page-level coordinate (following an alternating 0–1 pattern) and a monotonic intra-page coordinate:
<BLOCK>equation[412, 724, 584, 750]<FORK> $$ \mathbf{p}_{i,j}=\left[\mathbf{p}_{i}^{\mathrm{p a g e}},\mathbf{p}_{i,j}^{\mathrm{i n t r a}}\right], $$ 
<BLOCK>equation[860, 730, 883, 745]<FORK>(3)
<BLOCK>text[114, 760, 772, 777]<FORK>ensuring both stable length extrapolation and efficient cross-page visual routing.
<BLOCK>title<LEVEL4>[115, 796, 268, 812]<FORK>3.3.2. Token Sieve
<BLOCK>text[113, 823, 885, 904]<FORK>While Page Gate restricts the routing space across pages, efficient fixed-budget routing requires a principled block selection strategy. Conventional sparse routers typically rely on dense-attention distillation, which lacks direct alignment with the language modeling objective. Inspired by recent end-to-end routing designs [12, 13], Token Sieve directly optimizes fixed-budget block selection under the language modeling objective.
<BLOCK>page_number[493, 925, 507, 936]<FORK>6
<PAGE>
<BLOCK>text[113, 103, 886, 152]<FORK>Given the candidate set  $ C_{\text{page}}(q) $ from Page Gate, Token Sieve summarizes each candidate block b into a compact key representation  $ \mathbf{k}_b $ and computes its relevance to query q. The router then forms a sparse routing distribution and retains only the top-k blocks:
<BLOCK>equation[280, 161, 716, 216]<FORK> $$ \begin{aligned}&s_{b}(q)=\frac{\tilde{\mathbf{q}}^{\top}\tilde{\mathbf{k}}_{b}}{\sqrt{d}},\qquad\pi(q)=\alpha-entmax\left(\{s_{b}(q)\}_{b\in C_{page}(q)}\right),\\&S(q)=TopK\left(\pi(q),k\right).\\ \end{aligned} $$ 
<BLOCK>equation[860, 181, 883, 196]<FORK>(4)
<BLOCK>text[113, 225, 885, 290]<FORK>Here, d denotes the hidden dimension. By selectively attending to a fixed number of top-k remote buckets, Token Sieve captures long-range visual dependencies with minimal computational overhead. To preserve relative preferences among selected blocks, the routing probabilities are converted into a mean-centered log-prior:
<BLOCK>equation[270, 298, 728, 336]<FORK> $$ \beta_{b}(q)=\log\pi_{b}(q)-\frac{1}{|\mathcal{S}(q)|}\sum_{b^{\prime}\in\mathcal{S}(q)}\log\pi_{b^{\prime}}(q),\qquad b\in\mathcal{S}(q). $$ 
<BLOCK>equation[860, 307, 883, 322]<FORK>(5)
<BLOCK>text[115, 353, 884, 386]<FORK>With the selected remote blocks  $ \mathcal{S}(q) $ and the fixed local window  $ \mathcal{L}(q) $, each query attends to a length-agnostic retained token set defined as:
<BLOCK>equation[400, 394, 598, 428]<FORK> $$ \mathcal{R}(q)=\mathcal{L}(q)\cup\bigcup_{b\in\mathcal{S}(q)}\mathcal{B}_{b}. $$ 
<BLOCK>equation[860, 399, 883, 414]<FORK>(6)
<BLOCK>text[115, 431, 528, 448]<FORK>The attention weights over  $ \mathcal{R}(q) $ are computed as:
<BLOCK>equation[303, 456, 694, 492]<FORK> $$ \alpha_{j}(q)=\operatorname{S o f t m a x}\left(\frac{q^{\top}k_{j}}{\sqrt{d}}+\mathbb{I}[b(j)\in\mathcal{S}(q)]\beta_{b(j)}(q)\right), $$ 
<BLOCK>equation[860, 467, 883, 482]<FORK>(7)
<BLOCK>text[112, 500, 885, 597]<FORK>where the softmax is normalized over all tokens in  $ \mathcal{R}(q) $,  $ b(j) $ denotes the block containing token j, and  $ k_j $ denotes its key projection. Tokens within the local window use standard attention logits, while tokens from selected remote blocks additionally incorporate the corresponding routing bias. Since the routing bias directly participates in these logits, gradients from the parsing objective can propagate to the routing module, enabling end-to-end optimization without an auxiliary KL-divergence distillation loss.
<BLOCK>algorithm[115, 603, 884, 836]<FORK>Algorithm 1 Cross-Chunk KV Reuse for Unlimited-Horizon Decoding

Require: Pages $P_1, \ldots, P_N$; instruction $q$; chunk size $W$; tail length $T$

Ensure: Transcription $Y$

Notation: $C$ is a KV cache; $\oplus$ concatenates KV in order;

ENC($X \mid C$) returns the KV of input $X$ computed with $C$ as context.

1: Partition $N$ pages into $K$ overlapping chunks $C_1, \ldots, C_K$ with a 1-page boundary overlap $\triangleright$

$K=1$ indicates unsegmented

2: $C \leftarrow \text{ENC}(C_1 \oplus q)$

$\triangleright$ only the first chunk carries $q$

3: $y_1 \leftarrow \text{DECODE}(C)$; $Y \leftarrow y_1$

4: for $k = 2$ to $K$ do

5: $C \leftarrow C_{\text{overlap}}^{\text{img}} \oplus \text{ENC}(C_k \mid C_{\text{overlap}}^{\text{img}}) \oplus C_{\text{tail}}^{\text{txt}}$

6: $y_k \leftarrow \text{DECODE}(C)$; $Y \leftarrow Y \oplus y_k$

7: end for

8: return $Y$
<BLOCK>title<LEVEL3>[114, 844, 490, 861]<FORK>3.4. Unlimited-Horizon Decoding Paradigm
<BLOCK>cross_text[112, 871, 886, 904]<FORK>While SR-SWA bounds the number of reference tokens accessed by each query, processing extremely long multi-page documents remains constrained by the GPU memory required to
<BLOCK>page_number[493, 924, 506, 936]<FORK>7
<PAGE>
<BLOCK>image[126, 105, 882, 315]<FORK>
<BLOCK>image_caption[112, 321, 885, 419]<FORK>Figure 4 | Comparison of (a) standard decoding and (b) our Unlimited-Horizon Decoding paradigm. Standard decoding retains visual KV caches for all pages, causing GPU memory usage to grow linearly with document length and eventually overflow. Our paradigm divides long documents into manageable chunks and processes them sequentially, reusing the KV cache of shared boundary pages and releasing inactive caches to keep GPU memory usage bounded while preserving cross-chunk continuity.
<BLOCK>append_text[112, 423, 886, 570]<FORK>prefill and retain their visual KV caches. As illustrated in Figure 4, conventional all-at-once prefill keeps all page-level visual KV caches in memory, causing the memory footprint to grow linearly with document length. To overcome this limitation, we introduce the Unlimited-Horizon Decoding Paradigm, which partitions the document into overlapping page-level chunks and processes them sequentially. Adjacent chunks share their boundary page, whose KV cache is directly reused to preserve cross-chunk continuity, while visual KV caches outside the active chunk are released. Consequently, only a fixed-size chunk remains resident in GPU memory, bounding the memory footprint independently of the total document length. The complete procedure is formally described in Algorithm 1.
<BLOCK>title<LEVEL2>[114, 583, 368, 602]<FORK>4. Experimental Settings
<BLOCK>title<LEVEL3>[114, 616, 262, 632]<FORK>4.1. Data Engine
<BLOCK>text[112, 642, 887, 805]<FORK>To train Unlimited OCR 2 for versatile document parsing, we construct a large-scale multimodal corpus comprising 2M multi-page and 10M single-page documents. Pseudo-labels across both single- and multi-page subsets are extracted from MinerU2.5-Pro [26] and PaddleOCR-VLv1.6 [40], tailored to their respective domain strengths. To enrich visual and layout diversity, we additionally incorporate versatile synthetic document generation pipelines. Across both real and synthetic data, we leverage VLM-based multi-dimensional feature analysis to conduct balance filtering, ensuring uniform coverage over complex layout distributions. Finally, to explicitly fortify structural hierarchy and cross-page continuity in multi-page scenarios, we apply MinerU-Popo [34] refinement alongside dedicated heuristic post-processing rules to reconcile layout inconsistencies across adjacent page boundaries.
<BLOCK>title<LEVEL3>[114, 827, 359, 843]<FORK>4.2. Implementation Details
<BLOCK>text[113, 853, 886, 902]<FORK>We initialize Unlimited OCR 2 from the Unlimited OCR checkpoint and perform post-training for 1 epoch on the curated dataset. Training is conducted on a cluster of  $ 25 \times 8 $ GPU NVIDIA H800 nodes with a global batch size of 1,600 and a maximum sequence length of 64K tokens.
<BLOCK>page_number[493, 924, 506, 936]<FORK>8
<PAGE>
<BLOCK>text[113, 102, 886, 185]<FORK>During training, we freeze both the DeepEncoder and the aligner, optimizing only the LLM parameters. Optimization is carried out using the AdamW optimizer with a cosine learning rate decay schedule, starting at an initial learning rate of  $ 2.5 \times 10^{-4} $. The entire training workflow is implemented using the MS-Swift framework. For deployment, the Unlimited-Horizon Decoding Paradigm is integrated into SGLang to facilitate efficient long-document inference.
<BLOCK>title<LEVEL2>[115, 208, 257, 225]<FORK>5. Evaluation
<BLOCK>title<LEVEL3>[115, 241, 367, 257]<FORK>5.1. Benchmarks and Metrics
<BLOCK>text[113, 268, 886, 414]<FORK>To comprehensively evaluate Unlimited OCR 2 on document parsing across varying lengths, we conduct experiments on two challenging multi-page benchmarks: MPDocBench [41] (up to 30 pages) and our proposed MarathonDocBench (ranging from 100 to 300 pages). Following the official evaluation protocol of MPDocBench, we report Edit Distance for textual content and reading order recognition, TEDS for table structure recognition, CDM for mathematical formula recognition, and F1 score for figure region detection. To further evaluate cross-page consistency, we include T-TextEdit and T-TEDS to assess text and table parsing across page boundaries, respectively. An overall score aggregated across these fine-grained structural categories is also computed to summarize overall parsing performance.
<BLOCK>image[123, 437, 872, 838]<FORK>
<BLOCK>image_caption[114, 850, 885, 883]<FORK>Figure 5 | Performance comparison on multi-page benchmarks. Unlimited OCR 2 (Ours) consistently achieves state-of-the-art performance across all evaluation categories.
<BLOCK>page_number[493, 924, 506, 935]<FORK>9
<PAGE>
<BLOCK>image_caption[114, 100, 884, 133]<FORK>Table 1 | Performance comparison on MPDocBench. All models in the table are end-to-end VLM-based architectures.
<BLOCK>table[118, 140, 882, 407]<FORK><table><tr><td>Method</td><td>Overall $ ^{\dagger} $</td><td>Text $ ^{Edit} $  $ \downarrow $</td><td>T-Formal $ ^{CM} $  $ \uparrow $</td><td>Table $ ^{TIES} $  $ \uparrow $</td><td>T-Table $ ^{TIES} $  $ \uparrow $</td><td>Figure $ ^{H} $  $ \uparrow $</td><td>ReadOrder $ ^{Edit} $  $ \downarrow $</td><td>Heading $ ^{TIES} $  $ \uparrow $</td></tr><tr><td colspan="9">General VLMs</td></tr><tr><td>Gemini-3.1-Pro Preview [9]</td><td>71.79</td><td>0.070</td><td>0.232</td><td>88.37</td><td>81.68</td><td>61.26</td><td>59.08</td><td>0.127</td></tr><tr><td>Qwen3.8-Max [25]</td><td>78.30</td><td>0.057</td><td>0.196</td><td>89.24</td><td>85.89</td><td>67.18</td><td>70.19</td><td>0.097</td></tr><tr><td>Kimi-K3 [24]</td><td>60.77</td><td>0.162</td><td>0.551</td><td>41.16</td><td>75.45</td><td>81.57</td><td>40.47</td><td>0.206</td></tr><tr><td>Qwen3.6-plus [25]</td><td>72.14</td><td>0.096</td><td>0.245</td><td>88.77</td><td>82.99</td><td>60.50</td><td>64.80</td><td>0.179</td></tr><tr><td>Qwen3-VL-235B [1]</td><td>73.98</td><td>0.087</td><td>0.187</td><td>84.71</td><td>81.35</td><td>61.25</td><td>63.25</td><td>0.136</td></tr><tr><td>InternVL-3.5-38B [27]</td><td>57.01</td><td>0.133</td><td>0.515</td><td>84.30</td><td>69.86</td><td>51.69</td><td>7.97</td><td>0.199</td></tr><tr><td>Minimax-M3 [16]</td><td>62.65</td><td>0.167</td><td>0.422</td><td>81.30</td><td>77.91</td><td>76.51</td><td>7.18</td><td>0.231</td></tr><tr><td>Gemini-3.8-Flash [10]</td><td>76.25</td><td>0.081</td><td>0.122</td><td>83.27</td><td>83.15</td><td>87.41</td><td>62.86</td><td>0.294</td></tr><tr><td>Qwen-3.8-Flash-Next [25]</td><td>77.36</td><td>0.078</td><td>0.163</td><td>86.70</td><td>86.24</td><td>88.36</td><td>64.31</td><td>0.277</td></tr><tr><td>DeepSeek-4.1-Flash [32]</td><td>66.25</td><td>0.114</td><td>0.451</td><td>82.12</td><td>80.14</td><td>87.72</td><td>35.55</td><td>0.312</td></tr><tr><td>GLM-5.3-Flash [39]</td><td>68.02</td><td>0.111</td><td>0.344</td><td>81.86</td><td>83.70</td><td>70.71</td><td>43.09</td><td>0.288</td></tr><tr><td colspan="9">Specialized VLMs (OCR-2.0)</td></tr><tr><td>Unlimited-OCR [37]</td><td>61.89</td><td>0.167</td><td>0.605</td><td>73.71</td><td>73.51</td><td>60.76</td><td>53.19</td><td>0.204</td></tr><tr><td colspan="9">Specialized VLMs (OCR-3.0)</td></tr><tr><td>Unlimited-OCR 2 (Ours)</td><td>84.59</td><td>0.052</td><td>0.100</td><td>89.39</td><td>87.78</td><td>89.54</td><td>72.91</td><td>0.077</td></tr></table>
<BLOCK>image_caption[114, 416, 884, 448]<FORK>Table 2 | Performance comparison on MarathonDocBench. Unlimited OCR is not reported on MarathonDocBench due to its 40-page input limitation.
<BLOCK>table[117, 454, 882, 644]<FORK><table><tr><td>Method</td><td>Overall $ ^{\dagger} $</td><td>Text $ ^{dd} $  $ ^{\ddagger} $</td><td>T-Text $ ^{dd} $  $ ^{\parallel} $</td><td>Formula $ ^{CM} $  $ ^{\dagger} $</td><td>Table $ ^{TEDS} $  $ ^{\dagger} $</td><td>T-Table $ ^{TEDS} $  $ ^{\dagger} $</td><td>Figure $ ^{3} $  $ ^{\dagger} $</td><td>ReadOrede $ ^{dd} $  $ ^{\ddagger} $</td><td>Heading $ ^{TEDS} $  $ ^{\dagger} $</td></tr><tr><td colspan="10">General VLMs</td></tr><tr><td>Gemini-3.1-Pro Preview [9]</td><td>26.97</td><td>0.516</td><td>0.764</td><td>30.44</td><td>28.48</td><td>21.71</td><td>18.94</td><td>0.654</td><td>9.68</td></tr><tr><td>Qwen3.8-Max [25]</td><td>22.93</td><td>0.600</td><td>0.804</td><td>18.82</td><td>15.20</td><td>13.71</td><td>33.39</td><td>0.733</td><td>16.09</td></tr><tr><td>Kimi-K3 [24]</td><td>33.74</td><td>0.475</td><td>0.664</td><td>40.50</td><td>27.80</td><td>26.30</td><td>21.38</td><td>0.611</td><td>28.90</td></tr><tr><td>Minimax-M3 [16]</td><td>17.97</td><td>0.659</td><td>0.818</td><td>14.20</td><td>17.79</td><td>14.84</td><td>3.67</td><td>0.763</td><td>17.16</td></tr><tr><td>Gemini-3.8-Flash [10]</td><td>28.61</td><td>0.519</td><td>0.658</td><td>26.35</td><td>23.40</td><td>19.56</td><td>18.62</td><td>0.687</td><td>27.34</td></tr><tr><td>Qwen-3.8-Flash-Next [25]</td><td>27.83</td><td>0.544</td><td>0.753</td><td>28.31</td><td>25.38</td><td>21.04</td><td>24.07</td><td>0.679</td><td>21.50</td></tr><tr><td>DeepSeek-4.1-Flash [32]</td><td>12.34</td><td>0.740</td><td>0.933</td><td>10.45</td><td>12.74</td><td>12.59</td><td>6.95</td><td>0.843</td><td>7.62</td></tr><tr><td>GLM-5.3-Flash [39]</td><td>7.51</td><td>0.860</td><td>0.936</td><td>4.50</td><td>7.12</td><td>6.65</td><td>7.10</td><td>0.923</td><td>6.55</td></tr><tr><td colspan="10">Specialized VLMs (OCR-3.0)</td></tr><tr><td>Unlimited-OCR 2 (Ours)</td><td>77.06</td><td>0.104</td><td>0.123</td><td>87.93</td><td>74.81</td><td>71.11</td><td>71.79</td><td>0.147</td><td>48.20</td></tr></table>
<BLOCK>title<LEVEL3>[115, 654, 269, 668]<FORK>5.2. Main Results
<BLOCK>text[113, 673, 886, 900]<FORK>As visually summarized in Figure 6 and further detailed in Table 1 and Table 2, Unlimited OCR 2 takes a significant step toward the OCR 3.0 era, advancing document parsing along three key dimensions. First, it establishes the practical feasibility of hundred-page end-to-end parsing, achieving a score of 77.06 on MarathonDocBench and demonstrating its ability to handle documents at a scale beyond conventional multi-page settings. Second, it maintains robust parsing performance as document length increases. Through end-to-end modeling with SR-SWA, Unlimited OCR 2 filters redundant visual tokens while preserving relevant cross-page information, achieving the highest Overall scores and state-of-the-art performance on both cross-page text and table metrics, T-Text and T-Table, respectively, from dozens of pages on MPDocBench to hundreds of pages on MarathonDocBench. Third, it establishes a new state-of-the-art in long-document parsing, outperforming leading closed-source models and top-tier open-source MLLMs across these benchmarks. Furthermore, Unlimited OCR 2 retains strong single-page recognition capabilities, achieving an Overall score of 94.02% on OmniDocBench in Table 3, confirming that its sparse attention design preserves fine-grained local recognition.
<BLOCK>page_number[489, 923, 511, 936]<FORK>10