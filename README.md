# hku.G.rag: A General-Purpose RAG Framework for University Courseware

**An Advanced, Multi-Stage Retrieval and Reranking Architecture for High-Fidelity Academic Q&A**

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Key%20Technologies-vLLM%20%7C%20FAISS%20%7C%20Qwen-blueviolet" alt="Key Technologies"></a>
</p>

### 1. Abstract: Redefining Academic Information Retrieval

**HKU.G.RAG** (Hong Kong University General-purpose Retrieval-Augmented Generation) represents a significant leap forward in the domain of academic AI assistants. Standard Retrieval-Augmented Generation (RAG) systems, while functional, often fail to capture the high information density, complex logical structures, and instructional nuance inherent in university-level courseware. This project addresses these shortcomings by architecting a sophisticated, multi-stage pipeline that ensures both exceptional recall and state-of-the-art precision.

Our framework is engineered around a powerful Large Language Model (`Qwen1.5-7B-Chat`) and is meticulously designed to transform dense academic materials, such as lecture slides, into a dynamic and queryable knowledge fabric. By synergizing a **hybrid retrieval engine** (combining semantic and lexical search) with a cutting-edge **cross-encoder reranking** stage, HKU.G.RAG delivers unparalleled accuracy and contextual relevance. This system is not merely a Q&A bot; it is a robust framework for building high-fidelity knowledge discovery tools for the academic community.

### 2. The Problem Domain: The Unique Challenges of Academic RAG

Academic documents are a distinct and challenging modality for information retrieval, differing fundamentally from generic web text. A successful system must overcome:

*   **Instructional Nuance & Complexity:** Academic queries are rarely simple fact-finding missions. They often contain complex, multi-part instructions (e.g., *"Compare and contrast the Keynesian and Monetarist perspectives on inflation as detailed in the lecture notes"*). This requires a retrieval system that can follow instructions, not just match keywords.
*   **The Semantic-Lexical Chasm:** The language of academia is precise. A query might use a semantic synonym (e.g., *"capital allocation strategies"*) that a purely keyword-based search would fail to capture. Conversely, a purely semantic search might overlook a critical, non-negotiable term of art (e.g., *"Fama-French three-factor model"*). An effective system must bridge this chasm.
*   **Semantic Fragmentation:** In slide-based courseware, a single coherent concept is often fragmented across multiple pages, bullet points, and speaker notes. Naive, fixed-size chunking strategies can sever these critical logical connections, feeding the LLM an incomplete or misleading context and leading to shallow, incomplete answers.

### 3. Architectural Blueprint: A Multi-Stage Pipeline for Precision-Recall Optimization

To conquer these challenges, HKU.G.RAG implements a meticulously designed four-stage pipeline, moving from coarse-grained recall to fine-grained precision.

#### Stage 1: Ingestion & Intelligent Chunking
The integrity of the knowledge base is paramount. We reject a one-size-fits-all approach to document parsing and instead employ a multi-strategy methodology to ensure semantic coherence.
1.  **Layout-Aware Block Parsing:** Leveraging advanced PDF parsing libraries, this primary strategy segments the document based on its visual and structural layout. It intelligently groups elements like a slide title with its corresponding bullet points, preserving the immediate, author-intended context.
2.  **Overlapping Sliding Window:** To reconstruct concepts that span arbitrary boundaries (like pages or blocks), we deploy a sliding window strategy. The entire document text is serialized and split into sentences. A window of a defined sentence count then slides across this sequence with a significant overlap, ensuring that no logical transition is severed.
3.  **Uniform Baseline Chunking:** As a fallback and for capturing broader thematic context, we also generate conventional, non-overlapping chunks based on a fixed token length.

This tri-modal chunking strategy produces a rich, redundant set of document fragments, which are subsequently deduplicated to form a comprehensive and semantically robust knowledge corpus.

#### Stage 2: Hybrid Retrieval Engine for Maximized Recall
This stage is engineered to cast the widest possible net, ensuring that no potentially relevant information is left behind. We implement a **hybrid retrieval** engine that synergizes two complementary search paradigms:

1.  **Dense Retrieval (Semantic Search):** At the core of our semantic search is the **`Qwen/Qwen3-Embedding-8B`** model, a state-of-the-art text embedding model chosen for its demonstrated excellence in instruction following (via the **FollowIR** benchmark). It transforms both user queries and document chunks into high-dimensional vectors, capturing deep semantic intent. These vectors are indexed in **FAISS** (Facebook AI Similarity Search) for blazingly fast, massively parallel similarity search.
2.  **Sparse Retrieval (Lexical Search):** To ground our semantic search and guarantee that critical, literal terms are captured, we implement the **Okapi BM25** algorithm. This battle-tested sparse retrieval method excels at matching exact keywords and technical jargon, providing an essential safeguard against the potential for semantic drift in dense models.

The union of candidates from both dense and sparse retrieval forms a comprehensive result set, optimized for maximum recall, which is then passed to the next stage for refinement.

#### Stage 3: Cross-Encoder Reranking for Precision Enhancement
With a high-recall set of candidate documents, the focus shifts to precision. This is accomplished by a **cross-encoder reranking** stage, powered by the **`Qwen/Qwen3-Reranker-8B`** model.

Unlike bi-encoder models (like our embedding model) that generate vector representations of the query and documents independently, a cross-encoder performs a full, attention-based comparison of the query and each candidate document simultaneously. This allows for a far deeper, more contextual, and more accurate assessment of relevance. The reranker assigns a precise relevance score to each document, allowing us to reorder the candidates and select a top-K subset of only the most pertinent information to inject into the LLM's context. This critical step drastically reduces context noise, mitigates the "lost in the middle" problem, and significantly lowers the computational cost of the final generation step.

#### Stage 4: Accelerated Generation with vLLM
The final stage is the synthesis of the answer by the generative LLM. To ensure production-grade performance, we utilize **vLLM** to serve our base model, **`Qwen/Qwen1.5-7B-Chat`**. By implementing advanced techniques like **PagedAttention** and **continuous batching**, vLLM dramatically increases inference throughput and reduces latency compared to standard inference frameworks, enabling a fluid, real-time conversational experience.

### 4. Technology Stack & Rationale

| Component | Selected Technology | Rationale & Justification |
| :--- | :--- | :--- |
| **Generative LLM** | `Qwen/Qwen1.5-7B-Chat` | A powerful and well-balanced open-source model with excellent instruction-following capabilities and a manageable size for efficient deployment. |
| **Embedding Model** | `Qwen/Qwen3-Embedding-8B` | State-of-the-art performance on the **FollowIR** benchmark, which specifically evaluates instruction-based retrieval, making it ideal for our academic use case. |
| **Reranker Model** | `Qwen/Qwen3-Reranker-8B` | A powerful cross-encoder model from the same family as our embedding model, ensuring architectural synergy. Its performance is competitive with top-tier proprietary solutions. |
| **Inference Engine**| **vLLM** | The industry standard for high-performance LLM serving. Its PagedAttention algorithm optimizes memory usage and enables high-throughput, low-latency inference. |
| **Vector Store** | **FAISS** | A highly efficient, scalable library for similarity search over dense vectors, developed and maintained by Meta AI. |
| **Sparse Retriever**| **Okapi BM25** | The gold standard for lexical search, providing a robust and computationally efficient method for keyword-based retrieval. |

### 5. Roadmap & Future Horizons

This project establishes a powerful baseline. Our roadmap for future development includes:

1.  **Multi-Modal Fusion:** Extending the ingestion pipeline to parse and index images, diagrams, and complex tables from academic papers, and enabling the LLM to reason over this fused multi-modal context.
2.  **Dynamic Knowledge Graph Integration:** Augmenting the retrieval process by constructing a real-time knowledge graph from the document chunks. This will enable the system to answer complex, multi-hop questions that require synthesizing information across dozens of documents.
3.  **Adaptive Retrieval Strategies:** Implementing a meta-agent that analyzes the incoming query and dynamically adjusts the retrieval strategy—for instance, weighting dense vs. sparse results differently, or initiating advanced query transformation techniques (e.g., HyDE) for ambiguous questions.

### 6. Quick Start Guide

This section provides a step-by-step guide to setting up and running the HKU.G.RAG system.

#### 6.1 Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/hku.G.rag.git
    cd hku.G.rag
    ```

2.  **Install System Dependencies (for `git-lfs`):**
    ```bash
    sudo apt-get update && sudo apt-get install git-lfs
    git lfs install
    ```

3.  **Install Python Dependencies:**
    It is strongly recommended to use a dedicated virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

#### 6.2 Model Acquisition

The system requires several large models from the Hugging Face Hub. It is recommended to have `huggingface-cli` installed and logged in.

1.  **LLM (Qwen-1.5-7B-Chat):**
    ```bash
    git clone https://huggingface.co/Qwen/Qwen1.5-7B-Chat models/Qwen1.5-7B-Chat
    ```

2.  **Embedding Model (Qwen3-Embedding-8B):**
    ```bash
    git clone https://huggingface.co/Qwen/Qwen3-Embedding-8B models/Qwen3-Embedding-8B
    ```

3.  **Reranker Model (Qwen3-Reranker-8B):**
    ```bash
    git clone https://huggingface.co/Qwen/Qwen3-Reranker-8B models/Qwen3-Reranker-8B
    ```

#### 6.3 Running the Application

1.  **Prepare Data:** Place your courseware PDF files (e.g., `IDAT7215_slides.pdf`) in the `/data` directory. The system will automatically index them on the first run.

2.  **Test vLLM Acceleration (Optional but Recommended):**
    Verify that the vLLM service is correctly configured to accelerate your Qwen model.
    ```bash
    python vllm_model.py
    ```

3.  **Run the Main Q&A Application:**
    Launch the main script to start an interactive Q&A session or to process a batch of questions from a file.
    ```bash
    python run.py
    ```
    The application will log its detailed operations and save any generated results to the `/data` directory.

#### 6.4 Evaluation
To benchmark the system's performance against a gold-standard dataset, use the provided evaluation script. Ensure your question file (e.g., `data/7215_question.json`) and ground-truth answer file (e.g., `data/7215_gold.json`) are present.
```bash
python test_score.py
```
A detailed breakdown of the evaluation metrics will be logged to `output_test.log`.


Some Random Test:

```bash

conda create -n rag python=3.12
python -m venv v2/venv/
pip install --upgrade pip
source v2/venv/bin/activate
source config.ini

###################################################
# when libcrypto.so.1.1: cannot open shared object file: No such file or directory
cd /tmp
wget https://www.openssl.org/source/openssl-1.1.1o.tar.gz

sudo tar -zxvf openssl-1.1.1o.tar.gz

cd openssl-1.1.1o

# compile and install
sudo ./config
sudo make
sudo make install

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
/root/autodl-tmp/hku.G.rag/v2/mongodb-7.0.20/bin/mongod --version

###################################################

cd models
bash download.sh

python infer.py

# when you want to empty the GPUs
python3 -c '''
import torch
torch.cuda.empty_cache()
'''
# or you can
ps aux | grep python
ps aux | grep vllm
```
