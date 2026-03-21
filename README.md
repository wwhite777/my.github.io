# Woncheol Jade Jeong

**ML/AI Engineer** | Building production-grade LLM systems, agentic AI, and distributed training infrastructure. Targeting applied research and engineering roles at top US AI labs.

---

## Featured Projects

### [MedLlama](https://github.com/wwhite777/MedLlama) -- Healthcare-Specialized LLM
<img width="1920" height="1080" alt="image for linkedin v2" src="https://github.com/user-attachments/assets/59b7ba75-6c50-4866-8ec8-4df5ddc563e5" />

Fine-tuned Llama 3.1 8B on medical QA datasets (MedQA, PubMedQA, ChatDoctor) to build a healthcare-specialized language model. Designed and deployed a production-grade RAG pipeline backed by PubMed abstracts in a Qdrant vector database, served through vLLM and FastAPI with W&B experiment tracking.

**Technologies:** Llama 3.1 8B, QLoRA, vLLM, FastAPI, Qdrant, RAG, Weights & Biases

**Impact:** Demonstrates end-to-end LLM engineering for domain-specific AI -- from fine-tuning to retrieval-augmented generation to production serving. Directly applicable to healthcare AI roles at companies building clinical decision support, medical chatbots, or drug discovery tools.

---

### [MedAgent](https://github.com/wwhite777/MedAgent) -- Multi-Agent Clinical Decision Support
<img width="1920" height="1080" alt="image v2" src="https://github.com/user-attachments/assets/c6e6d1de-241f-4551-baa2-64d60248e2f7" />

Multi-agent clinical decision support system built on LangGraph and MCP (Model Context Protocol). Implements an agentic workflow -- triage, literature retrieval, clinical reasoning, report generation -- with conditional routing by ESI (Emergency Severity Index) level and human-in-the-loop safeguards for emergency cases. Integrates with MedLlama's RAG API via MCP tool calling, with a Gradio demo UI.

**Technologies:** LangGraph StateGraph, MCP, GPT-4o-mini, Gradio, multi-agent orchestration, human-in-the-loop

**Impact:** Showcases cutting-edge agentic AI engineering -- multi-agent orchestration, tool use, and human-in-the-loop safety for high-stakes domains. Directly relevant to AI agent roles at Anthropic, Google DeepMind, Amazon, and Microsoft.

---

### [DistTrain](https://github.com/wwhite777/distributed-fine-tuning) -- Distributed Training and Inference Infrastructure

Head-to-head comparison of distributed training frameworks -- PyTorch FSDP2, DeepSpeed ZeRO-3, and JAX/Flax -- fine-tuning Llama 3.1 8B across multi-GPU setups with comprehensive profiling of throughput, peak memory, scaling efficiency, and memory breakdown by component. Includes inference serving benchmarks (vLLM vs TensorRT-LLM with INT4/FP8 quantization) and a full MLOps pipeline with W&B dashboards and MLflow model registry.

**Technologies:** PyTorch FSDP2, DeepSpeed ZeRO-3, JAX/Flax, vLLM, TensorRT-LLM, W&B, MLflow, CUDA

**Impact:** Demonstrates deep understanding of GPU-scale infrastructure and distributed systems -- the core competency for Staff/Senior ML roles at top AI labs. The JAX implementation is specifically relevant to Anthropic and Google DeepMind.

---

## Portfolio Impact Summary

These three projects together cover the full ML engineering stack that top AI labs demand. **MedLlama** proves the ability to fine-tune and serve LLMs for real-world domains. **MedAgent** demonstrates mastery of the agentic AI paradigm -- multi-agent orchestration, tool use, and safety-critical human-in-the-loop design. **DistTrain** shows the systems-level depth to train and serve models at scale across competing frameworks. From model training to agentic reasoning to distributed infrastructure, this portfolio maps directly to the end-to-end skill set required at Anthropic, Google DeepMind, and other leading AI organizations.

---

## Technical Skills

* **Programming and Core Tools:** Python (PyTorch, JAX, NumPy, SciPy, FastAPI, Pydantic), C/C++, SQL; Git, Linux, Docker, Kubernetes; AI-native IDEs (Cursor/GitHub Copilot)
* **Deep Learning and Model Development:** CNN, RNN, Vision Transformers (ViT, Swin), Diffusion Models; LLM fine-tuning (LoRA, QLoRA, SFT, RLHF, DPO); Distributed training (FSDP, DeepSpeed); HuggingFace Transformers and Accelerate
* **LLM and Generative AI:** RAG architecture design, prompt engineering (CoT, ReAct), LLM evaluation and safety; Foundation model integration (GPT-4, Claude, Gemini, Llama); Inference optimization (quantization, vLLM, TensorRT-LLM)
* **AI Agents and Agentic Systems:** LangChain/LangGraph, MCP (Model Context Protocol), tool use and function calling, multi-agent orchestration
* **Computer Vision:** Object detection (YOLO, R-CNN), segmentation (SAM, U-Net), Vision-Language Models, 3D vision; Video understanding, medical image analysis
* **NLP and Language Understanding:** Transformer architectures, text generation, multilingual NLP, information extraction
* **MLOps and Infrastructure:** MLflow, Weights & Biases, CI/CD (GitLab CI, GitHub Actions); Model serving (vLLM, Triton, TorchServe); Data pipelines (Airflow), experiment tracking, model monitoring
* **Cloud and Compute:** AWS (SageMaker) / GCP (Vertex AI) / Azure ML; GPU cluster management, multi-node training, CUDA
* **Data Infrastructure:** Vector databases (pgvector, Qdrant/Pinecone), feature stores, data versioning (DVC)
* **Domain Expertise:** Healthcare AI (medical imaging, clinical NLP, FDA/regulatory), Robotics and Manufacturing (ROS, sim-to-real, motion planning), Finance/Operations (decision support systems)
