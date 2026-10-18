# Vec3 - Chroma vs pgvector

**Performance Evaluation of Vector Database Systems**

A comparative benchmarking study of **Chroma** and **pgvector (PostgreSQL)**, developed as part of the *Information Management Systems* course, Academic Year 2026–2027.

## Project Overview

This project aims to evaluate and compare two open-source vector database systems under consistent experimental conditions.

The evaluation focuses on:

- Vector ingestion and index construction performance
- Storage efficiency and memory consumption
- Similarity search accuracy and latency
- Query throughput under different concurrency levels
- Filtered vector search
- Performance under memory-constrained environments

## Technologies

- Python
- Docker / Docker Compose
- Chroma
- PostgreSQL with pgvector

## Evaluation Metrics

The benchmarking methodology includes Recall@10, query latency (p50, p95, p99), queries per second (QPS), CPU utilization, memory consumption, and index construction time.

Comparisons will be performed at comparable recall levels to ensure a fair evaluation.

## Project Status

Under development. Implementation details, reproducibility instructions, and experimental results will be added as the project progresses.

## Reproducibility

The final repository will include configuration files, dependency versions, benchmark scripts, and instructions for reproducing the experiments.
