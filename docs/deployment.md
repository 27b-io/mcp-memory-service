# Deployment Guide

## Quick Start (Docker)

Run with embedded Qdrant and local embedding — no external dependencies:

```bash
docker run -d \
  -p 8001:8001 \
  -v memory-data:/app/data \
  -e MCP_TRANSPORT_MODE=streamable-http \
  -e MCP_SERVER_PORT=8001 \
  ghcr.io/27b-io/mcp-memory-service:latest
```

MCP client config:

```json
{
  "mcpServers": {
    "memory": {
      "type": "http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

Data stored in the container volume. No graph layer. Good for trying it out.

## Docker Compose

Full stack: Qdrant (vector search) + FalkorDB (knowledge graph) + memory service.

```bash
docker compose up -d
```

### Services

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `qdrant` | `qdrant/qdrant:latest` | 6333 | Vector database (HNSW index) |
| `falkordb` | `falkordb/falkordb:latest` | internal | Knowledge graph (Redis protocol, AOF persistence) |
| `mcp-memory` | Built from Dockerfile | 8000, 8001 | Memory service |

### Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP | REST API |
| 8001 | HTTP | MCP streamable-http transport |
| 6333 | HTTP | Qdrant API (only expose if you need direct access) |

### Volumes

| Volume | Mounted at | Purpose |
|--------|-----------|---------|
| `qdrant-data` | `/qdrant/storage` | Vector index + data |
| `falkordb-data` | `/data` | Graph database (AOF persistence) |
| `mcp-memory-backups` | `~/.local/share/mcp-memory/backups` | Memory backups |

### Resource Limits

| Service | CPU limit | Memory limit | CPU reservation | Memory reservation |
|---------|-----------|-------------|-----------------|-------------------|
| Qdrant | 2 cores | 2 GB | 1 core | 512 MB |
| FalkorDB | 1 core | 1 GB | 0.25 cores | 512 MB |
| Memory service | 2 cores | 2 GB | 1 core | 1 GB |

Total: ~5 cores, ~5 GB RAM at limits. Minimum viable: ~2.25 cores, ~2 GB RAM at reservations.

### Environment Variables in Compose

The `docker-compose.yml` sets these for the memory service:

```yaml
environment:
  # Storage
  - MCP_MEMORY_EMBEDDING_MODEL=intfloat/e5-small-v2
  - MCP_QDRANT_URL=http://qdrant:6333

  # FalkorDB graph layer
  - MCP_FALKORDB_ENABLED=true
  - MCP_FALKORDB_HOST=falkordb
  - MCP_FALKORDB_PORT=6379

  # CacheKit L2 (reuses FalkorDB on db 1)
  - REDIS_URL=redis://falkordb:6379/1

  # HTTP server
  - MCP_HTTP_ENABLED=true
  - MCP_HTTP_PORT=8000

  # MCP server (streamable-http transport)
  - MCP_TRANSPORT_MODE=streamable-http
  - MCP_SERVER_PORT=8001
```

## From Source (stdio)

For local development:

```bash
git clone https://github.com/27b-io/mcp-memory-service.git
cd mcp-memory-service
uv sync
uv run memory server
```

MCP client config for stdio transport:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-memory-service", "memory", "server"]
    }
  }
}
```

## Kubernetes (Recommended for Production)

The recommended production pattern uses three components:

| Component | Image | Steady-state RAM | Purpose |
|-----------|-------|-----------------|---------|
| **mcp-memory** (slim) | `ghcr.io/27b-io/mcp-memory-service:vX.Y.Z-slim` | ~135 MB | Memory service (no embedded model) |
| **TEI** | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9` | ~3.9 GB | Embedding inference (shared, OpenAI-compatible API) |
| **Qdrant** | `qdrant/qdrant:v1.16.3` | ~300 MB | Vector database |

This separates embedding inference from the memory service. The slim image excludes PyTorch/ONNX (~2 GB), keeping mcp-memory lean while TEI serves embeddings via an OpenAI-compatible `/v1/embeddings` endpoint. TEI can be shared across multiple services.

### Qdrant

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: mcp
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.16.3
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        env:
        - name: QDRANT__STORAGE__ON_DISK_PAYLOAD
          value: "true"
        - name: QDRANT__STORAGE__MMAP_ENABLED
          value: "true"
        - name: QDRANT__SERVICE__MAX_REQUEST_SIZE_MB
          value: "16"
        resources:
          requests:
            cpu: 15m
            memory: 512Mi
          limits:
            memory: 1Gi
        volumeMounts:
        - name: qdrant-data
          mountPath: /qdrant/storage
        livenessProbe:
          httpGet:
            path: /healthz
            port: 6333
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /readyz
            port: 6333
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: qdrant-data
        persistentVolumeClaim:
          claimName: qdrant-data
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: mcp
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
    name: http
  - port: 6334
    targetPort: 6334
    name: grpc
  type: ClusterIP
```

Enable MMAP and on-disk payload to keep Qdrant memory-efficient. For small datasets (<100k vectors), a 1 GB limit is sufficient.

### TEI (Text Embeddings Inference)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tei
  namespace: mcp
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: tei
  template:
    metadata:
      labels:
        app: tei
    spec:
      containers:
      - name: tei
        image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
        args:
        - --model-id
        - Snowflake/snowflake-arctic-embed-l-v2.0
        - --port
        - "80"
        - --tokenization-workers
        - "4"
        - --max-batch-tokens
        - "2048"
        ports:
        - containerPort: 80
          name: http
        env:
        - name: HF_HOME
          value: "/data"
        - name: OMP_NUM_THREADS
          value: "2"
        - name: RAYON_NUM_THREADS
          value: "2"
        resources:
          requests:
            cpu: 100m
            memory: 4Gi
          limits:
            memory: 12Gi
        volumeMounts:
        - name: tei-data
          mountPath: /data
        readinessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
      volumes:
      - name: tei-data
        persistentVolumeClaim:
          claimName: tei-data
---
apiVersion: v1
kind: Service
metadata:
  name: tei
  namespace: mcp
spec:
  selector:
    app: tei
  ports:
  - port: 80
    targetPort: 80
    name: http
  type: ClusterIP
```

#### TEI Configuration

TEI uses the ONNX backend on CPU. The critical tuning parameters are:

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--max-batch-tokens` | 16384 | **2048** | Default pre-allocates warmup buffers proportional to this value. 16384 causes ~12 GB usage for a 1.3 GB model. |
| `OMP_NUM_THREADS` | all cores | **2** | Limits OpenMP parallelism. Prevents thread contention on shared nodes. |
| `RAYON_NUM_THREADS` | all cores | **2** | Limits Rayon parallelism (Rust threadpool). Same reason. |
| `--tokenization-workers` | 1 | **4** | Tokenization is CPU-light; more workers reduce head-of-line blocking. |

**Image tag matters:** The plain `1.9` tag is CUDA-only and crashes without a GPU. Use `cpu-1.9` for CPU deployments.

**Model caching:** Mount a persistent volume at `/data` (set `HF_HOME=/data`). The model (~2.2 GB) downloads on first boot and persists across restarts. Without this, every pod restart re-downloads from HuggingFace.

**Memory profile after tuning:**

| Metric | Value |
|--------|-------|
| Steady-state RAM | ~3.9 GB |
| Warmup time | ~3.5 seconds |
| Model size on disk | ~2.2 GB |

#### Alternative Embedding Models

Pass a different `--model-id` to TEI. Update `MCP_EMBEDDING_DIMENSIONS` on mcp-memory to match.

| Model | Dims | Context | Notes |
|-------|------|---------|-------|
| `Snowflake/snowflake-arctic-embed-l-v2.0` | 1024 | 8192 | Recommended — best quality |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | Good balance |
| `intfloat/e5-large-v2` | 1024 | 512 | High quality, short context |
| `intfloat/e5-base-v2` | 768 | 512 | Smaller footprint |

Changing the model requires creating a new Qdrant collection (dimensions must match). Set `MCP_QDRANT_COLLECTION_NAME` to a new collection name to avoid conflicts.

### MCP Memory Service (Slim)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-memory
  namespace: mcp
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: mcp-memory
  template:
    metadata:
      labels:
        app: mcp-memory
    spec:
      containers:
      - name: mcp-memory
        image: ghcr.io/27b-io/mcp-memory-service:vX.Y.Z-slim
        ports:
        - containerPort: 8000
          name: http-api
        - containerPort: 8001
          name: mcp
        env:
        # Qdrant backend
        - name: MCP_QDRANT_URL
          value: "http://qdrant:6333"
        - name: MCP_QDRANT_COLLECTION_NAME
          value: "memories_arctic1024"

        # External embedding via TEI
        - name: MCP_EMBEDDING_PROVIDER
          value: "openai_compat"
        - name: MCP_EMBEDDING_URL
          value: "http://tei:80"
        - name: MCP_EMBEDDING_DIMENSIONS
          value: "1024"
        - name: MCP_MEMORY_EMBEDDING_MODEL
          value: "Snowflake/snowflake-arctic-embed-l-v2.0"

        # HTTP server
        - name: MCP_HTTP_ENABLED
          value: "true"
        - name: MCP_HTTP_PORT
          value: "8000"
        - name: MCP_HTTP_HOST
          value: "0.0.0.0"

        # MCP protocol
        - name: MCP_TRANSPORT_MODE
          value: "streamable-http"
        - name: MCP_SERVER_PORT
          value: "8001"

        # FalkorDB graph layer (optional)
        - name: MCP_FALKORDB_ENABLED
          value: "true"
        - name: MCP_FALKORDB_HOST
          value: "falkordb.default.svc.cluster.local"
        - name: MCP_FALKORDB_PORT
          value: "6379"

        # Workers
        - name: MCP_HTTP_WORKERS
          value: "2"

        # Logging
        - name: MCP_LOG_LEVEL
          value: "INFO"
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            cpu: 50m
            memory: 512Mi
          limits:
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-memory
  namespace: mcp
spec:
  selector:
    app: mcp-memory
  ports:
  - port: 8000
    targetPort: 8000
    name: http-api
  - port: 8001
    targetPort: 8001
    name: mcp
  type: ClusterIP
```

#### External Embedding Configuration

The slim image uses `MCP_EMBEDDING_PROVIDER=openai_compat` to delegate embedding to TEI (or any OpenAI-compatible endpoint):

| Variable | Description |
|----------|-------------|
| `MCP_EMBEDDING_PROVIDER` | Set to `openai_compat` for external embedding. Default: `local` (in-process). |
| `MCP_EMBEDDING_URL` | Base URL of the embedding service (e.g., `http://tei:80`). |
| `MCP_EMBEDDING_DIMENSIONS` | Vector dimensions. Must match the model and Qdrant collection. |
| `MCP_MEMORY_EMBEDDING_MODEL` | Model name passed to the provider. TEI ignores this but mcp-memory uses it for logging. |

See [configuration.md](configuration.md) for the full `MCP_EMBEDDING_*` reference.

### Network Policies

Lock down lateral movement with default-deny and explicit allowlists:

```yaml
# Default deny all traffic in namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: mcp
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
# Allow DNS for all pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: mcp
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
---
# Qdrant: only mcp-memory can connect
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qdrant-policy
  namespace: mcp
spec:
  podSelector:
    matchLabels:
      app: qdrant
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: mcp-memory
    ports:
    - protocol: TCP
      port: 6333
    - protocol: TCP
      port: 6334
---
# MCP Memory: reach Qdrant + TEI, accept inbound on API ports
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mcp-memory-policy
  namespace: mcp
spec:
  podSelector:
    matchLabels:
      app: mcp-memory
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    # Tighten this to match your ingress controller or consumer pods.
    # Example: only allow from pods in the same namespace:
    - podSelector: {}
    # Or restrict to a specific namespace:
    # - namespaceSelector:
    #     matchLabels:
    #       kubernetes.io/metadata.name: your-namespace
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8001
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
    - protocol: TCP
      port: 6334
  - to:
    - podSelector:
        matchLabels:
          app: tei
    ports:
    - protocol: TCP
      port: 80
---
# TEI: accept from mcp-memory, egress to internet for model download
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tei-policy
  namespace: mcp
spec:
  podSelector:
    matchLabels:
      app: tei
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: mcp-memory
    ports:
    - protocol: TCP
      port: 80
  egress:
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 10.0.0.0/8
        - 172.16.0.0/12
        - 192.168.0.0/16
        - 169.254.0.0/16
        - 100.64.0.0/10
    ports:
    - protocol: TCP
      port: 443
```

If using FalkorDB, add an egress rule to `mcp-memory-policy` for the FalkorDB service (TCP 6379).

Deploy network policies **before** the services that need egress — otherwise pods start with default-deny and fail connectivity checks.

### Resource Summary

Measured steady-state from production (single-node k3s):

| Component | CPU | RAM (actual) | RAM request | RAM limit |
|-----------|-----|-------------|-------------|-----------|
| mcp-memory (slim) | 3m | 135 MB | 512 Mi | 2 Gi |
| TEI (cpu-1.9) | 11m | 3.9 GB | 4 Gi | 12 Gi |
| Qdrant | 4m | 296 MB | 512 Mi | 1 Gi |
| **Total** | **18m** | **~4.3 GB** | **5 Gi** | **15 Gi** |

TEI dominates memory usage. The 12 Gi limit is conservative — after tuning `--max-batch-tokens` to 2048, 6 Gi is sufficient for most workloads.

## Health Checks

### HTTP Health Endpoint

```text
GET /api/health    # 200 when service is ready
```

Used by Kubernetes probes and the Dockerfile `HEALTHCHECK`.

### MCP Health

Use the `check_database_health` MCP tool — returns backend status, memory count, storage stats.

### Docker Compose Startup Order

All three services have health checks. `depends_on` with `condition: service_healthy` ensures:

1. Qdrant starts and passes HTTP readiness probe
2. FalkorDB starts and passes `redis-cli ping`
3. Memory service starts after both are healthy

## Image Tags

Published to `ghcr.io/27b-io/mcp-memory-service`:

| Tag | Description |
|-----|-------------|
| `latest` | Latest release (includes local embedding model) |
| `vX.Y.Z` | Specific version (semver, from release-please) |
| `vX.Y.Z-slim` | Slim variant — no PyTorch/ONNX, requires external embedding provider |

### Choosing an Image

| Image | Size | Use Case |
|-------|------|----------|
| `latest` / `vX.Y.Z` | ~2.5 GB | Standalone or Docker Compose — everything in one container |
| `vX.Y.Z-slim` | ~500 MB | Kubernetes with TEI — lighter, embedding handled externally |

The slim image requires `MCP_EMBEDDING_PROVIDER=openai_compat` and `MCP_EMBEDDING_URL` pointing to a compatible endpoint (TEI, vLLM, or any OpenAI-compatible `/v1/embeddings` service).

## Build Args

| Arg | Default | Purpose |
|-----|---------|---------|
| `EMBEDDING_MODEL` | `intfloat/e5-small-v2` | Pre-downloaded embedding model |
| `CUDA_ENABLED` | `false` | Include CUDA/GPU support (~5 GB vs ~1.2 GB image) |
