# Deployment Guide

## Docker Standalone (Simplest)

Run with embedded Qdrant — no external dependencies:

```bash
docker run -d \
  -p 8001:8001 \
  -v memory-data:/app/data \
  -e MCP_TRANSPORT_MODE=http \
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

## Docker Compose (Recommended)

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
| 8000 | HTTP | REST API + dashboard |
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
  - MCP_MEMORY_STORAGE_BACKEND=qdrant
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

## Kubernetes

### Minimal (Qdrant only)

Point the memory service at an in-cluster or external Qdrant:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-memory
spec:
  replicas: 1
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
        image: ghcr.io/27b-io/mcp-memory-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: MCP_TRANSPORT_MODE
          value: "http"
        - name: MCP_SERVER_PORT
          value: "8001"
        - name: MCP_QDRANT_URL
          value: "http://qdrant:6333"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 30
        resources:
          requests:
            cpu: "1"
            memory: 1Gi
          limits:
            cpu: "2"
            memory: 2Gi
```

### With FalkorDB

Add to the container env:

```yaml
- name: MCP_FALKORDB_ENABLED
  value: "true"
- name: MCP_FALKORDB_HOST
  value: "falkordb-service"
- name: MCP_FALKORDB_PORT
  value: "6379"
```

## Health Checks

### HTTP Health Endpoint

```
GET /api/health
```

Returns 200 when the service is ready. Used by the Dockerfile `HEALTHCHECK`.

### MCP Health

Use the `check_database_health` MCP tool — returns backend status, memory count, storage stats.

### Docker Compose Startup Order

All three services have health checks. `depends_on` with `condition: service_healthy` ensures:

1. Qdrant starts and passes HTTP readiness probe
2. FalkorDB starts and passes `redis-cli ping`
3. Memory service starts after both are healthy

## Build Args

| Arg | Default | Purpose |
|-----|---------|---------|
| `EMBEDDING_MODEL` | `intfloat/e5-small-v2` | Pre-downloaded embedding model |
| `CUDA_ENABLED` | `false` | Include CUDA/GPU support (~5GB vs ~1.2GB image) |

## Image Tags

Published to `ghcr.io/27b-io/mcp-memory-service`:

| Tag | Description |
|-----|-------------|
| `latest` | Latest release |
| `vX.Y.Z` | Specific version (semver, from release-please) |
