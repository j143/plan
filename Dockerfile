# Build stage
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source files
WORKDIR /app
COPY . /app/

# Build C++ library and Python bindings
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . -j$(nproc) \
    && cd .. \
    && pip install -e .

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy built package from builder stage
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Set working directory
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command: run the dashboard
CMD ["python", "python/dist_training_sim/visualization.py"]

# Expose dashboard port
EXPOSE 8050