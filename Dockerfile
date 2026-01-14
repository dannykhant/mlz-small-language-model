FROM rust:1.92-trixie AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    clang \
    pkg-config \
    libssl-dev \
    ca-certificates

WORKDIR /app
ENV ORT_STRATEGY=download

COPY Cargo.toml ./
COPY src ./src
COPY models ./models

RUN cargo build --release

FROM debian:trixie-slim
WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/mlz-small-language-model .

EXPOSE 3000
CMD ["./mlz-small-language-model"]
