FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    make \
    g++ \
  && rm -rf /var/lib/apt/lists/*

COPY package*.json ./

RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY server.js ./
COPY public ./public

RUN mkdir -p /app/data

ENV PORT=7860
ENV DB_PATH=/app/data/ai-studio.sqlite

EXPOSE 7860

CMD ["npm", "start"]