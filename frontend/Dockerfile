# RiskIQ frontend — Phase 0 development image.
# Phase 10 adds a production target that builds static assets and serves them.
FROM node:24-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

# Probe 127.0.0.1, not localhost: busybox resolves localhost to ::1 first and Vite
# binds IPv4 only, so a localhost probe fails against a perfectly healthy server.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD wget --spider -q http://127.0.0.1:5173/ || exit 1

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
