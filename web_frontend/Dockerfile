FROM node:20-alpine

ENV NODE_ENV=production \
    PORT=3000

WORKDIR /app/web_frontend

COPY web_frontend/package.json web_frontend/package-lock.json ./
RUN npm ci

COPY web_frontend ./
RUN npm run build

EXPOSE 3000

CMD ["sh", "-c", "npm run start -- --hostname 0.0.0.0 --port ${PORT:-3000}"]