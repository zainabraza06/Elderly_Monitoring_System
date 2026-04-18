FROM node:20-alpine

ENV PORT=80

WORKDIR /app/web_frontend

COPY web_frontend/package.json web_frontend/package-lock.json ./
# Build needs devDependencies (e.g. Tailwind PostCSS plugin).
RUN npm ci --include=dev

COPY web_frontend ./
RUN npm run build

ENV NODE_ENV=production

EXPOSE 80

CMD ["sh", "-c", "npm run start -- --hostname 0.0.0.0 --port ${PORT:-80}"]