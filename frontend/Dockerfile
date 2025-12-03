# Frontend Dockerfile - Horus AI
# Angular 19 com Nginx para servir arquivos estáticos

# ===== STAGE 1: Build =====
FROM node:20-alpine AS builder

WORKDIR /app

# Copiar arquivos de dependências primeiro (cache)
COPY package*.json ./

# Instalar dependências
RUN npm ci

# Copiar código fonte
COPY . .

# Build de produção
RUN npm run build -- --configuration=production

# ===== STAGE 2: Serve =====
FROM nginx:alpine

# Copiar configuração customizada do Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar arquivos buildados do stage anterior
COPY --from=builder /app/dist/squad-globo/browser /usr/share/nginx/html

# Expor porta
EXPOSE 80

# Nginx roda em foreground
CMD ["nginx", "-g", "daemon off;"]
