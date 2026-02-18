FROM caddy:2-alpine

# Remove default config if present
RUN rm -f /etc/caddy/Caddyfile

# Copy our config
COPY Caddyfile /etc/caddy/Caddyfile

# Caddy runs as non-root by default in this image
