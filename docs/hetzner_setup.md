# Hetzner Server Setup — PUBLIC SUMMARY

This document summarizes the server setup for a small hobby site hosted on a Hetzner cloud instance with a domain managed by a third-party DNS provider. It records configuration structure and “where things live” without exposing personal details.

---

## 1. Overview of the stack

- **OS:** Ubuntu LTS (current stable release)
- **Domain:** <DOMAIN_NAME> (managed by registrar/DNS provider)
- **Public IP:** <SERVER_IPv4>  (intentionally generalized)
- **Server stack:**
  - **nginx** — reverse proxy and public-facing HTTP/HTTPS server.
  - **Gunicorn** — serves the Python application on localhost.
  - **Let's Encrypt (Certbot)** — provides and renews free TLS certificates for HTTPS.
- **Application:** Python web app running on `127.0.0.1:8000`

**Request flow (high-level):**
1. Browser → `https://<DOMAIN_NAME>`
2. nginx (ports 80/443) → redirects HTTP→HTTPS; terminates TLS.
3. nginx reverse-proxies to Gunicorn at `127.0.0.1:8000`.
4. Gunicorn serves the Python app.

---

## 2. Domain and DNS

`<DOMAIN_NAME>` resolves to `<SERVER_IPv4>`. This is required so Let’s Encrypt can verify domain ownership for certificate issuance.

Example checks (from server):
```bash
hostname -I
getent hosts <DOMAIN_NAME>
```

---

## 3. nginx configuration (illustrative)

Site configuration is managed under `/etc/nginx/`. The structure is standard for Ubuntu:

- `/etc/nginx/nginx.conf` — main config
- `/etc/nginx/sites-available/<DOMAIN_NAME>` — site config
- `/etc/nginx/sites-enabled/` — symlinks to enabled sites

Illustrative server blocks (replace placeholders with your values):

```nginx
server {
    listen 80;
    server_name <DOMAIN_NAME>;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name <DOMAIN_NAME>;

    ssl_certificate /etc/letsencrypt/live/<DOMAIN_NAME>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<DOMAIN_NAME>/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 4. HTTPS with Let's Encrypt (Certbot)

Installed via apt with the nginx plugin, then certificate obtained with:
```bash
sudo certbot --nginx -d <DOMAIN_NAME>
```

Certificates live under:
- `/etc/letsencrypt/live/<DOMAIN_NAME>/fullchain.pem`
- `/etc/letsencrypt/live/<DOMAIN_NAME>/privkey.pem`

Renewal is automated by Certbot (systemd/cron). Test with:
```bash
sudo certbot renew --dry-run
```

---

## 5. Useful commands

```bash
# nginx
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl status nginx

# cert status
sudo certbot certificates

# ports
sudo ss -tuln
```

---

## 6. Baseline hardening checklist (non-exhaustive)

- **Firewall:** Only allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS).
- **Automatic security updates:** Enable unattended upgrades.
- **Security headers:** Add HSTS, X-Content-Type-Options, etc.
- **Backups:** Code + key configs.
- **Monitoring/logging:** Basic system and nginx log checks.

---

*This public summary intentionally omits personal email addresses, exact package versions, and exact IP. Replace placeholders as needed for private/internal documentation.*
