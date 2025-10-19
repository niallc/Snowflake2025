# Hetzner Server Setup — Public Summary

This document summarizes the setup of a small hobby site hosted on a Hetzner cloud instance with a domain managed by a third-party DNS provider.  
For further details or questions, contact **Niall Cardin** at **niallc@gmail.com**.

---

## Overview

- **Domain:** `sf25.niallcardin.com`  
- **Server:** Hetzner cloud instance running Ubuntu 24.04 LTS  
- **Web server:** nginx (reverse proxy)  
- **Primary purpose:** Serving a small web project as a public hobby site

---

## Key Setup Details

### HTTPS with Let's Encrypt
The site uses HTTPS via a free Let's Encrypt TLS certificate, issued and managed by `certbot`.  
- Certificates are stored under `/etc/letsencrypt/live/sf25.niallcardin.com/`.
- Renewal is automatic (`certbot renew` is scheduled via systemd).

Visiting `http://sf25.niallcardin.com` automatically redirects to HTTPS.

---

### Firewall (UFW)

A minimal firewall is enabled via **UFW** (Uncomplicated Firewall), configured to:
- **Allow:**  
  - Port 22 (SSH)  
  - Port 80 (HTTP)  
  - Port 443 (HTTPS)  
- **Deny:** All other inbound traffic by default

This reduces the attack surface while preserving normal web and SSH access.

---

### Automatic Security Updates

`unattended-upgrades` is installed and enabled to automatically install daily security patches.  
Configuration file: `/etc/apt/apt.conf.d/20auto-upgrades`

---

### Basic System Monitoring

The lightweight tool **Monit** is configured to monitor CPU and memory usage and send email alerts if they remain high for extended periods.  
This provides early warnings of potential performance or security issues.

---

### HTTP Security Headers

Several security-related headers are added via nginx:

- `Strict-Transport-Security: max-age=31536000; includeSubDomains`  
- `X-Content-Type-Options: nosniff`  
- `X-Frame-Options: DENY`  
- `Referrer-Policy: no-referrer-when-downgrade`  
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`

These headers strengthen browser-side security and reduce common web vulnerabilities.

Additionally, `server_tokens off;` is set to prevent nginx from exposing its version number.

---

## Future Improvements (Optional)

- **Content-Security-Policy (CSP):** Further limits browser behavior and reduces XSS risk.  
- **Backups:** Automate snapshots or periodic data backups.  
- **Monitoring Dashboard:** Add metrics dashboards (e.g. Grafana + Prometheus) for deeper visibility.  
- **HSTS Preload:** Enforce HTTPS even before first visit (requires commitment to HTTPS for all subdomains).

---

## Contact

For questions about this setup or the project:  
📧 **niallc@gmail.com**
