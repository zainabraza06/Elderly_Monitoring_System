# CapRover CI/CD Setup

This repository now contains independent deployment workflows for backend and web frontend.

- Backend auto-deploy workflow: `.github/workflows/deploy-backend.yml`
- Frontend auto-deploy workflow: `.github/workflows/deploy-frontend.yml`

## Trigger behavior

- Backend deploy triggers on pushes to `main` or `master` when files under `flask_backend/` change.
- Frontend deploy triggers on pushes to `main` or `master` when files under `web_frontend/` change.
- Both workflows also support manual execution through `workflow_dispatch`.

## Required GitHub repository secrets

- `CAPROVER_SERVER`
  - Example: `https://captain.apps.your-domain.com`
- `CAPROVER_BACKEND_APP`
  - CapRover app name for backend service
- `CAPROVER_BACKEND_APP_TOKEN`
  - App token from backend app Deployment tab
- `CAPROVER_FRONTEND_APP`
  - CapRover app name for web frontend service
- `CAPROVER_FRONTEND_APP_TOKEN`
  - App token from frontend app Deployment tab

## CapRover app expectations

### Backend app

- Uses `flask_backend/Dockerfile`
- Exposes port `8000`
- Starts with:
  - `uvicorn flask_backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}`

### Frontend app

- Uses `web_frontend/Dockerfile`
- Exposes port `3000`
- Starts with:
  - `npm run start -- --hostname 0.0.0.0 --port ${PORT:-3000}`

## Notes

- The workflows package only the relevant folder and a root-level `captain-definition` file before deploying.
- Frontend deployment archive excludes `web_frontend/node_modules` and `web_frontend/.next`.
- Backend deployment archive excludes common cache and generated report folders.