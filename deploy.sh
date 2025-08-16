#!/usr/bin/env bash
# ASCII-only, safe to paste.
set -euo pipefail

# ---- Config (override via env) ----
PROJECT_ID=${PROJECT_ID:-cv-mlops-lab}
REGION=${REGION:-europe-west1}
REPO=${REPO:-cv}
API_TAG=${API_TAG:-v1}
UI_TAG=${UI_TAG:-v1}

API_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/api:$API_TAG"
UI_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/ui:$UI_TAG"

say()  { printf "\n%s\n" "$*"; }
warn() { printf "\nWARN: %s\n" "$*"; }
die()  { printf "\nERROR: %s\n" "$*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing '$1'."; }

# ---- Preflight ----
need gcloud

ACTIVE_ACCOUNT=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null || true)
if [ -z "${ACTIVE_ACCOUNT}" ]; then
  gcloud auth login
  ACTIVE_ACCOUNT=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)")
fi

if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  gcloud auth application-default login
fi

gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1 || die "Project '$PROJECT_ID' not found."
gcloud config set project "$PROJECT_ID" >/dev/null
gcloud config set run/region "$REGION"   >/dev/null

# Billing must already be linked
BILLING_ENABLED=$(gcloud beta billing projects describe "$PROJECT_ID" --format="value(billingEnabled)" 2>/dev/null || echo "False")
[ "$BILLING_ENABLED" = "True" ] || die "Project not linked to billing."

# Enable required APIs (idempotent)
gcloud services enable \
  serviceusage.googleapis.com \
  cloudbilling.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com \
  aiplatform.googleapis.com

# IAM (best-effort)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CB_SA="$PROJECT_NUMBER@cloudbuild.gserviceaccount.com"
RUN_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${ACTIVE_ACCOUNT}" --role="roles/cloudbuild.builds.editor" >/dev/null 2>&1 || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${CB_SA}" --role="roles/artifactregistry.writer" >/dev/null 2>&1 || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUN_SA}" --role="roles/artifactregistry.reader" >/dev/null 2>&1 || true

# Ensure Artifact Registry repo
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"
fi

# Sanity: required files
for f in cloudbuild.api.yaml cloudbuild.ui.yaml dockerfile.serve dockerfile.ui; do
  [ -f "$f" ] || die "Missing '$f' in repo root."
done

# ---- Build images (unless SKIP_BUILD is set) ----
if [ -z "${SKIP_BUILD:-}" ]; then
  gcloud builds submit --project "$PROJECT_ID" \
    --config cloudbuild.api.yaml \
    --substitutions=_REGION="$REGION",_REPO="$REPO",_IMAGE=api,_TAG="$API_TAG" .

  gcloud builds submit --project "$PROJECT_ID" \
    --config cloudbuild.ui.yaml \
    --substitutions=_REGION="$REGION",_REPO="$REPO",_IMAGE=ui,_TAG="$UI_TAG" .
else
  warn "SKIP_BUILD=1 set: skipping image builds."
fi

# ---- Deploy API ----
gcloud run deploy cv-api \
  --image "$API_IMG" \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu=2 \
  --memory=4Gi \
  --concurrency=1 \
  --min-instances=1 \
  --timeout=600 \
  --quiet

API_URL=$(gcloud run services describe cv-api --region "$REGION" --format='value(status.url)')
[ -n "$API_URL" ] || die "Failed to resolve API URL."
echo "API_URL=$API_URL"
curl -fsS "$API_URL/health" >/dev/null 2>&1 || warn "API /health not ready yet (cold start is OK)."

# ---- Deploy UI ----
gcloud run deploy cv-ui --image "$UI_IMG" --allow-unauthenticated \
  --region "$REGION" --set-env-vars "API_URL=$API_URL" --quiet

UI_URL=$(gcloud run services describe cv-ui --region "$REGION" --format='value(status.url)')
[ -n "$UI_URL" ] || die "Failed to resolve UI URL."
echo "UI_URL=$UI_URL"
