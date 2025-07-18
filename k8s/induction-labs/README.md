# GCR Artifact Registry Setup

## Create service account
gcloud iam service-accounts create k8s-artifact-reader \
    --display-name "Kubernetes Artifact Registry Reader"

## Grant Artifact Registry Reader role
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:k8s-artifact-reader@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"

## Create and download key
gcloud iam service-accounts keys create key.json \
    --iam-account=k8s-artifact-reader@PROJECT_ID.iam.gserviceaccount.com



## Create docker registry secret
```
kubectl create secret docker-registry artifact-registry-secret \
    --docker-server=us-central1-docker.pkg.dev \
    --docker-username=_json_key \
    --docker-password="$(cat key.json)" \
    --docker-email=your-email@example.com
```



# GCR Docker Secret

```sh
set PROJECT_ID "induction-labs"

set BUCKET_NAME "induction-labs"

set SERVICE_ACCOUNT_NAME "k8s-bucket-access-sa"

gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
            --description="Service account for k8s bucket access" \
            --display-name="K8s Bucket Access Service Account"


gcloud iam service-accounts keys create ./secrets/gcp_service_account_key.json \
            --iam-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com

gsutil iam ch serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$BUCKET_NAME

gsutil iam ch serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com:storageObjectViewer gs://$BUCKET_NAME
```
