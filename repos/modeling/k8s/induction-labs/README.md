# K8s Setup


## Hyperbolic

```bash
set CLUSTER_SERVER "https://cluster-endpoint:6443"

set CONTEXT_NAME "hyperbolic"
set CLUSTER_NAME "$CONTEXT_NAME-cluster"
# These should start with LS and end with = Note these are not the same
set CLUSTER_CERTIFICATE_DATA "LS0tLS1CRUdJTi..."
set CLIENT_CERTIFICATE_DATA "LS0tLS1CRUdJTi..."
set CLIENT_KEY_DATA "LS0tLS1CRUdJTi..."

set CLUSTER_USERNAME "$CONTEXT_NAME-user"

kubectl config set-cluster "$CLUSTER_NAME" \
    --server="$CLUSTER_SERVER"
kubectl config set "clusters.$CLUSTER_NAME".certificate-authority-data "$CLUSTER_CERTIFICATE_DATA"

kubectl config set-credentials "$CLUSTER_USERNAME" 


kubectl config set "users.$CLUSTER_USERNAME".client-certificate-data "$CLIENT_CERTIFICATE_DATA"
kubectl config set "users.$CLUSTER_USERNAME".client-key-data "$CLIENT_KEY_DATA"

kubectl config set-context "$CONTEXT_NAME" \
    --cluster="$CLUSTER_NAME" \
    --user="$CLUSTER_USERNAME"

kubectl config set "contexts.$CONTEXT_NAME.namespace" induction-labs
kubectl config get-contexts

kubectl config use-context "$CONTEXT_NAME"

kubectl cluster-info
```


# Container Registry Setup
```sh
set DEPOT_KEY "depot_project_..."
kubectl create secret docker-registry depot-registry-secret \
    --namespace induction-labs \
    --docker-server=registry.depot.dev \
    --docker-username=x-token \
    --docker-password="$DEPOT_KEY"
kubectl get secrets
```


# Kueue Setup
https://kueue.sigs.k8s.io/docs/installation/#before-you-begin
```sh
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.12.4 \
  --namespace  kueue-system \
  --create-namespace \
  --wait --timeout 300s
```



<!-- ! TODO: Package this config with helm -->

# Cluster config: 
```sh
kubectl create namespace induction-labs

kubectl apply -f cluster-queue.yaml
kubectl apply -f local-queue.yaml

kubectl apply -f flavours/gpu.yaml
kubectl apply -f flavours/cpu.yaml
```




# Job submission
```sh
# Submit job
kubectl create -f mdl.yaml

# Logs
kubectl get jobs
# Successful jobs
# Server side
kubectl get jobs --field-selector=status.successful=1 -o name
kubectl get jobs -o jsonpath='{range .items[?(@.status.failed>0)].metadata}{.name}{"\n"}{end}'

# Running jobs
kubectl get jobs -o jsonpath='{range .items[?(@.status.active>0)].metadata}{.name}{"\n"}{end}'
# Queued jobs
kubectl get jobs -o jsonpath='{range .items[?(@.spec.suspend==true)].metadata}{.name}{"\n"}{end}'


# Delete failed jobs
kubectl get jobs -o jsonpath='{range .items[?(@.status.failed>0)].metadata}{.name}{"\n"}{end}' |
        xargs -r kubectl delete job

kubectl logs -f job/modeling-48trw
kubectl get job modeling-48trw -o yaml

# Get pods
kubectl get pods -l job-name=modeling-48trw

# Cluster Queue
kubectl get clusterqueue cluster-queue -o yaml


# Local Queue
kubectl get localqueue local-queue -o yaml


# Workloads
kubectl get workloads

# View Job
kubectl describe workload job-modeling-dw6sw-eed43 
```









# OLD: 
```sh
kubectl config set-context --current --namespace=induction-labs
kubectl logs -f induction-labs/mdl.yaml
```

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
kubectl create secret docker-registry gcr-artifact-secret \
    --docker-server=us-central1-docker.pkg.dev \
    --docker-username=_json_key \
    --docker-password="$(cat key.json)" \
    --docker-email=jeffrey@inductionlabs.com \
    --namespace=induction-labs
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
