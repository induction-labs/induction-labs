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


# https://depot.dev/orgs/xzsqsmmrvp/projects/v2tbx2d1w1/settings

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


https://tailscale.com/kb/1236/kubernetes-operator#helm
# Tailscale Setup
[tailscale oauth](https://login.tailscale.com/admin/settings/oauth)
Need auth_keys WRITE, devices:core WRITE, set tag to `tag:k8s-operator`

```sh
helm repo add tailscale https://pkgs.tailscale.com/helmcharts
helm repo update

set OAUTH_CLIENT_ID "<OAuth client ID>"
set OAUTH_CLIENT_SECRET "<OAuth client secret>"
helm upgrade \
  --install \
  tailscale-operator \
  tailscale/tailscale-operator \
  --namespace=tailscale \
  --create-namespace \
  --set-string oauth.clientId="$OAUTH_CLIENT_ID" \
  --set-string oauth.clientSecret="$OAUTH_CLIENT_SECRET" \
  --wait
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
kubectl exec -n induction-labs -it modeling-7fc4bd8965-p2bzz -- /bin/bash

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