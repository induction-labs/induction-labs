
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
kubectl apply -f k8s/induction-labs/cluster-queue.yaml
kubectl apply -f k8s/induction-labs/local-queue.yaml

kubectl apply -f k8s/induction-labs/flavours/gpu.yaml
kubectl apply -f k8s/induction-labs/flavours/cpu.yaml
```


```sh
# Submit job
kubectl create -f k8s/induction-labs/mdl.yaml


# Cluster Queue
kubectl get clusterqueue cluster-queue -o yaml


# Local Queue
kubectl get localqueue local-queue -o yaml


# Workloads
kubectl get workloads

# View Job
kubectl describe workload job-modeling-dw6sw-eed43 
```

