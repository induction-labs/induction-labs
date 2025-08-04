# Grafana setup
```sh
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring
helm install kp-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword='' \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```



```sh
kubectl -n monitoring port-forward svc/kp-stack-grafana 3000:80
```