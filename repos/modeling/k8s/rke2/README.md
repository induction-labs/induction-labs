
# NVIDIA

## On the GPU node
```sh
sudo systemctl stop k3s-agent
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


## Taints

```sh
kubectl taint nodes gpu1 nvidia.com/gpu=present:NoSchedule
kubectl get nodes gpu1 -o yaml | grep -C 2 taint
```
