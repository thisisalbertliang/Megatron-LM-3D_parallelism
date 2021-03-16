#kubectl delete pytorchjobs megatron-pytorch
#kubectl create -f pytorch-operator.yaml
#kubectl get pods -l pytorch-job-name=megatron-pytorch -o wide

#kubectl delete mpijob dse-experiments-fixed-global-bsz-32
#kubectl create -f albert_yaml_files/fixed_global_batch_size_experiments/345M_model/dp8-mp1-pp1.yaml

#kubectl delete mpijob dse-experiments
#kubectl create -f mpi-operator0.yaml

kubectl delete deploy dse-experiments
kubectl create -f dse-experiments-deployment.yaml

sleep 10
kubectl get pods
