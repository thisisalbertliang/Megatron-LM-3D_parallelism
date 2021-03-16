DIR=$(dirname $0)

#docker system prune

# Docker build
#docker build -t ajliang/dse-experiments .
docker build -t ajliang/dse-experiments -f $DIR/Dockerfile $DIR/..

docker push ajliang/dse-experiments

kubectl delete deploy dse-experiments
kubectl create -f dse-experiments-deployment.yaml

sleep 5
kubectl get pods