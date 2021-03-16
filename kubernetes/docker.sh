DIR=$(dirname $0)

#docker system prune

# Docker build
#docker build -t ajliang/dse-experiments .
docker build -t ajliang/dse-experiments -f $DIR/Dockerfile $DIR/..

docker push ajliang/dse-experiments
