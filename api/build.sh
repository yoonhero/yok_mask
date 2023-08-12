#!/usr/local/bin/zsh

echo "building start!"

docker build . --tag yok2vec
docker tag yok2vec:latest ghcr.io/yoonhero/yok2vec:latest
docker push ghcr.io/yoonhero/yok2vec:latest