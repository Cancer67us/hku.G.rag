#!/bin/bash
URL=boris_dotv
IMAGE_NAME=dotv_RAG
VERSION=0.1
docker build -t $URL/$IMAGE_NAME:$VERSION .