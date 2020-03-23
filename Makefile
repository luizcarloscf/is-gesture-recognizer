VERSION := 0.0.5
USER := luizcarloscf
IMAGE := is-gesture-recognizer
PYTHON := python3
PWD := ${CURDIR}

.PHONY: help, clean, login, build, push, proto

help:
	@ echo "Usage:\n"
	@ echo "make clean-pyc     		Remove python files artifacts."
	@ echo "make clean-docker  		Clean all stopped containers and build cache."
	@ echo "make clean         		clean-pyc and clean-docker."
	@ echo "make build-dev          Build docker image for development."
	@ echo "make build         		Build docker image."
	@ echo "make push          		Push docker image to dockerhub."
	@ echo "make login	       		Login on docker (necessary to push image)."
	@ echo "make lint          		Check style with flake8."

clean-pyc:
	@ find . -name '*.pyc' -exec rm -f {} +
	@ find . -name '*.pyo' -exec rm -f {} +
	@ find . -name '*~' -exec rm -f {} +
	@ find . -name '__pycache__' -exec rm -fr {} +

clean-docker:
	@ docker system prune

clean: clean-pyc clean-docker

build-dev:
	docker build -f etc/docker/Dockerfile.dev -t $(IMAGE)/dev .

build:
	docker build -f etc/docker/Dockerfile -t $(USER)/$(IMAGE):$(VERSION) .

push:
	docker push $(USER)/$(IMAGE):$(VERSION)

login:
	docker login

proto:
	@ find . -name 'options_pb2.py' -exec rm -fr {} +
	@ docker run --rm -v $(PWD):$(PWD) -w $(PWD) luizcarloscf/docker-protobuf:master \
												--python_out=./src/is_gesture_recognizer \
												-I./src/conf/ options.proto
	@ echo "src/is-gesture/ptions_pb2.py file successfully written"