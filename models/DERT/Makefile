

install:
	@pip3 install --default-timeout=900 -r requirements.txt

clean:
	@rm -rf src/__pycache__
	@rm -rf src/models/__pycache__
	@rm -rf src/predictors/__pycache__
	@rm -rf src/utils/__pycache__

docker:
	@docker build . -t detr

docker_run:
	@docker run -v ${v}:/opt/Detr -v ${n}:${n} -v ${t}:${t} --rm detr -n ${n} -t ${t}

UNAME := $(shell uname)
#.PHONY: sample
test:

	@if [ "Darwin" = $(UNAME) ]; then\
        ./src/client/client_darwin -p=${p} -c=${c};\
    fi
	@if [ "Linux" = $(UNAME) ]; then\
        ./src/client/client_linux -p=${p} -c=${c};\
    fi