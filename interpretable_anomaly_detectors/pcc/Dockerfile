FROM registry.gitlab.hpi.de/akita/i/pyod:0.2.5

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"


# install algorithm dependencies
COPY requirements.txt .
RUN pip install git+https://github.com/thu-ml/zhusuan.git
RUN pip install git+https://github.com/haowen-xu/tfsnippet.git@v0.1.2
RUN pip install -r ./requirements.txt


COPY algorithm.py .
