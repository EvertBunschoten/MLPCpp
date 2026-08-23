


FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ccache \
    python3

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY runTests.sh /runTests.sh

ENTRYPOINT ["/runTests.sh"]
