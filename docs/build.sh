#!/bin/bash

set -e

source /tmp/virtdocs/bin/activate
cd source && make html 2>&1 | grep -v "WARNING" && \
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "\033[32mDocumentation build successful! Path to the generated file: \
\033]8;;file://$(pwd)/_build/html/index.html\033\\source/_build/html/index.html\033]8;;\033\\"
        echo -e "\033[0m"  # Reset the color back to default
    else
        echo "Documentation build failed!"
    fi
