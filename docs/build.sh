#!/bin/bash

set -e

source /tmp/virtdocs/bin/activate
cd source && make html 2>&1 | grep -v "WARNING" && \
    [ ${PIPESTATUS[0]} -eq 0 ] && echo "Documentation build successful!" || \
    echo "Documentation build failed!"
