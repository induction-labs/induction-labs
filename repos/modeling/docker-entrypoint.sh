#!/bin/bash
# shellcheck disable=SC1091
source ./.devenv/load-exports 


# Execute the command passed to docker run
exec "$@"