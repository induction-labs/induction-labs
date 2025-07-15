from __future__ import annotations

import logging
from contextlib import contextmanager

from modeling.modules.base_module import cast
import ray
import ray.scripts.scripts as ray_cli
from pydantic import AnyUrl, UrlConstraints
from synapse.utils.logging import configure_logging

from modeling.utils.temp_env import temp_env
import json

logger = configure_logging(__name__, level=logging.DEBUG)


# This is really stupid - if we try to start ray with the default LD_LIBRARY_PATH
# that is set by uv managed by devenv, then we get /home/ubuntu/documents/induction-labs/repos/modeling/.devenv/state/venv/lib/python3.12/site-packages/ray/core/src/ray/gcs/gcs_server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.36' not found (required by /nix/store/7n3q3rgy5382di7ccrh3r6gk2xp51dh7-gcc-14.2.1.20250322-lib/lib/libstdc++.so.6)
# because it tries to use the glibc from the venv, which is not compatible with the system glibc.
# So we need to set the LD_LIBRARY_PATH to the system glibc and gcc libraries
# I'm not sure why other libraries like numpy and torch work with nix LD_LIBRARY_PATH but ray doesn't,
# I think it is related to ray starting subprocesses that need to link against the system glibc and gcc libraries.


HOST = "localhost"
PORT = 6379


class RayUrl(AnyUrl):
    """ """

    host_required = True

    @property
    def host(self) -> str:
        """The required URL host."""
        return cast(str, self._url.host)  # pyright: ignore[reportAttributeAccessIssue]

    _constraints = UrlConstraints(allowed_schemes=["ray"])


@contextmanager
def initialize_ray_head(resources: dict[str, float] | None = None):
    logger.debug("Initializing Ray head node...")
    assert not ray.is_initialized(), "Ray is already initialized."

    # resolved_host = services.resolve_ip_for_localhost(HOST)
    # Scheme here is whatever for now idk
    ray_host = RayUrl.build(scheme="ray", host=HOST, port=PORT)

    try:
        with temp_env(
            {
                "LD_LIBRARY_PATH": None,
                # TODO(logging): Ray logs are chanelled through `from ray.autoscaler._private.cli_logger import cli_logger`
                # Put these in seperate logs.
                # Also, capture the stdout of the `ray_cli.start` command and put those in seperate logs so we can
                # always enable RAY_LOG_TO_STDERR
                # "RAY_LOG_TO_STDERR": "1",
                # Set CUDA_VISIBLE_DEVICES to empty string to avoid ray trying to use GPUs on the head node.
                # "CUDA_VISIBLE_DEVICES": "",
            }
        ):
            logger.debug(f"Starting Ray head node at {ray_host} with {resources=}")
            ray_cli.start.main(
                # URL interpolation is so troll
                [
                    "--head",
                    # TODO: Setting address on head node programatically does not work right now.
                    # TODO: Read address from ray_cli return
                    # f"--address={ray_host.host}:{ray_host.port}",
                    f"--port={str(ray_host.port)}",
                    f"--resources={json.dumps(resources)}" if resources else "",
                    # TODO: ray dashboard configuration
                    "--dashboard-host=0.0.0.0",
                ],
                prog_name="ray",
                # Otherwise click will os.exit on completion
                standalone_mode=False,
            )
        ray.init()
        assert ray.is_initialized(), (
            "Ray failed to initialize. Please check the logs for more details."
        )
        logger.debug(f"Ray head node finished initializing at {ray_host}")
        yield ray_host
    except Exception as e:
        logger.error(f"Failed to initialize Ray head node: {e}")
        raise e
    finally:
        if ray.is_initialized():
            ray.shutdown()
        else:
            logger.warning("Ray was not initialized, nothing to shutdown.")
        # Ensure the ray processes are cleaned up
        try:
            ray_cli.stop.main(
                # Run with empty args, else will be run with sys.argv
                args=[],
                prog_name="ray",
                standalone_mode=False,
            )
        except Exception as e:
            logger.error(f"Failed to stop Ray head node: {e}")
        logger.info("Ray head node shutdown complete.")
