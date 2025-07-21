from typing import cast

from synapse.utils.logging import configure_logging, logging

logger = configure_logging(__name__, level=logging.INFO)


def load_kubernetes_config(context: str | None = None) -> None:
    from kubernetes import config as k8s_config

    # Load kubernetes config (from ~/.kube/config or in-cluster config)
    contexts, active_context = k8s_config.list_kube_config_contexts()
    if context:
        # Verify the context exists
        context_names = [cast(dict, ctx)["name"] for ctx in contexts]
        assert context in context_names, (
            f"Context '{context}' not found. Available contexts: {context_names}"
        )
        k8s_config.load_kube_config(context=context)
        logger.debug(f"Loaded kubernetes config with context: {context}")
    else:
        k8s_config.load_kube_config(context=active_context["name"])
        logger.debug(f"Loaded kubernetes config with context {active_context['name']}")
