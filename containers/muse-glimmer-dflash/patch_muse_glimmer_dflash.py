from pathlib import Path


SPECULATIVE_CONFIG = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py"
)
MODEL_INTERFACES = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/interfaces.py"
)
QWEN3_DFLASH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
    "qwen3_dflash.py"
)
DFLASH_UTILS = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/"
    "dflash/utils.py"
)

BEFORE = """                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
"""

AFTER = """                    if (
                        isinstance(
                            self.draft_model_config.hf_config,
                            (EAGLEConfig, SpeculatorsConfig),
                        )
                        or "MuseGlimmerAssistantModel"
                        in self.draft_model_config.architectures
                    ):
                        # Muse Glimmer already ships a native DFlash config and
                        # has a dedicated registry alias. Wrapping it as a
                        # generic EAGLE config invents an unsupported
                        # DFlashMuseGlimmerAssistantModel architecture.
                        if "MuseGlimmerAssistantModel" in (
                            self.draft_model_config.architectures
                        ):
                            target_text_config = get_hf_text_config(
                                self.target_model_config.hf_config
                            )
                            draft_hf_config = self.draft_model_config.hf_config
                            # vLLM currently aliases the assistant config to
                            # Qwen3Config. That class replaces the checkpoint's
                            # sliding window with None and supplies its own
                            # default vocabulary size, so restore both from the
                            # target model that trained this assistant.
                            draft_hf_config.sliding_window = (
                                target_text_config.sliding_window
                            )
                            draft_hf_config.vocab_size = target_text_config.vocab_size
                            draft_hf_config.dflash_config = {
                                "mask_token_id": draft_hf_config.mask_token_id,
                                "target_layer_ids": draft_hf_config.target_layer_ids,
                                "swa_window_size": target_text_config.sliding_window,
                            }
                        pass
                    else:
"""

source = SPECULATIVE_CONFIG.read_text()
matches = source.count(BEFORE)
if matches != 1:
    raise RuntimeError(
        f"Expected one vLLM patch target in {SPECULATIVE_CONFIG}, found {matches}"
    )

SPECULATIVE_CONFIG.write_text(source.replace(BEFORE, AFTER))

INTERFACE_BEFORE = """        assert hasattr(parent_ref, "model"), (
            "Model instance must have 'model' attribute to set number of layers"
        )
        assert isinstance(parent_ref.model, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        parent_ref.model._set_aux_hidden_state_layers(layers)
"""

INTERFACE_AFTER = """        eagle_model = (
            parent_ref
            if isinstance(parent_ref, EagleModelMixin)
            else getattr(parent_ref, "model", None)
        )
        assert isinstance(eagle_model, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        eagle_model._set_aux_hidden_state_layers(layers)
"""

interface_source = MODEL_INTERFACES.read_text()
interface_matches = interface_source.count(INTERFACE_BEFORE)
if interface_matches != 1:
    raise RuntimeError(
        f"Expected one vLLM interface patch target in {MODEL_INTERFACES}, "
        f"found {interface_matches}"
    )

MODEL_INTERFACES.write_text(
    interface_source.replace(INTERFACE_BEFORE, INTERFACE_AFTER)
)

WEIGHTS_BEFORE = """        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
"""

WEIGHTS_AFTER = """        is_muse_glimmer = "MuseGlimmerAssistantModel" in (
            self.draft_model_config.architectures
        )
        for name, loaded_weight in weights:
            if is_muse_glimmer:
                # Muse's assistant exporter keeps the projection and input
                # normalization under an encoder prefix. The native vLLM
                # DFlash module names them fc and hidden_norm.
                if name == "encoder.fc.weight":
                    name = "fc.weight"
                elif name == "encoder.output_norm_enc.weight":
                    name = "hidden_norm.weight"
            assert "mask_hidden" not in name, (
"""

weights_source = QWEN3_DFLASH.read_text()
weights_matches = weights_source.count(WEIGHTS_BEFORE)
if weights_matches != 1:
    raise RuntimeError(
        f"Expected one vLLM weights patch target in {QWEN3_DFLASH}, "
        f"found {weights_matches}"
    )

QWEN3_DFLASH.write_text(weights_source.replace(WEIGHTS_BEFORE, WEIGHTS_AFTER))

TARGET_INNER_BEFORE = """    target_inner = target_language_model.model
    draft_inner = dflash_model.model
"""

TARGET_INNER_AFTER = """    target_inner = getattr(
        target_language_model, "model", target_language_model
    )
    draft_inner = dflash_model.model
"""

utils_source = DFLASH_UTILS.read_text()
utils_matches = utils_source.count(TARGET_INNER_BEFORE)
if utils_matches != 1:
    raise RuntimeError(
        f"Expected one vLLM DFlash utility patch target in {DFLASH_UTILS}, "
        f"found {utils_matches}"
    )

DFLASH_UTILS.write_text(
    utils_source.replace(TARGET_INNER_BEFORE, TARGET_INNER_AFTER)
)
