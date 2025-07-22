for i in {1..48}; do
    export SEED=$i
    if (( i % 2 == 0 )); then
        export WARMUP_STEPS=200
    else
        export WARMUP_STEPS=0
    fi

    mdl export modeling.experiments.action_instruct.qwen_25o_test.Qwen25OActionExperimentConfig_GPU --submit
done