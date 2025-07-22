for i in {1..48}; do
    export SEED=$i
    export PEAK_LR="1e-3"
    export END_LR="1e-5"

    if (( i % 3 == 0 )); then
        export WARMUP_STEPS=0
        export END_LR="1e-3"
    elif (( i % 3 == 1 )); then
        export WARMUP_STEPS=200
    else
        export WARMUP_STEPS=0
    fi

    mdl export modeling.experiments.action_instruct.qwen_25o_test.Qwen25OActionExperimentConfig_GPU --submit
done