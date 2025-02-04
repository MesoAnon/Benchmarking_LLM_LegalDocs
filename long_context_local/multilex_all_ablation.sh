summary_types=( "tiny" "short" "long" )

for summary in ${summary_types[@]}; do
    if [ ! -d "answers_multilex/primera_$summary" ]; then
        python primera_ablation_all.py -d $summary -i 0 -m "primera_$summary" &
    fi
    if [ ! -d "answers_multilex/led_$summary" ]; then
        python primera_ablation_all.py -d $summary -i 1 -m "led_$summary" &
    fi
done
wait