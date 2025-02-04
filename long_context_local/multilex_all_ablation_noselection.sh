summary_types=( "tiny" "short" "long" )

for summary in ${summary_types[@]}; do
    if [ ! -d "answers_multilex_replication/primera_$summary" ]; then
        python primera_ablation_replication_noselection.py -d $summary -i 1 -m "primera_$summary" &
    fi
    if [ ! -d "answers_multilex_replication/led_$summary" ]; then
        python primera_ablation_replication_noselection.py -d $summary -i 1 -m "led_$summary" &
    fi
done
wait